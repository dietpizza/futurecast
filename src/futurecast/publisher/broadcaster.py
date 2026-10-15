"""Forecast audit log: every forecast the engine emits is appended to a JSONL file.

The folder never holds more than `max_mb` of logs. Files are named
forecasts_<UTC date>.jsonl, then forecasts_<UTC date>_2.jsonl and so on: a file is closed
once it reaches a quarter of the limit or the day changes. Before a new file starts, the
oldest files are deleted until the new one fits, so the newest forecasts are always kept.

Every coin's engine has its own broadcaster writing to the same folder from its own thread,
so writes and clean-ups are serialised per folder.

Strictly zero trade execution logic.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Dict, List, Tuple

from futurecast.paths import resolve
from futurecast.schema import ForecastEvent

_PARTS_PER_LIMIT = 4
_folder_locks: Dict[Path, threading.Lock] = {}
_folder_locks_guard = threading.Lock()


def _folder_lock(folder: Path) -> threading.Lock:
    with _folder_locks_guard:
        return _folder_locks.setdefault(folder, threading.Lock())


def _order(path: Path) -> Tuple[str, int]:
    """(date, part) from forecasts_<date>[_<part>].jsonl, so part 10 sorts after part 9."""
    date, _, part = path.stem.removeprefix("forecasts_").partition("_")
    return date, int(part) if part.isdigit() else 1


class SignalBroadcaster:
    """Appends real-time forecasts to JSONL audit logs, keeping the folder under `max_mb`."""

    def __init__(self, output_dir: str | Path = "logs/forecasts", max_mb: float = 32.0) -> None:
        self.output_dir = resolve(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.part_bytes = max(self.max_bytes // _PARTS_PER_LIMIT, 1)
        self._lock = _folder_lock(self.output_dir)
        with self._lock:
            self._prune(room=0)

    def _files(self) -> List[Path]:
        return sorted(self.output_dir.glob("forecasts_*.jsonl"), key=_order)

    def _prune(self, room: int) -> None:
        """Delete the oldest logs until `room` more bytes fit under the limit."""
        files = self._files()
        sizes = [p.stat().st_size for p in files]
        total = sum(sizes)
        for path, size in zip(files, sizes):
            if total + room <= self.max_bytes:
                break
            path.unlink(missing_ok=True)
            total -= size

    def _file_for(self, date_str: str, incoming: int) -> Path:
        """Today's open log file, or a fresh one (after making room) when it is full or missing."""
        today = [p for p in self._files() if _order(p)[0] == date_str]
        path = today[-1] if today else self.output_dir / f"forecasts_{date_str}.jsonl"
        size = path.stat().st_size if path.exists() else 0
        if size and size + incoming > self.part_bytes:
            path = self.output_dir / f"forecasts_{date_str}_{_order(path)[1] + 1}.jsonl"
            size = 0
        if size == 0:
            self._prune(room=self.part_bytes)
        return path

    def broadcast(self, event: ForecastEvent) -> None:
        """Append the forecast event to the audit log for its UTC date."""
        assert event.execution_disabled is True, "Security violation: execution must be disabled"

        line = (event.model_dump_json() + "\n").encode("utf-8")
        with self._lock:
            with open(self._file_for(event.timestamp.strftime("%Y-%m-%d"), len(line)), "ab") as f:
                f.write(line)
