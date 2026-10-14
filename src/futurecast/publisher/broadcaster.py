"""Forecast audit log: every forecast the engine emits is appended to a daily JSONL file.

Strictly zero trade execution logic.
"""

from __future__ import annotations

from pathlib import Path

from futurecast.schema import ForecastEvent


class SignalBroadcaster:
    """Appends real-time forecasts to daily JSONL audit logs."""

    def __init__(
        self,
        output_dir: str | Path = "logs/forecasts",
        render_terminal: bool = False,
    ) -> None:
        # render_terminal is still accepted from existing callers; forecasts are shown in the dashboard only.
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_file(self, date_str: str) -> Path:
        return self.output_dir / f"forecasts_{date_str}.jsonl"

    def broadcast(self, event: ForecastEvent) -> None:
        """Append the forecast event to the audit log for its UTC date."""
        assert event.execution_disabled is True, "Security violation: execution must be disabled"

        date_str = event.timestamp.strftime("%Y-%m-%d")
        with open(self._get_log_file(date_str), "a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
