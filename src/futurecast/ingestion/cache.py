"""Disk caches of downloaded closed candles and settled funding rates."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
from typing import ClassVar, Dict, Optional, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from futurecast.paths import resolve

logger = logging.getLogger(__name__)


class CandleCache:
    """Closed candles already downloaded, one Parquet file per Binance market and interval.

    Closed candles never change, so each file holds one span of open times that was fully
    downloaded: every candle Binance has between `covered_from` and `covered_to` is in the
    file (an open time missing inside the span is an exchange gap, not a hole in the cache).
    A `covered_from` of 0 means the span reaches back to the listing. Downloads that join the
    span extend it; a newer download that cannot join replaces it. Files keep the newest
    `max_rows` candles.
    """

    COLUMNS = ["time", "open", "high", "low", "close", "volume", "quote_volume",
               "count", "taker_buy_volume", "taker_buy_quote_volume"]

    # One lock per file for the whole process: the stream loop and history downloads each use their own client
    _locks: ClassVar[Dict[Path, threading.Lock]] = {}
    _locks_guard: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, cache_dir: str | Path = "cache/candles", max_rows: int = 150_000) -> None:
        self.cache_dir = resolve(cache_dir)
        self.max_rows = max_rows

    @classmethod
    def empty(cls) -> pd.DataFrame:
        return pd.DataFrame({c: pd.Series(dtype="int64" if c in ("time", "count") else float) for c in cls.COLUMNS})

    def _path(self, market: str, interval: str) -> Path:
        return self.cache_dir / f"{market.upper()}_{interval}.parquet"

    def _lock(self, path: Path) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(path, threading.Lock())

    def _read(self, path: Path) -> Tuple[pd.DataFrame, Optional[Tuple[int, int]]]:
        if not path.exists():
            return self.empty(), None
        try:
            table = pq.read_table(path)
            meta = table.schema.metadata or {}
            span = (int(meta[b"covered_from"]), int(meta[b"covered_to"]))
            return table.to_pandas()[self.COLUMNS], span
        except Exception as err:
            logger.warning(f"Ignoring unreadable candle cache {path}: {err}")
            return self.empty(), None

    def load(self, market: str, interval: str) -> Tuple[pd.DataFrame, Optional[Tuple[int, int]]]:
        """Cached candles (columns COLUMNS, `time` = open time in ms) and the (from, to) span they cover."""
        path = self._path(market, interval)
        with self._lock(path):
            return self._read(path)

    def store(self, market: str, interval: str, step_ms: int, rows: pd.DataFrame,
              covered_from: int, covered_to: int) -> None:
        """Merge candles downloaded for open times `covered_from`..`covered_to` into the file."""
        path = self._path(market, interval)
        with self._lock(path):
            have, span = self._read(path)
            if span is not None and covered_from <= span[1] + step_ms and span[0] <= covered_to + step_ms:
                rows = pd.concat([have, rows[self.COLUMNS]], ignore_index=True)
                covered_from, covered_to = min(covered_from, span[0]), max(covered_to, span[1])
            elif span is not None and span[1] >= covered_to:
                return  # an older, separate stretch: keep the newer one on disk
            rows = rows[self.COLUMNS].drop_duplicates("time", keep="last").sort_values("time")
            rows = rows[(rows["time"] >= covered_from) & (rows["time"] <= covered_to)]
            if len(rows) > self.max_rows:
                rows = rows.tail(self.max_rows)
                covered_from = int(rows["time"].iloc[0])
            table = pa.Table.from_pandas(rows.reset_index(drop=True), preserve_index=False)
            table = table.replace_schema_metadata({
                **(table.schema.metadata or {}),
                b"covered_from": str(int(covered_from)).encode(),
                b"covered_to": str(int(covered_to)).encode(),
            })
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                pq.write_table(table, tmp, compression="zstd")
                os.replace(tmp, path)  # readers never see a half-written file
            except Exception as err:
                tmp.unlink(missing_ok=True)
                logger.warning(f"Could not write candle cache {path}: {err}")


class FundingCache:
    """Settled funding rates already downloaded, one Parquet file per Binance market.

    A file holds every settlement from `covered_from` (ms) on, as known at `fetched_at` (ms): no
    settlement before `fetched_at` can be missing, so after a restart only the ones since need
    downloading (BinancePublicClient.fetch_funding_rates).
    """

    _locks: ClassVar[Dict[Path, threading.Lock]] = {}
    _locks_guard: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, cache_dir: str | Path = "cache/funding") -> None:
        self.cache_dir = resolve(cache_dir)

    def _path(self, market: str) -> Path:
        return self.cache_dir / f"{market.upper()}.parquet"

    def _lock(self, path: Path) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(path, threading.Lock())

    def load(self, market: str) -> Optional[Tuple[int, int, pd.DataFrame]]:
        """(covered_from, fetched_at, events with columns time and funding_rate), or None."""
        path = self._path(market)
        with self._lock(path):
            if not path.exists():
                return None
            try:
                table = pq.read_table(path)
                meta = table.schema.metadata or {}
                events = table.to_pandas()[["time", "funding_rate"]]
                return int(meta[b"covered_from"]), int(meta[b"fetched_at"]), events
            except Exception as err:
                logger.warning(f"Ignoring unreadable funding cache {path}: {err}")
                return None

    def store(self, market: str, covered_from: int, fetched_at: int, events: pd.DataFrame) -> None:
        path = self._path(market)
        with self._lock(path):
            table = pa.Table.from_pandas(events[["time", "funding_rate"]].reset_index(drop=True), preserve_index=False)
            table = table.replace_schema_metadata({
                **(table.schema.metadata or {}),
                b"covered_from": str(int(covered_from)).encode(),
                b"fetched_at": str(int(fetched_at)).encode(),
            })
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                pq.write_table(table, tmp, compression="zstd")
                os.replace(tmp, path)
            except Exception as err:
                tmp.unlink(missing_ok=True)
                logger.warning(f"Could not write funding cache {path}: {err}")
