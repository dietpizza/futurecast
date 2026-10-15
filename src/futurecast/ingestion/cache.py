"""Persistent disk cache for downloaded public market data."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import threading
from typing import Any, ClassVar, Dict, Optional, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from futurecast.paths import resolve

logger = logging.getLogger(__name__)


class DataCache:
    """Manages file-based Parquet caching for market data queries in ~/.futurecast/cache/.
    
    Prevents repeated network downloads of historical immutable financial candles.
    """

    def __init__(
        self,
        cache_dir: str | Path = "cache",
        enabled: bool = True,
        compression: str = "zstd",
    ) -> None:
        self.cache_dir = resolve(cache_dir)
        self.enabled = enabled
        self.compression = compression
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_category_dir(self, category: str) -> Path:
        cat_dir = self.cache_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        return cat_dir

    def _sanitize_key(self, raw_key: str) -> str:
        """Create a filesystem-friendly key, hashing if too long."""
        clean = "".join(c if (c.isalnum() or c in "._-") else "_" for c in raw_key)
        if len(clean) > 120:
            h = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
            clean = f"{clean[:100]}_{h}"
        return clean

    def build_kline_key(
        self,
        symbol: str,
        interval: str,
        start_time_ms: Optional[int],
        end_time_ms: Optional[int],
        limit: int,
    ) -> str:
        s_str = str(start_time_ms) if start_time_ms is not None else "latest"
        e_str = str(end_time_ms) if end_time_ms is not None else "latest"
        return f"{symbol.upper()}_{interval}_{s_str}_{e_str}_{limit}"

    def build_funding_key(
        self,
        symbol: str,
        start_time_ms: Optional[int],
        end_time_ms: Optional[int],
        limit: int,
    ) -> str:
        s_str = str(start_time_ms) if start_time_ms is not None else "latest"
        e_str = str(end_time_ms) if end_time_ms is not None else "latest"
        return f"{symbol.upper()}_funding_{s_str}_{e_str}_{limit}"

    def build_oi_key(
        self,
        symbol: str,
        period: str,
        start_time_ms: Optional[int],
        end_time_ms: Optional[int],
        limit: int,
    ) -> str:
        s_str = str(start_time_ms) if start_time_ms is not None else "latest"
        e_str = str(end_time_ms) if end_time_ms is not None else "latest"
        return f"{symbol.upper()}_oi_{period}_{s_str}_{e_str}_{limit}"

    def get(self, category: str, key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached DataFrame if it exists."""
        if not self.enabled:
            return None

        clean_key = self._sanitize_key(key)
        file_path = self._get_category_dir(category) / f"{clean_key}.parquet"
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                logger.debug(f"Cache hit: {category}/{clean_key} ({len(df)} rows)")
                return df
            except Exception as err:
                logger.warning(f"Error reading cache file {file_path}: {err}")
                return None

        return None

    def put(self, category: str, key: str, df: pd.DataFrame) -> Optional[Path]:
        """Save DataFrame to cache."""
        if not self.enabled or df.empty:
            return None

        clean_key = self._sanitize_key(key)
        file_path = self._get_category_dir(category) / f"{clean_key}.parquet"
        try:
            df.to_parquet(file_path, index=False, compression=self.compression)
            logger.debug(f"Cached {len(df)} rows to {file_path}")
            return file_path
        except Exception as err:
            logger.warning(f"Error saving to cache {file_path}: {err}")
            return None

    def clear(self, category: Optional[str] = None) -> int:
        """Clear cached files. If category is specified, only that category is cleared."""
        if not self.cache_dir.exists():
            return 0

        target_dirs = [self._get_category_dir(category)] if category else list(self.cache_dir.iterdir())
        removed_count = 0
        for d in target_dirs:
            if d.is_dir():
                for p in d.glob("*.parquet"):
                    try:
                        p.unlink()
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove {p}: {e}")

        logger.info(f"Cleared {removed_count} cache files.")
        return removed_count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics: file count and total size in bytes."""
        if not self.cache_dir.exists():
            return {"file_count": 0, "total_bytes": 0, "categories": {}}

        total_bytes = 0
        file_count = 0
        categories: dict[str, int] = {}

        for item in self.cache_dir.iterdir():
            if item.is_dir():
                cat_files = list(item.glob("*.parquet"))
                categories[item.name] = len(cat_files)
                for f in cat_files:
                    file_count += 1
                    total_bytes += f.stat().st_size

        return {
            "cache_dir": str(self.cache_dir.resolve()),
            "file_count": file_count,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "categories": categories,
        }


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

    # One lock per file for the whole process: the stream loop and training each use their own client
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
