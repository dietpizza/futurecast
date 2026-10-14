"""Persistent disk cache for downloaded public market data."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """Manages file-based Parquet caching for market data queries in .cache/.
    
    Prevents repeated network downloads of historical immutable financial candles.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        enabled: bool = True,
        compression: str = "zstd",
    ) -> None:
        self.cache_dir = Path(cache_dir)
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
