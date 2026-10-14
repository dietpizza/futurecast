"""Parquet storage manager and continuous cross-asset time alignment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ParquetStorage:
    """Manages raw and processed time-series storage in Apache Parquet."""

    def __init__(
        self,
        base_dir: str | Path = "data",
        compression: str = "zstd",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.compression = compression

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def save_raw(
        self,
        df: pd.DataFrame,
        symbol: str,
        category: str = "klines",
    ) -> Path:
        """Partition by year and month and save as Parquet."""
        if df.empty:
            raise ValueError(f"Cannot save empty DataFrame for {symbol}")

        df = df.copy()
        if "timestamp" not in df.columns:
            raise KeyError("DataFrame must have 'timestamp' column.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month

        target_dir = self.raw_dir / category / f"symbol={symbol.upper()}"
        target_dir.mkdir(parents=True, exist_ok=True)

        file_path = target_dir / "data.parquet"
        df.to_parquet(
            file_path,
            index=False,
            compression=self.compression,
            partition_cols=["year", "month"],
        )
        logger.info(f"Saved {len(df)} records for {symbol} ({category}) to {target_dir}")
        return target_dir

    def load_raw(
        self,
        symbol: str,
        category: str = "klines",
    ) -> pd.DataFrame:
        """Load all partitioned Parquet files for a given symbol and category."""
        target_dir = self.raw_dir / category / f"symbol={symbol.upper()}"
        if not target_dir.exists():
            logger.warning(f"No data found at {target_dir}")
            return pd.DataFrame()

        df = pd.read_parquet(target_dir)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    def save_processed(self, df: pd.DataFrame, name: str = "feature_matrix") -> Path:
        """Save aligned feature matrix to processed storage."""
        file_path = self.processed_dir / f"{name}.parquet"
        df.to_parquet(file_path, index=True, compression=self.compression)
        logger.info(f"Saved processed feature matrix ({df.shape}) to {file_path}")
        return file_path

    def load_processed(self, name: str = "feature_matrix") -> pd.DataFrame:
        """Load processed feature matrix."""
        file_path = self.processed_dir / f"{name}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Processed file not found: {file_path}")
        return pd.read_parquet(file_path)


def align_cross_asset_data(
    target_df: pd.DataFrame,
    benchmark_dfs: dict[str, pd.DataFrame],
    max_gap_ffill_minutes: int = 3,
) -> pd.DataFrame:
    """Align target and benchmark assets onto a continuous master UTC 1-minute grid.
    
    1. Creates master date_range index spanning common min/max.
    2. Reindexes each asset.
    3. Forward-fills isolated gaps <= max_gap_ffill_minutes.
    4. Drops bars where continuous feed was interrupted for > max_gap_ffill_minutes.
    """
    if target_df.empty:
        raise ValueError("Target DataFrame is empty")

    target_symbol = target_df["symbol"].iloc[0].upper()
    all_dfs = {target_symbol: target_df}
    all_dfs.update({k.upper(): v for k, v in benchmark_dfs.items()})

    # Find earliest start and latest end across all series
    min_time = max(df["timestamp"].min() for df in all_dfs.values())
    max_time = min(df["timestamp"].max() for df in all_dfs.values())

    if min_time > max_time:
        raise ValueError(f"No overlapping time range found across assets (min: {min_time}, max: {max_time})")

    master_index = pd.date_range(start=min_time, end=max_time, freq="1min", tz="UTC", name="timestamp")

    merged = pd.DataFrame(index=master_index)

    for symbol, df in all_dfs.items():
        sub = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()
        sub = sub.reindex(master_index)

        # Detect gap lengths
        is_missing = sub["close"].isna()
        gap_groups = (~is_missing).cumsum()
        gap_lengths = is_missing.groupby(gap_groups).transform("sum")

        # Forward-fill prices up to limit
        price_cols = [c for c in ["open", "high", "low", "close", "funding_rate", "open_interest"] if c in sub.columns]
        sub[price_cols] = sub[price_cols].ffill(limit=max_gap_ffill_minutes)

        # Volumes for filled bars set to 0.0
        vol_cols = [c for c in ["volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"] if c in sub.columns]
        sub[vol_cols] = sub[vol_cols].fillna(0.0)

        # Prefix columns with symbol
        sub = sub.add_prefix(f"{symbol}_")
        merged = merged.join(sub)

        # Add downtime flag
        merged[f"{symbol}_gap_excessive"] = gap_lengths > max_gap_ffill_minutes

    # Drop any remaining unfillable rows (e.g., initial NaNs or prolonged exchange downtime)
    excessive_gap_cols = [c for c in merged.columns if c.endswith("_gap_excessive")]
    is_valid = ~merged[excessive_gap_cols].any(axis=1)
    
    # Check that core price columns have no NaNs
    core_close_cols = [f"{s}_close" for s in all_dfs.keys()]
    is_valid = is_valid & merged[core_close_cols].notna().all(axis=1)

    aligned = merged.loc[is_valid].drop(columns=excessive_gap_cols)
    logger.info(f"Aligned cross-asset matrix: {len(aligned)} continuous 1-minute bars.")
    return aligned
