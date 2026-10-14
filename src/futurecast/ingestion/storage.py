"""Continuous cross-asset time alignment."""

from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def align_cross_asset_data(
    target_df: pd.DataFrame,
    benchmark_dfs: dict[str, pd.DataFrame],
    max_gap_ffill_minutes: int = 3,
    freq: str = "1min",
) -> pd.DataFrame:
    """Align target and benchmark assets onto a continuous master UTC grid (1-minute by default).

    `max_gap_ffill_minutes` counts bars of `freq`.
    
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

    master_index = pd.date_range(start=min_time, end=max_time, freq=freq, tz="UTC", name="timestamp")

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
