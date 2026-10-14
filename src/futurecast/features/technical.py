"""Group D: Technical & Price Geometry features (strictly backward-looking)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vwap(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Compute rolling Volume-Weighted Average Price."""
    pv = close * volume
    cum_pv = pv.rolling(window, min_periods=max(window // 4, 10)).sum()
    cum_vol = volume.rolling(window, min_periods=max(window // 4, 10)).sum()
    return cum_pv / cum_vol.replace(0.0, np.nan)


def compute_technical_features(
    df: pd.DataFrame,
    target_symbol: str = "DOGEUSDT",
    extremes_window: int = 60,
    vwap_window: int = 240,  # 4 hours
) -> pd.DataFrame:
    """Compute Group D technical and price geometry features."""
    out = pd.DataFrame(index=df.index)

    o = df[f"{target_symbol}_open"]
    h = df[f"{target_symbol}_high"]
    l = df[f"{target_symbol}_low"]
    c = df[f"{target_symbol}_close"]
    v = df[f"{target_symbol}_volume"]

    # 1. Distance to Rolling Extremes (Stochastic %K equivalent over 60m)
    roll_min = l.rolling(extremes_window, min_periods=10).min()
    roll_max = h.rolling(extremes_window, min_periods=10).max()
    denom = (roll_max - roll_min).replace(0.0, np.nan)
    out["dist_to_extremes_60m"] = ((c - roll_min) / denom).fillna(0.5)

    # 2. VWAP Basis (spread between Close and rolling 4h VWAP in basis points)
    vwap_4h = compute_vwap(c, v, window=vwap_window)
    out["vwap_basis_4h_bps"] = (((c - vwap_4h) / vwap_4h.replace(0.0, np.nan)) * 10000.0).fillna(0.0)

    # 3. Intra-Candle Wick and Body Geometry
    candle_range = (h - l).replace(0.0, np.nan)
    body_top = np.maximum(o, c)
    body_bottom = np.minimum(o, c)

    out["upper_wick_ratio"] = (h - body_top) / candle_range
    out["lower_wick_ratio"] = (body_bottom - l) / candle_range
    out["body_to_range_ratio"] = (body_top - body_bottom) / candle_range

    # Fill 0-range candles (where high == low)
    out["upper_wick_ratio"] = out["upper_wick_ratio"].fillna(0.0)
    out["lower_wick_ratio"] = out["lower_wick_ratio"].fillna(0.0)
    out["body_to_range_ratio"] = out["body_to_range_ratio"].fillna(0.0)

    return out
