"""Funding-rate features (strictly backward-looking)."""

from __future__ import annotations

import pandas as pd


def compute_derivative_features(
    df: pd.DataFrame,
    target_symbol: str = "DOGEUSDT",
    benchmark_symbol: str = "BTCUSDT",
    funding_window_min: int = 4320,  # 3 days, in minutes of clock time (not candles)
) -> pd.DataFrame:
    """Settled funding rate of the target, its deviation from the 3-day mean, and its spread to the benchmark."""
    out = pd.DataFrame(index=df.index)

    # 1. Funding Rate Deviation (target vs its trailing 3-day mean). The window is clock time,
    # so the live candle buffer (LIVE_BARS: ~4 days of 1m candles) holds all of it and reproduces
    # the training value exactly; a 3-day EMA over candles would depend on how much history is loaded.
    if f"{target_symbol}_funding_rate" in df.columns:
        f_target = df[f"{target_symbol}_funding_rate"]
        f_mean = f_target.rolling(f"{funding_window_min}min", min_periods=60).mean()
        out["funding_rate_dev_3d"] = f_target - f_mean
        out["funding_rate_bps"] = f_target * 10000.0
    else:
        out["funding_rate_dev_3d"] = 0.0
        out["funding_rate_bps"] = 0.0

    # 2. Funding spread to the primary benchmark (target - benchmark)
    if f"{target_symbol}_funding_rate" in df.columns and f"{benchmark_symbol}_funding_rate" in df.columns:
        f_bench = df[f"{benchmark_symbol}_funding_rate"]
        out["funding_rate_spread_btc"] = (df[f"{target_symbol}_funding_rate"] - f_bench) * 10000.0
    else:
        out["funding_rate_spread_btc"] = 0.0

    return out
