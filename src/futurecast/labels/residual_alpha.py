"""Target formulation and labeling engine (Residual Alpha & Triple Barrier)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_forward_returns(prices: pd.Series, horizon_minutes: int = 15) -> pd.Series:
    """Compute forward log return: ln(P_{t+h} / P_t) = shift(-h) - ln(P_t)."""
    log_p = np.log(prices.clip(lower=1e-12))
    fwd_ret = log_p.shift(-horizon_minutes) - log_p
    return fwd_ret


def compute_residual_alpha_target(
    aligned_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_symbol: str = "DOGEUSDT",
    benchmark_symbol: str = "BTCUSDT",
    horizon_minutes: int = 15,
    beta_col: str = "beta_btc_1440m",
) -> pd.Series:
    """Compute 15-minute Residual Alpha target:
    
    y_t = R_{DOGE, t -> t+15} - beta_{t, 1440} * R_{BTC, t -> t+15}
    
    CRITICAL: beta_{t, 1440} is point-in-time at time t (backward-looking).
    Only the forward returns look ahead by exactly 15 minutes.
    """
    t_close = aligned_df[f"{target_symbol}_close"]
    b_close = aligned_df[f"{benchmark_symbol}_close"]

    # Forward returns looking ahead by horizon_minutes
    fwd_r_target = compute_forward_returns(t_close, horizon_minutes=horizon_minutes)
    fwd_r_bench = compute_forward_returns(b_close, horizon_minutes=horizon_minutes)

    # Point-in-time rolling beta from feature matrix
    if beta_col in features_df.columns:
        beta = features_df[beta_col]
    else:
        # Fallback if not computed in features
        r_t = np.log(t_close).diff()
        r_b = np.log(b_close).diff()
        cov = r_t.rolling(1440, min_periods=30).cov(r_b)
        var = r_b.rolling(1440, min_periods=30).var()
        beta = cov / var.replace(0.0, np.nan)

    # Backfill beta for warmup window so early bars have a sensible beta hedge ratio
    beta = beta.bfill().fillna(1.0)

    residual_alpha = fwd_r_target - (beta * fwd_r_bench)
    return residual_alpha.rename(f"target_residual_alpha_{horizon_minutes}m")


def compute_triple_barrier_labels(
    prices: pd.Series,
    atr: pd.Series,
    multiplier: float = 1.8,
    max_holding_bars: int = 30,
) -> pd.Series:
    """Compute volatility-adjusted triple barrier classification labels:
    
    +1: Upper barrier hit first (Take-Profit)
    -1: Lower barrier hit first (Stop-Loss)
     0: Vertical time barrier reached (Chop / Timeout)
    """
    labels = pd.Series(index=prices.index, dtype=float)
    n = len(prices)
    price_vals = prices.values
    atr_vals = atr.values

    for i in range(n - max_holding_bars):
        p0 = price_vals[i]
        curr_atr = atr_vals[i]
        if np.isnan(p0) or np.isnan(curr_atr) or curr_atr <= 0:
            continue

        upper_barrier = p0 + multiplier * curr_atr
        lower_barrier = p0 - multiplier * curr_atr

        window_prices = price_vals[i + 1 : i + 1 + max_holding_bars]
        label = 0.0

        for p in window_prices:
            if p >= upper_barrier:
                label = 1.0
                break
            elif p <= lower_barrier:
                label = -1.0
                break

        labels.iloc[i] = label

    return labels.rename("target_triple_barrier")
