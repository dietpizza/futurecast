"""Group C: Derivative Flow & Positioning features (strictly backward-looking)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_derivative_features(
    df: pd.DataFrame,
    target_symbol: str = "DOGEUSDT",
    benchmark_symbol: str = "BTCUSDT",
    funding_ema_window: int = 4320,  # 3 days in 1m bars
    oi_delta_windows: list[int] | None = None,
    divergence_window: int = 60,
) -> pd.DataFrame:
    """Compute Group C derivative flow features."""
    if oi_delta_windows is None:
        oi_delta_windows = [15, 60]

    out = pd.DataFrame(index=df.index)

    t_close = df[f"{target_symbol}_close"]

    # 1. Funding Rate Deviation (target vs its 3-day EMA)
    if f"{target_symbol}_funding_rate" in df.columns:
        f_target = df[f"{target_symbol}_funding_rate"]
        f_ema = f_target.ewm(span=funding_ema_window, min_periods=60, adjust=False).mean()
        out["funding_rate_dev_3d"] = f_target - f_ema
        out["funding_rate_bps"] = f_target * 10000.0
    else:
        out["funding_rate_dev_3d"] = 0.0
        out["funding_rate_bps"] = 0.0

    # 2. Funding Spread to Benchmark (DOGE - BTC)
    if f"{target_symbol}_funding_rate" in df.columns and f"{benchmark_symbol}_funding_rate" in df.columns:
        f_bench = df[f"{benchmark_symbol}_funding_rate"]
        out["funding_rate_spread_btc"] = (df[f"{target_symbol}_funding_rate"] - f_bench) * 10000.0
    else:
        out["funding_rate_spread_btc"] = 0.0

    # 3. Open Interest (OI) % Deltas (15m, 60m)
    if f"{target_symbol}_open_interest" in df.columns:
        oi = df[f"{target_symbol}_open_interest"]
        for w in oi_delta_windows:
            oi_prev = oi.shift(w).replace(0.0, np.nan)
            out[f"oi_delta_{w}m_pct"] = (oi - oi_prev) / oi_prev

        # 4. OI-Price Divergence: sign(Delta P_60) * sign(Delta OI_60)
        # +1: aggressive long buildup or short covering
        # -1: divergence (price up on dropping OI, or price down on rising OI)
        delta_p = t_close.diff(divergence_window)
        delta_oi = oi.diff(divergence_window)
        out["oi_price_divergence_60m"] = np.sign(delta_p) * np.sign(delta_oi)

        # 5. Liquidation Proxy (sharp OI drop accompanied by high wick volatility)
        oi_drop = (-delta_oi).clip(lower=0.0) / (oi.shift(divergence_window).replace(0.0, np.nan) + 1e-8)
        hl_spread = (df[f"{target_symbol}_high"] - df[f"{target_symbol}_low"]) / t_close
        out["liquidation_flush_proxy_60m"] = oi_drop * hl_spread
    else:
        for w in oi_delta_windows:
            out[f"oi_delta_{w}m_pct"] = 0.0
        out["oi_price_divergence_60m"] = 0.0
        out["liquidation_flush_proxy_60m"] = 0.0

    return out
