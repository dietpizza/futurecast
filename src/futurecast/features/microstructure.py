"""Group B: Volume & Volatility Microstructure features (strictly backward-looking)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score of volume: (V_t - mu) / sigma."""
    rolling_mean = volume.rolling(window, min_periods=min(window // 2, 20)).mean()
    rolling_std = volume.rolling(window, min_periods=min(window // 2, 20)).std()
    return ((volume - rolling_mean) / rolling_std.replace(0.0, np.nan)).fillna(0.0)


def compute_parkinson_volatility(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Compute Parkinson high-low volatility over window N bars:
    
    sigma = sqrt( 1 / (4 * ln(2) * N) * sum( ln(H_i / L_i)^2 ) )
    """
    ratio = np.log(high.clip(lower=1e-12) / low.clip(lower=1e-12))
    log_hl_sq = ratio ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    rolling_sum = log_hl_sq.rolling(window, min_periods=min(window // 2, 15)).mean()  # mean already divides by N
    parkinson = np.sqrt(factor * rolling_sum)
    return parkinson


def compute_microstructure_features(
    df: pd.DataFrame,
    target_symbol: str = "DOGEUSDT",
    benchmark_symbol: str = "BTCUSDT",
    volume_z_windows: list[int] | None = None,
    parkinson_windows: list[int] | None = None,
    rvr_window: int = 15,
) -> pd.DataFrame:
    """Compute Group B microstructure features."""
    if volume_z_windows is None:
        volume_z_windows = [60, 1440]
    if parkinson_windows is None:
        parkinson_windows = [15, 60, 240]

    out = pd.DataFrame(index=df.index)

    t_vol = df[f"{target_symbol}_volume"]

    t_high = df[f"{target_symbol}_high"]
    t_low = df[f"{target_symbol}_low"]
    b_high = df[f"{benchmark_symbol}_high"]
    b_low = df[f"{benchmark_symbol}_low"]

    # 1. Volume Z-scores (60m and 1440m)
    for w in volume_z_windows:
        out[f"volume_zscore_{w}m"] = compute_volume_zscore(t_vol, window=w)

    # 2. Relative Volume Ratio (RVR) normalized by quote volume
    t_quote = df[f"{target_symbol}_quote_volume"]
    b_quote = df[f"{benchmark_symbol}_quote_volume"]
    raw_ratio = t_quote / b_quote.replace(0.0, np.nan)
    rolling_baseline = raw_ratio.rolling(rvr_window * 4, min_periods=5).mean()
    norm_rvr = (raw_ratio / rolling_baseline.replace(0.0, np.nan)).fillna(1.0)
    out[f"relative_volume_ratio_{rvr_window}m"] = norm_rvr.rolling(rvr_window, min_periods=5).mean().clip(0.05, 25.0).fillna(1.0)

    # 3. Parkinson high-low volatility of the target over each window
    for w in parkinson_windows:
        out[f"parkinson_vol_{w}m"] = compute_parkinson_volatility(t_high, t_low, window=w)

    # Parkinson volatility of the primary benchmark (60 candles)
    btc_parkinson_60 = compute_parkinson_volatility(b_high, b_low, window=60)
    out["parkinson_vol_btc_60m"] = btc_parkinson_60

    # 4. Volatility ratio, target / primary benchmark (60 candles)
    out["realized_vol_ratio_60m"] = (out["parkinson_vol_60m"] / btc_parkinson_60.replace(0.0, np.nan)).fillna(1.0)

    # 5. Order Flow Imbalance (OFI Proxy) from taker volumes
    if f"{target_symbol}_taker_buy_volume" in df.columns:
        taker_buy = df[f"{target_symbol}_taker_buy_volume"]
        taker_sell = t_vol - taker_buy
        raw_ofi = (taker_buy - taker_sell) / (t_vol + 1e-8)
        out["ofi_proxy_5m"] = raw_ofi.rolling(5, min_periods=2).mean().fillna(0.0)
        out["ofi_proxy_15m"] = raw_ofi.rolling(15, min_periods=5).mean().fillna(0.0)
        ofi_std = raw_ofi.rolling(60, min_periods=20).std().replace(0.0, np.nan)
        out["ofi_zscore_60m"] = ((raw_ofi - raw_ofi.rolling(60, min_periods=20).mean()) / ofi_std).fillna(0.0)
    else:
        out["ofi_proxy_5m"] = 0.0
        out["ofi_proxy_15m"] = 0.0
        out["ofi_zscore_60m"] = 0.0

    return out
