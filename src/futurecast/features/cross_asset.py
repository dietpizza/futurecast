"""Group A: Cross-Asset Lead-Lag & Dynamics features (strictly backward-looking)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(series: pd.Series, periods: list[int], prefix: str) -> pd.DataFrame:
    """Compute multi-horizon log returns: ln(P_t / P_{t-k})."""
    feats = pd.DataFrame(index=series.index)
    log_p = np.log(series.clip(lower=1e-12))
    for k in periods:
        feats[f"{prefix}_ret_{k}m"] = log_p.diff(k)
    return feats


def compute_rolling_beta(
    target_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int,
    col_name: str,
) -> pd.Series:
    """Compute rolling beta: Cov(R_target, R_bench) / Var(R_bench)."""
    min_p = min(window // 2, 30)
    cov = target_returns.rolling(window, min_periods=min_p).cov(benchmark_returns)
    var = benchmark_returns.rolling(window, min_periods=min_p).var()
    # Avoid division by zero
    beta = (cov / var.replace(0.0, np.nan)).fillna(1.0)
    return beta.rename(col_name)


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute standard Relative Strength Index."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's Exponential Smoothing
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_cross_asset_features(
    df: pd.DataFrame,
    target_symbol: str = "DOGEUSDT",
    primary_benchmark: str = "BTCUSDT",
    secondary_benchmark: str = "ETHUSDT",
    horizons: list[int] | None = None,
    beta_windows: list[int] | None = None,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """Compute all 10 Group A cross-asset dynamic features."""
    if horizons is None:
        horizons = [1, 5, 15, 60]
    if beta_windows is None:
        beta_windows = [60, 1440]

    out = pd.DataFrame(index=df.index)

    t_close = df[f"{target_symbol}_close"]
    b_close = df[f"{primary_benchmark}_close"]
    e_close = df[f"{secondary_benchmark}_close"]

    # 1. Multi-horizon log returns for all 3 assets
    out = pd.concat([
        out,
        compute_log_returns(t_close, horizons, prefix=target_symbol),
        compute_log_returns(b_close, horizons, prefix=primary_benchmark),
        compute_log_returns(e_close, horizons, prefix=secondary_benchmark),
    ], axis=1)

    t_ret1 = out[f"{target_symbol}_ret_1m"]
    b_ret1 = out[f"{primary_benchmark}_ret_1m"]
    e_ret1 = out[f"{secondary_benchmark}_ret_1m"]

    # 2. Rolling Beta to BTC (60m and 1440m)
    for w in beta_windows:
        out[f"beta_btc_{w}m"] = compute_rolling_beta(t_ret1, b_ret1, window=w, col_name=f"beta_btc_{w}m")

    # Rolling Beta to ETH (1440m)
    out["beta_eth_1440m"] = compute_rolling_beta(t_ret1, e_ret1, window=1440, col_name="beta_eth_1440m")

    # 3. Beta residual momentum (15 candles): R_target,15 - beta_1440 * R_benchmark,15
    out["beta_residual_momentum_15m"] = (
        out[f"{target_symbol}_ret_15m"] - out["beta_btc_1440m"] * out[f"{primary_benchmark}_ret_15m"]
    )

    # 4. Return spreads to both benchmarks ("doge" in the names means the target coin)
    out["doge_btc_ret_spread_15m"] = out[f"{target_symbol}_ret_15m"] - out[f"{primary_benchmark}_ret_15m"]
    out["doge_eth_ret_spread_15m"] = out[f"{target_symbol}_ret_15m"] - out[f"{secondary_benchmark}_ret_15m"]

    # 5. Relative Strength Index (RSI) Spreads
    rsi_doge = compute_rsi(t_close, period=rsi_period)
    rsi_btc = compute_rsi(b_close, period=rsi_period)
    out["rsi_spread_doge_btc_14m"] = rsi_doge - rsi_btc

    return out
