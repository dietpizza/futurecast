"""High-Beta Shitcoin Small-Timeframe Prediction Strategy (Microstructure & Quant Signals).

STRICT APPLICATION BOUNDARY:
This module is strictly designed for high-beta, low-liquidity meme/alt tokens
(e.g., PEPE, DOGE, SHIB, FLOKI, BONK, WIF).
Large-volume major assets (BTC, SOL, ETH, AVAX, etc.) are explicitly excluded
and remain on standard benchmark feature pipelines without any modifications.

Implements the 6 small-timeframe quant techniques:
1. Volume/Tick Clock Information-Driven Burst Ratio (vs. rolling quiet chop baseline).
2. Lead-Lag Cross-Exchange Arbitrage & Mark-Price Divergence.
3. Shallow-Tier Order Book Skew (Depth Asymmetry in top 3-5 price levels / Micro-OFI).
4. Sector-Leader Spillover Velocity (DOGE acceleration leading meme followers).
5. Cumulative Volume Delta (CVD) Divergence (detecting absorption in small consolidation ranges).
6. Liquidation & Open Interest Flushes / Squeeze Risk Index.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from futurecast.ingestion.factory import SECTOR_MAP, extract_base_and_quote

logger = logging.getLogger(__name__)

# Canonical meme/high-beta shitcoin identifiers
SHITCOIN_BASES = {
    "DOGE", "SHIB", "PEPE", "1000PEPE", "FLOKI", "1000FLOKI",
    "BONK", "1000BONK", "WIF", "MEME", "POPCAT", "MOG", "NEIRO",
    "BOME", "MEW", "BRETT", "MYRO", "TURBO",
}


def is_high_beta_shitcoin(symbol: str, sector: Optional[str] = None) -> bool:
    """Return True if symbol is a high-beta shitcoin/meme token; False for BTC, SOL, ETH, etc.
    
    Strict invariant:
    - BTC, SOL, ETH, AVAX, SUI, ADA, DOT -> False.
    - PEPE, DOGE, SHIB, FLOKI, BONK, WIF -> True.
    """
    if sector and sector.upper() == "MEME":
        return True
    base, _ = extract_base_and_quote(symbol)
    base_upper = base.upper()
    base_clean = base_upper.replace("1000", "")

    if SECTOR_MAP.get(base_upper) == "MEME" or SECTOR_MAP.get(base_clean) == "MEME":
        return True

    return base_upper in SHITCOIN_BASES or base_clean in SHITCOIN_BASES


def compute_volume_tick_clock_burst(
    df: pd.DataFrame,
    target_symbol: str,
    short_window: int = 3,
    long_window: int = 60,
) -> pd.Series:
    """Volume & Tick Clock Burst Ratio.
    
    Compares the immediate short-window volume & trade frequency against a rolling
    60-minute quiet baseline.
    
    Ratio = V_3m / (V_60m_mean * 3)
    - Burst Ratio < 0.6: Dormant Chop (filter out false breakouts).
    - Burst Ratio > 1.8: Informational Burst (high-speed aggressive participant arrival).
    """
    vol_col = f"{target_symbol}_volume"
    if vol_col not in df.columns:
        return pd.Series(1.0, index=df.index)

    vol = df[vol_col].clip(lower=0.0)
    short_vol = vol.rolling(short_window, min_periods=1).sum()
    long_mean = vol.rolling(long_window, min_periods=min(15, len(df))).mean()
    expected_short_vol = (long_mean * short_window).replace(0.0, np.nan)
    vol_ratio = (short_vol / expected_short_vol).fillna(1.0)

    # Incorporate trade count (tick clock proxy) if present
    count_col = f"{target_symbol}_count"
    if count_col in df.columns:
        counts = df[count_col].clip(lower=0.0)
        short_count = counts.rolling(short_window, min_periods=1).sum()
        long_count_mean = counts.rolling(long_window, min_periods=min(15, len(df))).mean()
        expected_short_count = (long_count_mean * short_window).replace(0.0, np.nan)
        count_ratio = (short_count / expected_short_count).fillna(1.0)
        burst_ratio = 0.6 * vol_ratio + 0.4 * count_ratio
    else:
        burst_ratio = vol_ratio

    return burst_ratio.clip(lower=0.05, upper=15.0)


def compute_shallow_book_skew(
    df: pd.DataFrame,
    target_symbol: str,
    window: int = 5,
) -> pd.Series:
    """Shallow-Tier Order Book Skew (Depth Asymmetry in top 3-5 price levels).
    
    Synthesizes short-horizon taker volume imbalance (Order Flow Imbalance)
    with candlestick shadow asymmetry (lower wick vs. upper wick rejection)
    as a robust proxy for top-of-book depth asymmetry.
    
    - Positive (+0.3 to +1.0): Aggressive bid resting absorption / bid support.
    - Negative (-0.3 to -1.0): Bid exhaustion or aggressive spoof dump.
    """
    taker_buy_col = f"{target_symbol}_taker_buy_volume"
    vol_col = f"{target_symbol}_volume"
    high_col = f"{target_symbol}_high"
    low_col = f"{target_symbol}_low"
    close_col = f"{target_symbol}_close"
    open_col = f"{target_symbol}_open"

    if vol_col not in df.columns or close_col not in df.columns:
        return pd.Series(0.0, index=df.index)

    # 1. Taker OFI proxy
    if taker_buy_col in df.columns:
        taker_buy = df[taker_buy_col]
        taker_sell = df[vol_col] - taker_buy
        ofi_raw = (taker_buy - taker_sell) / (df[vol_col] + 1e-8)
        ofi_smooth = ofi_raw.rolling(window, min_periods=1).mean()
    else:
        ofi_smooth = pd.Series(0.0, index=df.index)

    # 2. Wick shadow asymmetry
    if high_col in df.columns and low_col in df.columns and open_col in df.columns:
        body_top = np.maximum(df[open_col], df[close_col])
        body_bottom = np.minimum(df[open_col], df[close_col])
        upper_wick = df[high_col] - body_top
        lower_wick = body_bottom - df[low_col]
        full_range = df[high_col] - df[low_col] + 1e-8
        wick_skew = (lower_wick - upper_wick) / full_range
        wick_skew_smooth = wick_skew.rolling(window, min_periods=1).mean()
    else:
        wick_skew_smooth = pd.Series(0.0, index=df.index)

    skew = 0.5 * ofi_smooth + 0.5 * wick_skew_smooth
    return skew.clip(lower=-1.0, upper=1.0).fillna(0.0)


def compute_sector_leader_spillover_velocity(
    df: pd.DataFrame,
    target_symbol: str,
    leader_symbol: str = "B-DOGE_USDT",
    window: int = 5,
) -> pd.Series:
    """Sector-Leader Spillover Velocity.
    
    Tracks the acceleration of the sector anchor (DOGE) and projects the lagged
    beta expansion velocity for followers (PEPE, FLOKI, BONK, SHIB, WIF).
    
    When leader DOGE experiences an aggressive 1-3m volume spike + upward momentum,
    followers experience a lagged beta expansion.
    """
    leader_close_col = f"{leader_symbol}_close"
    leader_vol_col = f"{leader_symbol}_volume"
    target_close_col = f"{target_symbol}_close"

    # If leader columns don't exist, try resolving alternative prefix
    if leader_close_col not in df.columns:
        alt_leader = "DOGEUSDT"
        if f"{alt_leader}_close" in df.columns:
            leader_close_col = f"{alt_leader}_close"
            leader_vol_col = f"{alt_leader}_volume"

    if leader_close_col not in df.columns or target_close_col not in df.columns:
        return pd.Series(0.0, index=df.index)

    # Leader 1m and 5m returns
    leader_close = df[leader_close_col]
    ret_leader_1m = np.log(leader_close.clip(lower=1e-12)).diff(1)
    ret_leader_5m = np.log(leader_close.clip(lower=1e-12)).diff(window)

    # Acceleration: 1m velocity compared to average 5m pace
    leader_accel = ret_leader_1m - (ret_leader_5m / float(window))

    # Leader volume multiplier
    if leader_vol_col in df.columns:
        l_vol = df[leader_vol_col]
        l_vol_mean = l_vol.rolling(60, min_periods=min(10, len(df))).mean().replace(0.0, np.nan)
        l_vol_burst = (l_vol.rolling(3, min_periods=1).sum() / (l_vol_mean * 3.0)).fillna(1.0).clip(0.5, 4.0)
    else:
        l_vol_burst = 1.0

    # If the target IS the leader (e.g. DOGE itself), spillover is its own acceleration
    is_self_leader = "DOGE" in target_symbol.upper()
    if is_self_leader:
        spillover = leader_accel * 10000.0  # in bps
    else:
        spillover = leader_accel * l_vol_burst * 10000.0

    return spillover.clip(lower=-50.0, upper=50.0).fillna(0.0)


def compute_cvd_divergence(
    df: pd.DataFrame,
    target_symbol: str,
    window: int = 15,
) -> pd.Series:
    """Cumulative Volume Delta (CVD) Divergence on small timeframes (5m-15m).
    
    Measures aggressive taker market accumulation/distribution vs. price movement.
    
    - Bullish Absorption: Price is flat or slightly dipping (range-bound),
      but taker buy delta is surging -> Breakout imminent upward.
    - Bearish Distribution: Price is flat or drifting up, but taker sell delta
      is surging -> Breakdown imminent downward.
    """
    taker_buy_col = f"{target_symbol}_taker_buy_volume"
    vol_col = f"{target_symbol}_volume"
    close_col = f"{target_symbol}_close"

    if taker_buy_col not in df.columns or vol_col not in df.columns or close_col not in df.columns:
        return pd.Series(0.0, index=df.index)

    taker_buy = df[taker_buy_col]
    taker_sell = df[vol_col] - taker_buy
    delta = taker_buy - taker_sell

    # Rolling window delta
    rolling_delta = delta.rolling(window, min_periods=min(5, len(df))).sum()
    delta_mean = rolling_delta.rolling(60, min_periods=min(15, len(df))).mean()
    delta_std = rolling_delta.rolling(60, min_periods=min(15, len(df))).std().replace(0.0, np.nan)
    z_cvd = ((rolling_delta - delta_mean) / delta_std).fillna(0.0)

    # Rolling price return
    close = df[close_col]
    ret = np.log(close.clip(lower=1e-12)).diff(window)
    ret_mean = ret.rolling(60, min_periods=min(15, len(df))).mean()
    ret_std = ret.rolling(60, min_periods=min(15, len(df))).std().replace(0.0, np.nan)
    z_price = ((ret - ret_mean) / ret_std).fillna(0.0)

    # Divergence: Positive indicates CVD lead over price (absorption)
    divergence = z_cvd - z_price
    return divergence.clip(lower=-5.0, upper=5.0).fillna(0.0)


def compute_squeeze_flush_risk(
    df: pd.DataFrame,
    target_symbol: str,
) -> pd.Series:
    """Liquidation & Open Interest Flushes / Squeeze Risk Index.
    
    Identifies extreme fragile positioning:
    - Rapid spikes in Open Interest + Extreme positive funding -> Long flush cascade risk.
    - Rapid spikes in Open Interest + Extreme negative funding -> Short squeeze breakout primed.
    """
    funding_col = f"{target_symbol}_funding_rate"
    oi_col = f"{target_symbol}_open_interest"

    if funding_col not in df.columns or oi_col not in df.columns:
        return pd.Series(0.0, index=df.index)

    funding = df[funding_col] * 10000.0  # in bps
    oi = df[oi_col]
    oi_15m = oi.shift(15)
    oi_delta_pct = ((oi - oi_15m) / oi_15m.replace(0.0, np.nan) * 100.0).fillna(0.0).clip(-20.0, 50.0)

    # Squeeze index: negative funding * expanding OI = Short Squeeze risk (positive for upside)
    # positive funding * expanding OI = Long Flush risk (negative for downside cascade)
    squeeze_index = -1.0 * (funding / 5.0) * (np.maximum(0.0, oi_delta_pct) / 2.0)
    return squeeze_index.clip(lower=-10.0, upper=10.0).fillna(0.0)


def compute_cross_exchange_lead_lag(
    df: pd.DataFrame,
    target_symbol: str,
    leader_symbol: str = "B-DOGE_USDT",
) -> pd.Series:
    """Lead-Lag Cross-Exchange Arbitrage Latency proxy.
    
    Calculates 1-minute lead-lag return spread between the primary sector benchmark
    and the target shitcoin to predict immediate 60-120s price adjustments.
    """
    leader_close_col = f"{leader_symbol}_close"
    target_close_col = f"{target_symbol}_close"

    if leader_close_col not in df.columns or target_close_col not in df.columns:
        return pd.Series(0.0, index=df.index)

    r_target_1m = np.log(df[target_close_col].clip(lower=1e-12)).diff(1)
    r_leader_1m = np.log(df[leader_close_col].clip(lower=1e-12)).diff(1)

    lead_lag = (r_leader_1m - r_target_1m) * 10000.0  # bps
    return lead_lag.clip(lower=-30.0, upper=30.0).fillna(0.0)


def evaluate_shitcoin_microstructure(
    df: pd.DataFrame,
    target_symbol: str,
    sector: Optional[str] = None,
    leader_symbol: str = "B-DOGE_USDT",
) -> Dict[str, Any]:
    """Evaluate full suite of 6 high-beta shitcoin signals strictly for shitcoins.
    
    If the coin is NOT a high-beta shitcoin (e.g. BTC, SOL, ETH, AVAX),
    returns an empty context with is_shitcoin_strategy_active=False.
    """
    if not is_high_beta_shitcoin(target_symbol, sector):
        return {
            "is_shitcoin_strategy_active": False,
        }

    if df.empty or len(df) < 5:
        return {
            "is_shitcoin_strategy_active": True,
            "volume_clock_burst": 1.0,
            "shallow_book_skew": 0.0,
            "sector_leader_spillover": 0.0,
            "cvd_divergence": 0.0,
            "squeeze_flush_risk": 0.0,
            "cross_exchange_lead_lag_bps": 0.0,
            "shitcoin_regime": "NORMAL_FLOW",
            "shitcoin_alpha_boost_bps": 0.0,
        }

    # Compute continuous series
    s_burst = compute_volume_tick_clock_burst(df, target_symbol)
    s_skew = compute_shallow_book_skew(df, target_symbol)
    s_spillover = compute_sector_leader_spillover_velocity(df, target_symbol, leader_symbol=leader_symbol)
    s_cvd = compute_cvd_divergence(df, target_symbol)
    s_squeeze = compute_squeeze_flush_risk(df, target_symbol)
    s_lead_lag = compute_cross_exchange_lead_lag(df, target_symbol, leader_symbol=leader_symbol)

    latest_burst = float(s_burst.iloc[-1])
    latest_skew = float(s_skew.iloc[-1])
    latest_spillover = float(s_spillover.iloc[-1])
    latest_cvd = float(s_cvd.iloc[-1])
    latest_squeeze = float(s_squeeze.iloc[-1])
    latest_lead_lag = float(s_lead_lag.iloc[-1])

    # Determine Regime
    if latest_burst < 0.6:
        regime = "DORMANT_CHOP"
    elif latest_squeeze > 3.0:
        regime = "SHORT_SQUEEZE_ALERT"
    elif latest_squeeze < -3.0:
        regime = "LONG_FLUSH_ALERT"
    elif latest_cvd > 1.2:
        regime = "BULLISH_ABSORPTION"
    elif latest_cvd < -1.2:
        regime = "BEARISH_DISTRIBUTION"
    elif latest_burst >= 1.8:
        regime = "BURST_EXPANSION"
    else:
        regime = "NORMAL_FLOW"

    # Compute calibrated alpha adjustment (bps)
    # - In Dormant Chop: dampen momentum signals to prevent false triggers
    # - In Burst Expansion: amplify leader spillover, shallow skew, and CVD absorption
    alpha_boost = 0.0

    if regime == "DORMANT_CHOP":
        # In dead illiquid periods, dampen alpha towards neutral
        alpha_boost = 0.0
    else:
        # Active microstructure composite
        alpha_boost = (
            (latest_skew * 6.0)               # Shallow depth skew
            + (latest_cvd * 4.0)               # CVD absorption
            + (latest_spillover * 0.25)        # Sector leader spillover velocity
            + (latest_lead_lag * 0.20)         # Cross-exchange lead-lag
            + (latest_squeeze * 1.5)           # Squeeze pressure
        )

        # Scale by volume clock burst
        if latest_burst > 1.5:
            alpha_boost *= min(2.0, latest_burst)

    alpha_boost = float(np.clip(alpha_boost, -35.0, 35.0))

    return {
        "is_shitcoin_strategy_active": True,
        "volume_clock_burst": round(latest_burst, 2),
        "shallow_book_skew": round(latest_skew, 2),
        "sector_leader_spillover": round(latest_spillover, 2),
        "cvd_divergence": round(latest_cvd, 2),
        "squeeze_flush_risk": round(latest_squeeze, 2),
        "cross_exchange_lead_lag_bps": round(latest_lead_lag, 2),
        "shitcoin_regime": regime,
        "shitcoin_alpha_boost_bps": round(alpha_boost, 2),
    }
