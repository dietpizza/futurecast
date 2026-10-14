"""Real-time Long / Short Trade Risk and Liquidation Hazard Evaluator.

Calculates continuous institutional risk scores (0-100) for Long and Short positions:
1. Overextension & Mean-Reversion Penalty (VWAP basis & rolling price extremes).
2. Funding & Liquidation Squeeze Hazard (crowded longs vs. crowded shorts).
3. Downside Volatility & ATR Whiplash Exposure (liquidation zone proximity).
4. Order Flow Absorption & Predictive Alpha Skew.
5. Recommended Maximum Safe Leverage Guidance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_trade_risk(
    df: pd.DataFrame,
    target_symbol: str,
    features: Optional[pd.DataFrame] = None,
    microstructure_ctx: Optional[Dict[str, Any]] = None,
    expected_residual_alpha_bps: float = 0.0,
    sector: Optional[str] = None,
    orderbook_ctx: Optional[Dict[str, Any]] = None,
    bar_minutes: int = 1,
) -> Dict[str, Any]:
    """Compute real-time 0-100 trade risk scores, levels, hazard tags, and leverage limits.

    Returns structured trade risk analytics for both Long and Short participants.
    Look-back windows are defined in minutes and converted to bars of `bar_minutes`.
    These are rule-based hazard scores, not probabilities.
    """
    vwap_bars = max(240 // bar_minutes, 10)
    range_bars = max(60 // bar_minutes, 5)
    atr_bars = max(15 // bar_minutes, 3)
    ctx = microstructure_ctx or {}
    ob = orderbook_ctx or {}
    close_col = f"{target_symbol}_close"
    high_col = f"{target_symbol}_high"
    low_col = f"{target_symbol}_low"
    vol_col = f"{target_symbol}_volume"

    # Default neutral baseline if insufficient history
    if df.empty or close_col not in df.columns:
        return {
            "long_risk_score": 50,
            "long_risk_level": "MODERATE",
            "long_risk_summary": "Insufficient tick history for full risk evaluation.",
            "long_hazard_tags": ["Uncalibrated Baseline"],
            "short_risk_score": 50,
            "short_risk_level": "MODERATE",
            "short_risk_summary": "Insufficient tick history for full risk evaluation.",
            "short_hazard_tags": ["Uncalibrated Baseline"],
            "max_recommended_leverage": "2x-3x",
            "asymmetric_bias": "BALANCED",
            "asymmetry_ratio": 1.0,
        }

    c = df[close_col]
    h = df[high_col] if high_col in df.columns else c
    l = df[low_col] if low_col in df.columns else c
    v = df[vol_col] if vol_col in df.columns else pd.Series(100.0, index=df.index)

    curr_p = float(c.iloc[-1])

    # 1. Overextension vs Rolling VWAP (Price vs. 4h Volume-Weighted Average Price)
    pv = c * v
    cum_pv = pv.rolling(min(vwap_bars, len(df)), min_periods=min(10, len(df))).sum()
    cum_v = v.rolling(min(vwap_bars, len(df)), min_periods=min(10, len(df))).sum()
    vwap_series = cum_pv / cum_v.replace(0.0, np.nan)
    curr_vwap = float(vwap_series.iloc[-1]) if not np.isnan(vwap_series.iloc[-1]) else curr_p

    vwap_basis_pct = ((curr_p - curr_vwap) / (curr_vwap + 1e-12)) * 100.0

    # 2. Position within 60-Minute Price Envelope
    min_60 = float(l.rolling(min(range_bars, len(df)), min_periods=min(5, range_bars)).min().iloc[-1])
    max_60 = float(h.rolling(min(range_bars, len(df)), min_periods=min(5, range_bars)).max().iloc[-1])
    range_60 = max_60 - min_60
    dist_60 = ((curr_p - min_60) / (range_60 + 1e-12)) if range_60 > 0 else 0.5
    dist_60 = float(np.clip(dist_60, 0.0, 1.0))

    # 3. ATR Volatility % (15m Average True Range relative to price)
    tr = np.maximum(h - l, np.abs(h - c.shift(1)))
    atr_15m = float(tr.rolling(min(atr_bars, len(df)), min_periods=min(3, atr_bars)).mean().iloc[-1])
    atr_pct = (atr_15m / (curr_p + 1e-12)) * 100.0

    # 4. Funding Rate & Squeeze Dynamics
    funding_bps = float(ctx.get("funding_rate_bps", 0.0))
    squeeze_risk = float(ctx.get("squeeze_flush_risk", 0.0))
    cvd_div = float(ctx.get("cvd_divergence", 0.0))
    vol_burst = float(ctx.get("volume_clock_burst", 1.0))

    # =========================================================================
    # COMPUTE LONG RISK SCORE (0 to 100)
    # =========================================================================
    long_score = 45.0
    long_hazards: List[str] = []

    # Overextension above VWAP penalty
    if vwap_basis_pct > 6.0:
        long_score += 28.0
        long_hazards.append(f"Parabolic Climax (+{vwap_basis_pct:.1f}% vs VWAP)")
    elif vwap_basis_pct > 3.0:
        long_score += 18.0
        long_hazards.append(f"Overextended (+{vwap_basis_pct:.1f}% vs VWAP)")
    elif vwap_basis_pct < -3.0:
        long_score -= 15.0  # Discounted / mean-reversion favorable for longs

    # Buying near local highs
    if dist_60 > 0.88:
        long_score += 16.0
        long_hazards.append("Near 1h Range High")
    elif dist_60 < 0.20:
        long_score -= 10.0

    # Funding rate crowding
    if funding_bps > 6.0:
        long_score += 15.0
        long_hazards.append(f"Crowded Longs (Funding +{funding_bps:.1f} bps)")
    elif funding_bps < -4.0:
        long_score -= 10.0  # Negative funding pays longs carry

    # Long flush cascade alert from microstructure
    if squeeze_risk < -2.0:
        long_score += 14.0
        long_hazards.append("Long Flush Cascade Hazard")

    # Bearish CVD distribution
    if cvd_div < -1.2:
        long_score += 12.0
        long_hazards.append("Taker Aggressive Selling (CVD Div)")

    # Model predicted alpha direction
    if expected_residual_alpha_bps < -25.0:
        long_score += 12.0
        long_hazards.append(f"Negative Residual Alpha ({expected_residual_alpha_bps:.1f} bps)")
    elif expected_residual_alpha_bps > 25.0:
        long_score -= 14.0

    # Volatility impact
    if atr_pct > 2.5:
        long_score += 10.0
        long_hazards.append(f"High ATR Volatility ({atr_pct:.1f}%)")

    # Orderbook Level-2 Depth Imbalance (Long side)
    ob_imb = float(ob.get("book_imbalance_pct", 0.0))
    ob_spread = float(ob.get("spread_bps", 0.0))
    if ob_imb <= -22.0:
        long_score += 14.0
        long_hazards.append(f"Ask Wall Overhead ({ob_imb:.1f}%)")
    elif ob_imb >= 22.0:
        long_score -= 8.0  # Deep bid wall cushions downside
    if ob_spread >= 15.0:
        long_score += 7.0
        long_hazards.append(f"Wide Spread ({ob_spread:.1f} bps)")

    # =========================================================================
    # COMPUTE SHORT RISK SCORE (0 to 100)
    # =========================================================================
    short_score = 45.0
    short_hazards: List[str] = []

    # Short squeeze from negative funding
    if funding_bps < -5.0:
        short_score += 25.0
        short_hazards.append(f"Crowded Shorts (Funding {funding_bps:.1f} bps)")
    elif funding_bps > 5.0:
        short_score -= 12.0  # Shorts earn carry from crowded longs

    # Short squeeze breakout alert from microstructure
    if squeeze_risk > 2.0:
        short_score += 22.0
        short_hazards.append("Short Squeeze Breakout Danger")

    # Orderbook Level-2 Depth Imbalance (Short side)
    if ob_imb >= 22.0:
        short_score += 15.0
        short_hazards.append(f"Deep Bid Support Wall (+{ob_imb:.1f}%)")
    elif ob_imb <= -22.0:
        short_score -= 8.0  # Ask wall aids short thesis
    if ob_spread >= 15.0:
        short_score += 7.0
        short_hazards.append(f"Wide Spread ({ob_spread:.1f} bps)")

    # High upward momentum / burst
    if vwap_basis_pct > 4.0 and vol_burst > 1.8:
        short_score += 18.0
        short_hazards.append("Shorting High-Velocity Momentum")
    elif vwap_basis_pct < -5.0:
        short_score += 16.0
        short_hazards.append(f"Oversold Bottom (-{abs(vwap_basis_pct):.1f}% vs VWAP)")

    # Shorting at the exact low of range
    if dist_60 < 0.15:
        short_score += 16.0
        short_hazards.append("Near 1h Range Low (Wick Rebound)")
    elif dist_60 > 0.85:
        short_score -= 10.0  # Favorable mean-reversion short setup

    # Bullish CVD absorption
    if cvd_div > 1.2:
        short_score += 14.0
        short_hazards.append("Taker Accumulation Absorbing Asks")

    # Model predicted alpha direction
    if expected_residual_alpha_bps > 25.0:
        short_score += 14.0
        short_hazards.append(f"Positive Residual Alpha (+{expected_residual_alpha_bps:.1f} bps)")
    elif expected_residual_alpha_bps < -25.0:
        short_score -= 12.0

    # High volatility whiplash
    if atr_pct > 2.5:
        short_score += 10.0
        short_hazards.append(f"High ATR Volatility ({atr_pct:.1f}%)")

    # =========================================================================
    # NORMALIZE AND CATEGORIZE
    # =========================================================================
    final_long_score = int(np.clip(round(long_score), 5, 98))
    final_short_score = int(np.clip(round(short_score), 5, 98))

    def _get_level(score: int) -> str:
        if score >= 82:
            return "EXTREME"
        elif score >= 65:
            return "HIGH"
        elif score >= 38:
            return "MODERATE"
        return "LOW"

    long_level = _get_level(final_long_score)
    short_level = _get_level(final_short_score)

    # Leverage Recommendation based on maximum of both risks and ATR
    max_risk = max(final_long_score, final_short_score)
    if max_risk >= 82 or atr_pct >= 3.5:
        max_leverage = "1x-2x (High Hazard)"
    elif max_risk >= 65 or atr_pct >= 2.0:
        max_leverage = "2x-3x (Strict Risk)"
    elif max_risk >= 45:
        max_leverage = "3x-5x (Moderate)"
    else:
        max_leverage = "5x-10x (Standard)"

    # Asymmetric Bias
    if final_long_score >= 70 and final_short_score >= 70:
        bias = "AVOID_BOTH_HIGH_RISK"
    elif final_long_score - final_short_score >= 18:
        bias = "FAVORS_SHORT"
    elif final_short_score - final_long_score >= 18:
        bias = "FAVORS_LONG"
    else:
        bias = "BALANCED"

    asym_ratio = round(final_long_score / max(1.0, float(final_short_score)), 2)

    # Human-friendly summaries
    if long_hazards:
        long_summary = f"{long_level} Risk: " + ", ".join(long_hazards[:2]) + "."
    else:
        long_summary = f"{long_level} Risk: Stable baseline, no severe long liquidation or overextension signals."

    if short_hazards:
        short_summary = f"{short_level} Risk: " + ", ".join(short_hazards[:2]) + "."
    else:
        short_summary = f"{short_level} Risk: Stable baseline, no violent squeeze or absorption hazards detected."

    return {
        "long_risk_score": final_long_score,
        "long_risk_level": long_level,
        "long_risk_summary": long_summary,
        "long_hazard_tags": long_hazards[:3] if long_hazards else ["Normal Range"],
        "short_risk_score": final_short_score,
        "short_risk_level": short_level,
        "short_risk_summary": short_summary,
        "short_hazard_tags": short_hazards[:3] if short_hazards else ["Normal Range"],
        "max_recommended_leverage": max_leverage,
        "asymmetric_bias": bias,
        "asymmetry_ratio": asym_ratio,
        "vwap_basis_pct": round(vwap_basis_pct, 2),
        "atr_pct": round(atr_pct, 2),
    }
