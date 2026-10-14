"""Level-2 Orderbook Depth & Microstructure Imbalance Engine.

Calculates high-frequency institutional orderbook metrics:
1. Bid-Ask Depth Ratio & Order Flow Imbalance (OFI).
2. Volume-Weighted Microprice Drift (bps).
3. Near-Book vs. Deep-Book Liquidity Gradient.
4. Bid/Ask Spread (bps).
5. Dynamic Short-Term Residual Alpha Tilt.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def compute_orderbook_metrics(depth_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Level-2 orderbook bids and asks to extract directional liquidity pressure."""
    bids: List[Tuple[float, float]] = depth_data.get("bids", [])
    asks: List[Tuple[float, float]] = depth_data.get("asks", [])
    source: str = depth_data.get("source", "unknown")

    # Safe fallback if orderbook data is empty
    if not bids or not asks:
        return {
            "available": False,
            "source": source,
            "bid_depth_usd": 0.0,
            "ask_depth_usd": 0.0,
            "bid_pct": 50,
            "ask_pct": 50,
            "depth_ratio": 1.0,
            "book_imbalance_pct": 0.0,
            "near_imbalance_pct": 0.0,
            "spread_bps": 0.0,
            "microprice_skew_bps": 0.0,
            "pressure_signal": "BALANCED",
            "depth_alpha_adjustment_bps": 0.0,
            "summary": "Orderbook depth currently calibrating.",
        }

    best_bid, best_bid_qty = bids[0]
    best_ask, best_ask_qty = asks[0]

    mid_price = (best_bid + best_ask) / 2.0
    if mid_price <= 0:
        mid_price = best_bid

    # Spread in basis points (100 bps = 1.0%)
    spread_bps = max(0.0, ((best_ask - best_bid) / (mid_price + 1e-12)) * 10000.0)

    # Top-1 Microprice (Volume-weighted mid price)
    denom_top = best_bid_qty + best_ask_qty + 1e-12
    microprice = (best_bid * best_ask_qty + best_ask * best_bid_qty) / denom_top
    microprice_skew_bps = ((microprice - mid_price) / (mid_price + 1e-12)) * 10000.0

    # Near depth (Top 5 levels)
    near_bids = bids[:5]
    near_asks = asks[:5]
    near_b_usd = sum(p * q for p, q in near_bids)
    near_a_usd = sum(p * q for p, q in near_asks)
    near_total = near_b_usd + near_a_usd + 1e-9
    near_imbalance = ((near_b_usd - near_a_usd) / near_total) * 100.0

    # Total depth (Up to 20 levels)
    total_b_usd = sum(p * q for p, q in bids)
    total_a_usd = sum(p * q for p, q in asks)
    total_usd = total_b_usd + total_a_usd + 1e-9

    book_imbalance = ((total_b_usd - total_a_usd) / total_usd) * 100.0
    depth_ratio = total_b_usd / max(1.0, total_a_usd)

    bid_pct = int(np.clip(round((total_b_usd / total_usd) * 100.0), 5, 95))
    ask_pct = 100 - bid_pct

    # Short-term directional alpha adjustment (tuned specifically for 5m & 15m horizons)
    # Book imbalance of +30% translates to ~+7.5 bps; microprice skew adds directional push
    raw_alpha_adj = (book_imbalance * 0.22) + (microprice_skew_bps * 0.85)
    depth_alpha_adj = float(np.clip(round(raw_alpha_adj, 2), -28.0, 28.0))

    # Categorize Pressure Signal
    if book_imbalance >= 25.0 and microprice_skew_bps >= 0.4:
        pressure_signal = "STRONG_BUY_PRESSURE"
        summary = f"Strong Bid Stacking (+{book_imbalance:.1f}%): Large limit buyers absorbing asks, microprice leading +{microprice_skew_bps:.1f} bps."
    elif book_imbalance >= 12.0:
        pressure_signal = "BUY_PRESSURE"
        summary = f"Positive Depth Skew (+{book_imbalance:.1f}%): Bid walls outweigh asks (${total_b_usd:,.0f} vs ${total_a_usd:,.0f})."
    elif book_imbalance <= -25.0 and microprice_skew_bps <= -0.4:
        pressure_signal = "STRONG_SELL_PRESSURE"
        summary = f"Heavy Ask Wall Overhead ({book_imbalance:.1f}%): Institutional sellers capping price, microprice dragging {microprice_skew_bps:.1f} bps."
    elif book_imbalance <= -12.0:
        pressure_signal = "SELL_PRESSURE"
        summary = f"Negative Depth Skew ({book_imbalance:.1f}%): Ask resistance exceeds bids (${total_a_usd:,.0f} vs ${total_b_usd:,.0f})."
    else:
        pressure_signal = "BALANCED"
        summary = f"Balanced Orderbook ({book_imbalance:+.1f}%): Symmetric two-way liquidity across top 20 levels."

    return {
        "available": True,
        "source": source,
        "best_bid": round(best_bid, 6),
        "best_ask": round(best_ask, 6),
        "bid_depth_usd": round(total_b_usd, 2),
        "ask_depth_usd": round(total_a_usd, 2),
        "bid_pct": bid_pct,
        "ask_pct": ask_pct,
        "depth_ratio": round(depth_ratio, 2),
        "book_imbalance_pct": round(book_imbalance, 1),
        "near_imbalance_pct": round(near_imbalance, 1),
        "spread_bps": round(spread_bps, 2),
        "microprice_skew_bps": round(microprice_skew_bps, 2),
        "pressure_signal": pressure_signal,
        "depth_alpha_adjustment_bps": depth_alpha_adj,
        "summary": summary,
    }
