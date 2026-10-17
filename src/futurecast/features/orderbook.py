"""Order book depth metrics for display (no statistic or signal uses them): the bid/ask split and
imbalance over the top 20 levels and over the top 5, and the spread."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def compute_orderbook_metrics(depth_data: Dict[str, Any]) -> Dict[str, Any]:
    """Bid/ask liquidity in USD, its split and imbalance, and the spread in bps."""
    bids: List[Tuple[float, float]] = depth_data.get("bids", [])
    asks: List[Tuple[float, float]] = depth_data.get("asks", [])
    if not bids or not asks:
        return {"available": False}

    best_bid, best_ask = bids[0][0], asks[0][0]
    mid_price = (best_bid + best_ask) / 2.0
    if mid_price <= 0:
        mid_price = best_bid
    spread_bps = max(0.0, ((best_ask - best_bid) / (mid_price + 1e-12)) * 10000.0)

    near_b_usd = sum(p * q for p, q in bids[:5])
    near_a_usd = sum(p * q for p, q in asks[:5])
    near_imbalance = ((near_b_usd - near_a_usd) / (near_b_usd + near_a_usd + 1e-9)) * 100.0

    total_b_usd = sum(p * q for p, q in bids)
    total_a_usd = sum(p * q for p, q in asks)
    total_usd = total_b_usd + total_a_usd + 1e-9
    book_imbalance = ((total_b_usd - total_a_usd) / total_usd) * 100.0
    bid_pct = int(np.clip(round((total_b_usd / total_usd) * 100.0), 5, 95))

    return {
        "available": True,
        "bid_depth_usd": round(total_b_usd, 2),
        "ask_depth_usd": round(total_a_usd, 2),
        "bid_pct": bid_pct,
        "ask_pct": 100 - bid_pct,
        "book_imbalance_pct": round(book_imbalance, 1),
        "near_imbalance_pct": round(near_imbalance, 1),
        "spread_bps": round(spread_bps, 2),
    }
