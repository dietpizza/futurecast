"""Tests for the order book depth metrics shown on the dashboard."""

import pytest

from futurecast.features.orderbook import compute_orderbook_metrics


def test_orderbook_metrics_empty_book():
    assert compute_orderbook_metrics({"bids": [], "asks": []}) == {"available": False}


def test_orderbook_metrics_balanced_book():
    # Mid-price = 100.0, spread = 0.02 (2 bps)
    metrics = compute_orderbook_metrics({
        "bids": [(99.99 - i * 0.01, 1000.0) for i in range(20)],
        "asks": [(100.01 + i * 0.01, 1000.0) for i in range(20)],
    })
    assert metrics["available"] is True
    assert 48 <= metrics["bid_pct"] <= 52 and metrics["bid_pct"] + metrics["ask_pct"] == 100
    assert -2.0 <= metrics["book_imbalance_pct"] <= 2.0
    assert metrics["spread_bps"] == pytest.approx(2.0, abs=0.01)


def test_orderbook_metrics_bid_wall():
    # 5x larger bid volume than asks
    metrics = compute_orderbook_metrics({
        "bids": [(99.99 - i * 0.01, 5000.0) for i in range(20)],
        "asks": [(100.01 + i * 0.01, 1000.0) for i in range(20)],
    })
    assert metrics["bid_pct"] >= 75
    assert metrics["book_imbalance_pct"] >= 50.0 and metrics["near_imbalance_pct"] >= 50.0
    assert metrics["bid_depth_usd"] > 4 * metrics["ask_depth_usd"]


def test_orderbook_metrics_ask_wall():
    # 4x larger ask volume than bids
    metrics = compute_orderbook_metrics({
        "bids": [(0.02250 - i * 0.00001, 2000.0) for i in range(20)],
        "asks": [(0.02251 + i * 0.00001, 8000.0) for i in range(20)],
    })
    assert metrics["ask_pct"] >= 75
    assert metrics["book_imbalance_pct"] <= -50.0
