"""Tests for Level-2 Orderbook Depth & Microprice Skew Analytics."""

from datetime import datetime, timezone
import pytest

from futurecast.features.orderbook import compute_orderbook_metrics
from futurecast.schema import ForecastEvent, SignalState


def test_orderbook_metrics_empty_fallback():
    """Empty or uncalibrated orderbook depth should return bounded neutral baseline."""
    empty_depth = {"bids": [], "asks": []}
    metrics = compute_orderbook_metrics(empty_depth)

    assert metrics["available"] is False
    assert metrics["bid_pct"] == 50
    assert metrics["ask_pct"] == 50
    assert metrics["depth_ratio"] == 1.0
    assert metrics["book_imbalance_pct"] == 0.0
    assert metrics["depth_alpha_adjustment_bps"] == 0.0
    assert metrics["pressure_signal"] == "BALANCED"


def test_orderbook_metrics_balanced_book():
    """Symmetric orderbook with equal bids and asks should produce neutral balanced signal."""
    # Mid-price = 100.0, spread = 0.02 (2 bps)
    depth_data = {
        "source": "coindcx",
        "bids": [(99.99 - i * 0.01, 1000.0) for i in range(20)],
        "asks": [(100.01 + i * 0.01, 1000.0) for i in range(20)],
    }
    metrics = compute_orderbook_metrics(depth_data)

    assert metrics["available"] is True
    assert 48 <= metrics["bid_pct"] <= 52
    assert 48 <= metrics["ask_pct"] <= 52
    assert -2.0 <= metrics["book_imbalance_pct"] <= 2.0
    assert 0.95 <= metrics["depth_ratio"] <= 1.05
    assert metrics["pressure_signal"] == "BALANCED"
    assert -1.5 <= metrics["depth_alpha_adjustment_bps"] <= 1.5


def test_orderbook_metrics_strong_bid_wall():
    """Heavy bid accumulation wall should generate buy pressure and positive alpha tilt."""
    # 5x larger bid volume than asks
    depth_data = {
        "source": "coindcx",
        "bids": [(99.99 - i * 0.01, 5000.0) for i in range(20)],
        "asks": [(100.01 + i * 0.01, 1000.0) for i in range(20)],
    }
    metrics = compute_orderbook_metrics(depth_data)

    assert metrics["available"] is True
    assert metrics["bid_pct"] >= 75
    assert metrics["book_imbalance_pct"] >= 50.0
    assert metrics["depth_ratio"] >= 4.0
    assert metrics["pressure_signal"] == "STRONG_BUY_PRESSURE"
    assert metrics["depth_alpha_adjustment_bps"] > 5.0
    assert "Bid" in metrics["summary"]


def test_orderbook_metrics_strong_ask_wall():
    """Heavy ask resistance overhead should generate sell pressure and negative alpha drag."""
    # 4x larger ask volume than bids
    depth_data = {
        "source": "binance_fallback",
        "bids": [(0.02250 - i * 0.00001, 2000.0) for i in range(20)],
        "asks": [(0.02251 + i * 0.00001, 8000.0) for i in range(20)],
    }
    metrics = compute_orderbook_metrics(depth_data)

    assert metrics["available"] is True
    assert metrics["ask_pct"] >= 75
    assert metrics["book_imbalance_pct"] <= -50.0
    assert metrics["depth_ratio"] <= 0.35
    assert metrics["pressure_signal"] == "STRONG_SELL_PRESSURE"
    assert metrics["depth_alpha_adjustment_bps"] < -5.0
    assert "Ask" in metrics["summary"]


def test_orderbook_microprice_skew():
    """When top bid size heavily exceeds top ask size, microprice should lead above mid-price."""
    # Equal depth further back, but top bid is 10x top ask
    bids = [(10.00, 10000.0)] + [(9.99 - i * 0.01, 1000.0) for i in range(19)]
    asks = [(10.02, 1000.0)] + [(10.03 + i * 0.01, 1000.0) for i in range(19)]

    metrics = compute_orderbook_metrics({"bids": bids, "asks": asks})

    assert metrics["microprice_skew_bps"] > 0.0
    assert metrics["spread_bps"] > 0.0


def test_orderbook_integration_in_forecast_event():
    """ForecastEvent should accept and validate orderbook_depth payload."""
    event = ForecastEvent(
        timestamp=datetime.now(timezone.utc),
        symbol="B-DOGE_USDT",
        benchmark_symbol="B-BTC_USDT",
        current_price=0.0825,
        benchmark_price=68000.0,
        expected_residual_alpha_15m_bps=22.5,
        z_score=1.45,
        beta_hedge_ratio=1.65,
        signal=SignalState.BULLISH_RESIDUAL,
        confidence_percentile=88.5,
        orderbook_depth={
            "available": True,
            "bid_pct": 65,
            "ask_pct": 35,
            "book_imbalance_pct": 30.0,
            "depth_alpha_adjustment_bps": 7.5,
            "pressure_signal": "STRONG_BUY_PRESSURE",
        },
        execution_disabled=True,
    )

    assert event.orderbook_depth["available"] is True
    assert event.orderbook_depth["bid_pct"] == 65
    assert event.orderbook_depth["pressure_signal"] == "STRONG_BUY_PRESSURE"
