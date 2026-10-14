"""Tests for real-time Long / Short trade risk and liquidation hazard evaluation."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from futurecast.features.trade_risk import evaluate_trade_risk
from futurecast.schema import ForecastEvent, SignalState


def _make_dummy_ohlcv(symbol: str, n_rows: int = 120, trend: float = 0.0, volatility: float = 0.01) -> pd.DataFrame:
    """Helper to generate continuous OHLCV data."""
    dates = pd.date_range("2026-10-14 00:00:00", periods=n_rows, freq="1min", tz="UTC")
    np.random.seed(42)
    base_p = 100.0
    returns = np.random.normal(trend, volatility, n_rows)
    prices = base_p * np.cumprod(1 + returns)

    df = pd.DataFrame(index=dates)
    df[f"{symbol}_close"] = prices
    df[f"{symbol}_open"] = np.roll(prices, 1)
    df.iloc[0, df.columns.get_loc(f"{symbol}_open")] = base_p
    df[f"{symbol}_high"] = np.maximum(df[f"{symbol}_close"], df[f"{symbol}_open"]) * (1 + np.random.uniform(0.001, 0.005, n_rows))
    df[f"{symbol}_low"] = np.minimum(df[f"{symbol}_close"], df[f"{symbol}_open"]) * (1 - np.random.uniform(0.001, 0.005, n_rows))
    df[f"{symbol}_volume"] = np.random.uniform(500, 2000, n_rows)
    return df


def test_trade_risk_empty_dataframe_fallback():
    """Empty dataframe should gracefully return baseline 50 score without crashing."""
    empty_df = pd.DataFrame()
    risk = evaluate_trade_risk(empty_df, "B-TEST_USDT")

    assert risk["long_risk_score"] == 50
    assert risk["short_risk_score"] == 50
    assert risk["long_risk_level"] == "MODERATE"
    assert risk["short_risk_level"] == "MODERATE"
    assert risk["asymmetric_bias"] == "BALANCED"
    assert "2x-3x" in risk["max_recommended_leverage"]


def test_trade_risk_normal_data():
    """Standard market data should produce bounded risk scores and valid leverage guidance."""
    df = _make_dummy_ohlcv("B-BTC_USDT", n_rows=180)
    risk = evaluate_trade_risk(
        df=df,
        target_symbol="B-BTC_USDT",
        microstructure_ctx={"funding_rate_bps": 1.0, "squeeze_flush_risk": 0.0},
        expected_residual_alpha_bps=5.0,
    )

    assert 0 <= risk["long_risk_score"] <= 100
    assert 0 <= risk["short_risk_score"] <= 100
    assert risk["long_risk_level"] in ["LOW", "MODERATE", "HIGH", "EXTREME"]
    assert risk["short_risk_level"] in ["LOW", "MODERATE", "HIGH", "EXTREME"]
    assert risk["asymmetric_bias"] in ["BALANCED", "FAVORS_LONG", "FAVORS_SHORT", "AVOID_BOTH_HIGH_RISK"]
    assert isinstance(risk["long_hazard_tags"], list)
    assert isinstance(risk["short_hazard_tags"], list)


def test_trade_risk_parabolic_pump_hazard():
    """When a coin has pumped vertically above VWAP with crowded positive funding, Long risk should spike."""
    df = _make_dummy_ohlcv("B-TRUTH_USDT", n_rows=180, trend=0.005)
    # Simulate parabolic vertical climax on the last 5 bars
    close_val = df["B-TRUTH_USDT_close"].iloc[-6] * 1.25
    df.iloc[-5:, df.columns.get_loc("B-TRUTH_USDT_close")] = close_val
    df.iloc[-5:, df.columns.get_loc("B-TRUTH_USDT_high")] = close_val * 1.02

    risk = evaluate_trade_risk(
        df=df,
        target_symbol="B-TRUTH_USDT",
        microstructure_ctx={
            "funding_rate_bps": 12.0,  # Highly crowded longs
            "squeeze_flush_risk": -2.5,  # Long flush danger
            "volume_clock_burst": 2.2,
        },
        expected_residual_alpha_bps=-35.0,  # Negative forward alpha (mean reversion)
    )

    # Long risk score must be elevated due to VWAP overextension and crowded funding
    assert risk["long_risk_score"] >= 65
    assert risk["long_risk_level"] in ["HIGH", "EXTREME"]
    assert any("VWAP" in tag or "Overextended" in tag or "Parabolic" in tag for tag in risk["long_hazard_tags"])
    # Asymmetry should favor short or avoid both
    assert risk["asymmetric_bias"] in ["FAVORS_SHORT", "AVOID_BOTH_HIGH_RISK"]


def test_trade_risk_severe_dump_hazard():
    """When a coin crashes far below VWAP with crowded negative funding, Short risk should spike."""
    df = _make_dummy_ohlcv("B-DUMP_USDT", n_rows=180, trend=-0.005)
    # Simulate capitulation wick at bottom
    dump_val = df["B-DUMP_USDT_close"].iloc[-6] * 0.75
    df.iloc[-5:, df.columns.get_loc("B-DUMP_USDT_close")] = dump_val
    df.iloc[-5:, df.columns.get_loc("B-DUMP_USDT_low")] = dump_val * 0.98

    risk = evaluate_trade_risk(
        df=df,
        target_symbol="B-DUMP_USDT",
        microstructure_ctx={
            "funding_rate_bps": -9.0,  # Crowded shorts paying heavy carry
            "squeeze_flush_risk": 3.0,  # Short squeeze breakout danger
            "volume_clock_burst": 2.5,
        },
        expected_residual_alpha_bps=40.0,  # Bullish mean-reversion alpha
    )

    # Short risk should be elevated due to short squeeze and negative funding
    assert risk["short_risk_score"] >= 65
    assert risk["short_risk_level"] in ["HIGH", "EXTREME"]
    assert risk["asymmetric_bias"] in ["FAVORS_LONG", "AVOID_BOTH_HIGH_RISK"]


def test_trade_risk_integration_in_forecast_event():
    """ForecastEvent schema should accept and validate trade_risk payload."""
    event = ForecastEvent(
        timestamp=datetime.now(timezone.utc),
        symbol="B-TRUTH_USDT",
        benchmark_symbol="B-BTC_USDT",
        current_price=0.0225,
        benchmark_price=68000.0,
        expected_residual_alpha_15m_bps=-27.65,
        z_score=-0.42,
        beta_hedge_ratio=1.85,
        signal=SignalState.NEUTRAL,
        confidence_percentile=74.2,
        trade_risk={
            "long_risk_score": 78,
            "long_risk_level": "HIGH",
            "short_risk_score": 42,
            "short_risk_level": "MODERATE",
            "max_recommended_leverage": "2x-3x (Strict Risk)",
            "asymmetric_bias": "FAVORS_SHORT",
        },
        execution_disabled=True,
    )

    assert event.trade_risk["long_risk_score"] == 78
    assert event.trade_risk["asymmetric_bias"] == "FAVORS_SHORT"
    assert event.execution_disabled is True
