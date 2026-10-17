"""Momentum and market-state labels on hand-made price paths."""

import numpy as np
import pandas as pd
import pytest

from deltarisk.stats.regime import WINDOW_BY_NAME, WINDOWS, regime_frame, regime_view

SYM = "B-TEST_USDT"
HOUR = WINDOW_BY_NAME["1h"]  # 12 candles of 5m, looking back 36
SIGMA = 0.002


def _frame(log_returns: np.ndarray) -> pd.DataFrame:
    close = 100 * np.exp(np.cumsum(log_returns))
    idx = pd.date_range("2026-01-01", periods=len(close), freq="5min", tz="UTC")
    return pd.DataFrame({f"{SYM}_close": close, f"{SYM}_high": close * 1.001, f"{SYM}_low": close * 0.999}, index=idx)


def _last(tail: np.ndarray) -> pd.Series:
    history = np.random.default_rng(3).normal(0, SIGMA, 4000)
    return regime_frame(_frame(np.r_[history, tail]), SYM, HOUR).iloc[-1]


def test_windows_cover_5m_to_8h():
    assert [w.name for w in WINDOWS] == ["5m", "15m", "30m", "1h", "2h", "3h", "4h", "8h"]
    assert all(w.minutes == int(w.name[:-1]) * (60 if w.name.endswith("h") else 1) for w in WINDOWS)


def test_straight_rise_is_trending_and_bullish():
    last = _last(np.full(36, SIGMA))
    assert last["state"] == "trending" and last["momentum_label"] == "bullish"
    assert last["efficiency"] == pytest.approx(1.0) and last["momentum"] == 100.0


def test_straight_fall_is_bearish():
    last = _last(np.full(36, -SIGMA))
    assert last["state"] == "trending" and last["momentum_label"] == "bearish"


def test_zigzag_is_choppy():
    last = _last(np.tile([SIGMA, -SIGMA], 18))
    assert last["state"] == "choppy" and last["efficiency"] < 0.1 and last["momentum_label"] == "neutral"


def test_flat_is_quiet():
    last = _last(np.tile([SIGMA / 50, -SIGMA / 50], 18))
    assert last["state"] == "quiet" and last["volatility_label"] == "calm"


def test_spike_is_wild():
    last = _last(np.tile([6 * SIGMA, -6 * SIGMA], 18))
    assert last["state"] == "wild" and last["volatility_label"] == "wild"


def test_no_reading_without_history_and_none_for_missing_charts():
    assert pd.isna(regime_frame(_frame(np.full(20, SIGMA)), SYM, HOUR)["state"].iloc[-1])
    view = regime_view({"5m": _frame(np.random.default_rng(1).normal(0, SIGMA, 4000))}, SYM)
    assert view["8h"] is None and view["2h"] is None
    assert view["1h"]["state"] in {"trending", "choppy", "quiet", "wild"}
    assert -100 <= view["5m"]["momentum"] <= 100


def test_trend_record_counts_what_followed_trending_readings():
    from deltarisk.stats.regime import trend_record

    # A random history, then a long straight rise: its trending-bullish readings are followed by more rise
    returns = np.r_[np.random.default_rng(5).normal(0, SIGMA, 4000), np.full(200, SIGMA)]
    frame = _frame(returns)
    reg = regime_frame(frame, SYM, HOUR)
    out = trend_record(frame, SYM, HOUR, reg, 0.177)
    assert out["round_trip_pct"] == 0.177
    assert out["long"]["n"] > 100 and out["long"]["mean_pct"] > out["base"]["mean_pct"]
    assert out["long"]["win_rate"] > out["base"]["win_rate"]
    assert out["base"]["n"] == int(reg["state"].notna()[: -HOUR.candles].sum())
    assert 0 <= out["short"]["n"] < out["long"]["n"]
    assert out["short"]["n"] == 0 or 0 <= out["short"]["win_rate"] <= 1
