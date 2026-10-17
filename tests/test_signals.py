"""Supporting signals: RSI, the moving-average stack, ranges, volatility, volume and funding."""

import numpy as np
import pandas as pd
import pytest

from futurecast.stats.signals import (
    ema_stack,
    funding,
    range_position,
    realized_vol,
    relative_volume,
    rsi,
    signals_view,
)

SYM = "B-TEST_USDT"


def _series(values):
    return pd.Series(np.asarray(values, dtype=float))


def test_rsi_bounds_and_balance():
    assert rsi(_series(np.linspace(100, 200, 100))) == 100.0
    assert rsi(_series(np.linspace(200, 100, 100))) == pytest.approx(0.0)
    assert rsi(_series(100 + np.tile([1.0, -1.0], 100).cumsum())) == pytest.approx(50.0, abs=3)
    assert rsi(_series([1.0] * 10)) is None  # too few candles


def test_ema_stack_orders():
    up = ema_stack(_series(100 * np.exp(np.linspace(0, 0.5, 800))))
    down = ema_stack(_series(100 * np.exp(np.linspace(0.5, 0, 800))))
    assert up["stack"] == "up" and up["vs_slowest_pct"] > 0
    assert down["stack"] == "down" and down["vs_slowest_pct"] < 0
    assert set(up["ema"]) == {"20", "50", "200"}
    assert ema_stack(_series(np.ones(100))) is None


def test_range_position():
    high, low = _series([10, 12, 11, 9]), _series([8, 9, 7, 8])
    assert range_position(high, low, _series([9, 11, 8, 9.5]), 4) == {"high": 12.0, "low": 7.0, "position": 0.5}
    assert range_position(high, low, _series([9, 11, 8, 9.5]), 5) is None


def test_realized_vol_against_earlier_days():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.002, 96 * 60)
    r[-96:] *= 3  # the last day three times as wild
    out = realized_vol(_series(100 * np.exp(np.cumsum(r))), 96)
    assert out["daily_pct"] == pytest.approx(100 * 0.006 * np.sqrt(96), rel=0.2)
    assert out["percentile"] > 99


def test_relative_volume_against_the_median_day():
    hourly = np.r_[np.full(24 * 40, 1000.0), np.full(24, 2500.0)]
    out = relative_volume(_series(hourly), 24)
    assert out == {"last_24h_usd": 60_000.0, "typical_24h_usd": 24_000.0, "ratio": 2.5}


def test_funding_against_its_recent_range():
    rates = _series(np.r_[np.full(24 * 89, 1e-4), np.full(24, 5e-4)])
    out = funding(rates, 24)
    assert out["rate_pct"] == pytest.approx(0.05) and out["percentile"] > 98 and out["days"] == 90
    assert funding(_series([np.nan] * 10), 24) is None


def test_view_reports_none_for_what_is_missing():
    idx = pd.date_range("2026-01-01", periods=800, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.003, 800)))
    hourly = pd.DataFrame({f"{SYM}_close": close, f"{SYM}_high": close * 1.002, f"{SYM}_low": close * 0.998,
                           f"{SYM}_quote_volume": 1000.0}, index=idx)
    out = signals_view({"1h": hourly}, SYM)
    assert out["rsi"]["15m"] is None and 0 <= out["rsi"]["1h"] <= 100
    assert out["ranges"]["24h"] is None and 0 <= out["ranges"]["30d"]["position"] <= 1
    assert out["volatility"] is None and out["funding"] is None
    assert out["ema"]["interval"] == "1h" and out["volume"]["ratio"] == pytest.approx(1.0)
