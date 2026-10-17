"""The funding rate reading."""

import numpy as np
import pandas as pd
import pytest

from futurecast.stats.signals import funding, funding_view, percentile_of

SYM = "B-TEST_USDT"


def test_funding_against_its_recent_range():
    rates = pd.Series(np.r_[np.full(24 * 89, 1e-4), np.full(24, 5e-4)])
    out = funding(rates, 24)
    assert out["rate_pct"] == pytest.approx(0.05) and out["percentile"] > 98 and out["days"] == 90
    assert funding(pd.Series([np.nan] * 10), 24) is None


def test_view_needs_the_hourly_funding_column():
    idx = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    hourly = pd.DataFrame({f"{SYM}_close": 1.0, f"{SYM}_funding_rate": 1e-4}, index=idx)
    assert funding_view({"1h": hourly}, SYM)["rate_pct"] == pytest.approx(0.01)
    assert funding_view({"1h": hourly.drop(columns=f"{SYM}_funding_rate")}, SYM) is None
    assert funding_view({"15m": hourly}, SYM) is None


def test_percentile_counts_ties_half():
    assert percentile_of(np.array([1.0, 2.0, 3.0, 4.0]), 3.0) == 62.5
    assert percentile_of(np.array([np.nan]), 1.0) is None
