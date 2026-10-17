"""Fair-value-gap setups: the rules that find them, what their orders did, and the models that judge them."""

import numpy as np
import pandas as pd
import pytest

from deltarisk.features.setups import SetupSpec, find_setups
from deltarisk.labels.setups import break_even, setup_outcomes

SPEC = SetupSpec(ltf_minutes=1, htf_minutes=15, ltf_swing_len=1, htf_swing_len=1, fill_window=5, max_hold=20)
START = pd.Timestamp("2026-03-02 00:00", tz="UTC")  # a Monday


def _frame(high, low, close, start, minutes):
    idx = pd.date_range(start, periods=len(close), freq=f"{minutes}min", tz="UTC")
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": high, "low": low, "close": close}, index=idx)


def _a_plus_long():
    """15m: an up break and a bullish gap at 101.2-102.5. 1m: a pullback into it, then a CHoCH up with a gap."""
    htf = _frame(
        high=[100.5, 101.5, 101.0, 101.2, 104.0, 105.0, 105.0, 104.5, 104.8, 105.2],
        low=[99.0, 99.5, 100.0, 100.2, 101.0, 102.5, 103.0, 102.6, 103.5, 104.0],
        close=[100.0, 101.0, 100.5, 100.8, 103.8, 104.8, 104.0, 103.0, 104.5, 105.0],
        start=START, minutes=15)
    #       0      1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16     17
    high = [104.0, 104.4, 104.1, 103.9, 103.8, 103.3, 103.0, 102.8, 103.0, 104.2, 104.9, 104.6, 104.0, 105.0, 106.0, 106.8, 106.5, 104.0]
    low = [103.5, 103.6, 103.2, 103.4, 103.0, 102.6, 102.3, 102.0, 102.4, 102.95, 103.3, 103.9, 103.1, 103.2, 104.8, 105.8, 104.0, 101.5]
    close = [103.8, 104.0, 103.4, 103.7, 103.1, 102.7, 102.4, 102.6, 102.9, 104.1, 104.8, 104.0, 103.3, 104.9, 105.9, 106.7, 104.2, 101.6]
    ltf = _frame(high, low, close, start=START + pd.Timedelta(minutes=90), minutes=1)  # after the 15m gap formed
    return ltf, htf


def _mirror(df, around=206.0):
    """Upside down around a level close to the prices, so percentages barely change."""
    return pd.DataFrame({"open": around - df["open"], "high": around - df["low"],
                         "low": around - df["high"], "close": around - df["close"]}, index=df.index)


def test_a_plus_long_setup_orders_and_outcomes():
    ltf, htf = _a_plus_long()
    setups = find_setups(ltf, htf, SPEC)
    longs = setups[setups["side"] == 1]
    assert len(longs) == 1
    s = longs.iloc[0]
    assert s["idx"] == 10 and s["break_idx"] == 10 and s["break_level"] == 104.4  # close 104.8 clears the swing high
    assert s["gap_bottom"] == 103.0 and s["gap_top"] == 103.3  # candle 8's high to candle 10's low
    assert s["entry"] == pytest.approx(103.15)  # the gap's midpoint
    assert s["extreme"] == 102.0 and s["stop"] == pytest.approx(102.0 * (1 - 2e-4))  # just under the pullback's low
    assert s["htf_trend"] == 1 and s["htf_bias"] and s["htf_zone"] and s["grade"] == "A+"
    assert (s["zone_bottom"], s["zone_top"]) == (101.2, 102.5)

    out = setup_outcomes(longs, ltf["high"], ltf["low"], ltf["close"], SPEC).iloc[0]
    assert out["fill"] == "filled" and out["fill_idx"] == 12  # low 103.1 trades through 103.15
    R = 103.15 - s["stop"]
    assert 106.0 >= 103.15 + 2 * R  # candle 14 reaches 2R ...
    assert (out["outcome_2"], out["r_2"], out["end_2"]) == ("won", 2.0, 14)
    assert 106.8 >= 103.15 + 3 * R > 106.0  # ... candle 15 reaches 3R ...
    assert (out["outcome_3"], out["win_3"], out["r_3"], out["end_3"]) == ("won", 1.0, 3.0, 15)
    # ... but not 4R, and candle 17 hits the stop
    assert (out["outcome_4"], out["win_4"], out["r_4"], out["end_4"]) == ("lost", 0.0, -1.0, 17)


def test_shorts_mirror_longs_exactly():
    ltf, htf = _a_plus_long()
    long_ = find_setups(ltf, htf, SPEC)
    short = find_setups(_mirror(ltf), _mirror(htf), SPEC)
    long_, short = long_[long_["side"] == 1], short[short["side"] == -1]
    assert len(short) == 1 and short.iloc[0]["grade"] == "A+" and short.iloc[0]["entry"] == pytest.approx(206 - 103.15)

    o_long = setup_outcomes(long_, ltf["high"], ltf["low"], ltf["close"], SPEC)
    m = _mirror(ltf)
    o_short = setup_outcomes(short, m["high"], m["low"], m["close"], SPEC)
    pd.testing.assert_frame_equal(o_long, o_short)


def test_without_the_15m_context_a_setup_is_grade_b():
    ltf, htf = _a_plus_long()
    flat = htf.copy()
    flat[["high", "low", "close"]] = 90.0  # no trend, no gaps on the 15m chart
    s = find_setups(ltf, flat, SPEC)
    s = s[s["side"] == 1].iloc[0]
    assert not s["htf_bias"] and not s["htf_zone"] and s["grade"] == "B"


def _orders(rows):
    return pd.DataFrame(rows, columns=["idx", "side", "entry", "stop"])


def test_order_and_trade_edge_cases():
    spec = SetupSpec(ltf_minutes=1, htf_minutes=15, fill_window=3, max_hold=4, targets=(3.0,))
    n = 12
    high, low, close = np.full(n, 101.0), np.full(n, 100.5), np.full(n, 100.8)
    candles = [pd.Series(a) for a in (high, low, close)]

    # Never trades back to the entry within the window: expired
    assert setup_outcomes(_orders([[0, 1, 99.0, 98.0]]), *candles, spec)["fill"].iat[0] == "expired"
    # Window still running at the end of the data: pending
    assert setup_outcomes(_orders([[10, 1, 99.0, 98.0]]), *candles, spec)["fill"].iat[0] == "pending"

    # Runs to 3R before filling: cancelled
    l = np.full(n, 100.6); h = high.copy(); h[1] = 103.5
    out = setup_outcomes(_orders([[0, 1, 100.0, 99.0]]), pd.Series(h), pd.Series(l), pd.Series(close), spec)
    assert (out["fill"].iat[0], out["order_end"].iat[0]) == ("cancelled", 1)

    # Filled, then neither level within max_hold: timeout at the close max_hold candles later
    l = np.full(n, 100.6); l[1] = 99.9
    out = setup_outcomes(_orders([[0, 1, 100.0, 99.0]]), pd.Series(high), pd.Series(l), pd.Series(close), spec)
    assert out["outcome_3"].iat[0] == "timeout" and out["end_3"].iat[0] == 5 and out["r_3"].iat[0] == pytest.approx(0.8)

    # The fill candle itself reaches the stop: a loss even if it also reached the target
    h = high.copy(); h[1] = 104.0; l = np.full(n, 100.6); l[1] = 98.9
    out = setup_outcomes(_orders([[0, 1, 100.0, 99.0]]), pd.Series(h), pd.Series(l), pd.Series(close), spec)
    assert out["outcome_3"].iat[0] == "lost" and out["end_3"].iat[0] == 1

    # Target and stop inside one later candle: ambiguous, counted as a loss
    h = high.copy(); h[2] = 104.0; l = np.full(n, 100.6); l[1] = 99.9; l[2] = 98.9
    out = setup_outcomes(_orders([[0, 1, 100.0, 99.0]]), pd.Series(h), pd.Series(l), pd.Series(close), spec)
    assert out["outcome_3"].iat[0] == "lost"

    # Filled, but the hold window runs past the data: still open, never guessed
    l = np.full(n, 100.6); l[9] = 99.9
    out = setup_outcomes(_orders([[8, 1, 100.0, 99.0]]), pd.Series(high), pd.Series(l), pd.Series(close), spec)
    assert out["outcome_3"].iat[0] == "open" and np.isnan(out["win_3"].iat[0])


def test_costs_depend_on_how_a_trade_ends():
    spec = SetupSpec(maker_bps=2.0, taker_bps=5.0, slippage_bps=3.0)
    # Limit entry (maker), then a limit take-profit (maker), a stop (taker + slippage) or a timeout exit (taker)
    assert spec.round_trip("won") == pytest.approx(4e-4)
    assert spec.round_trip("lost") == pytest.approx(10e-4)
    assert spec.round_trip("timeout") == pytest.approx(7e-4)
    assert spec.costs() == pytest.approx({"maker_pct": 0.02, "taker_pct": 0.05, "slippage_pct": 0.03,
                                          "won_pct": 0.04, "lost_pct": 0.1, "timeout_pct": 0.07})

    # Break-even balances a win's net (3 - 0.08) against a stop's (1 + 0.2): cheaper wins lower it
    assert break_even(3.0, 1.0, 0.08, 0.2) == pytest.approx(1.2 / (2.92 + 1.2))
    assert break_even(3.0, 1.0, 0.08, 0.2) < break_even(3.0, 1.0, 0.2, 0.2) == pytest.approx(1.2 / 4)

