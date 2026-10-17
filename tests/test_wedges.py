"""Wedge breakouts with a retest: the rules that find them, what their trades did, and their backtest."""

import numpy as np
import pandas as pd
import pytest

from deltarisk.backtest import BacktestConfig, backtest_wedge_candles, format_report
from deltarisk.features.wedges import WedgeSpec, average_true_range, find_wedge_setups
from deltarisk.labels.setups import resolve_trade
from deltarisk.labels.wedges import wedge_outcomes

SPEC = WedgeSpec(minutes=60, swing_len=2, window=40, apex_min=3, apex_max=40, atr_len=3, volume_window=5,
                 retest_window=3, targets=(2.0, "measured"), breakeven_r=None, max_hold=30)
# Resistance 113 - 0.5 (x - 14) through the highs at candles 2 and 14, support 98 - 0.1 (x - 20) through the
# lows at 8 and 20: a falling wedge whose lines meet at candle 50. Candle 23 breaks out on 4x volume.
WEDGE_TURNS = [(0, 110.0), (2, 119.0), (8, 99.2), (14, 113.0), (20, 98.0), (22, 102.0), (23, 110.0)]
AFTER = [110.4, 113.0, 116.0, 119.0, 112.0, 105.0, 103.0]  # retest and trigger, up past 2R, then down to the stop


def _candles(after=AFTER, high=None, low=None, breakout_volume=400.0):
    """The wedge, then `after` closes; each candle opens at the last close, with no wicks unless given."""
    xs, ys = zip(*WEDGE_TURNS)
    close = np.r_[np.interp(np.arange(24), xs, ys), after]
    open_ = np.r_[close[0], close[:-1]]
    h, l = np.maximum(open_, close), np.minimum(open_, close)
    for i, v in (high or {}).items():
        h[i] = v
    for i, v in (low or {}).items():
        l[i] = v
    volume = np.full(len(close), 100.0)
    volume[23] = breakout_volume
    idx = pd.date_range("2026-03-02", periods=len(close), freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": h, "low": l, "close": close, "volume": volume}, index=idx)


def _retested():
    # Candle 24 dips to 108.1, within 0.2% of the broken line (108 there), and closes near its high
    return _candles(high={24: 110.6}, low={24: 108.1})


def _mirror(df, around=220.0):
    return pd.DataFrame({"open": around - df["open"], "high": around - df["low"], "low": around - df["high"],
                         "close": around - df["close"], "volume": df["volume"]}, index=df.index)


def _without_attrs(df):
    df = df.copy()
    df.attrs = {}
    return df


def test_falling_wedge_breakout_retest_and_trigger():
    candles = _retested()
    setups = find_wedge_setups(candles, SPEC)
    assert setups.attrs["funnel"] == {"wedges": 1, "breakouts": 1, "failed": 0, "expired": 0, "triggered": 1}
    assert len(setups) == 1
    s = setups.iloc[0]
    assert (s["high_1_idx"], s["high_2_idx"], s["low_1_idx"], s["low_2_idx"]) == (2, 14, 8, 20)
    assert s["res_slope"] == pytest.approx(-0.5) and s["sup_slope"] == pytest.approx(-0.1)
    assert s["shape"] == "falling wedge" and s["apex_ahead"] == pytest.approx(27.0)  # gap 10.8 closing 0.4 a candle
    assert (s["res_touches"], s["sup_touches"]) == (2, 2)
    assert s["breakout_idx"] == 23 and s["breakout_close"] == 110.0  # above the line's 108.5
    assert s["volume_ratio"] == pytest.approx(2.5) and s["body_atr"] > 1 and s["grade"] == "A+"

    assert s["idx"] == s["retest_idx"] == 24 and s["side"] == 1
    assert s["entry"] == 110.4  # the trigger candle's close
    atr = average_true_range(*(candles[k].to_numpy() for k in ("high", "low", "close")), 3)[24]
    assert s["line"] == pytest.approx(108.0) and s["stop"] == pytest.approx(108.0 - 0.5 * atr)  # below the line
    R = 110.4 - s["stop"]
    assert s["height"] == pytest.approx(19.2) and s["measured_r"] == pytest.approx(19.2 / R)  # 119 - 99.8 at candle 2

    out = wedge_outcomes(setups, candles["high"], candles["low"], candles["close"], SPEC).iloc[0]
    assert (out["fill"], out["fill_idx"]) == ("filled", 24)
    assert 119.0 >= 110.4 + 2 * R > 116.0  # candle 27 reaches 2R ...
    assert (out["outcome_2"], out["r_2"], out["end_2"]) == ("won", 2.0, 27)
    # ... but never the measured move (129.6), and candle 29's low of 105 takes the stop
    assert (out["outcome_measured"], out["r_measured"], out["end_measured"]) == ("lost", -1.0, 29)


def test_shorts_mirror_longs_exactly():
    candles = _retested()
    long_ = find_wedge_setups(candles, SPEC).iloc[0]
    mirrored = _mirror(candles)
    shorts = find_wedge_setups(mirrored, SPEC)
    assert len(shorts) == 1
    short = shorts.iloc[0]
    assert short["side"] == -1 and short["shape"] == "rising wedge" and short["grade"] == "A+"
    assert (short["low_1_idx"], short["low_2_idx"], short["high_1_idx"], short["high_2_idx"]) == (2, 14, 8, 20)
    for col in ("idx", "breakout_idx", "retest_idx"):
        assert short[col] == long_[col]
    for col in ("entry", "stop", "line"):
        assert short[col] == pytest.approx(220.0 - long_[col])
    for col in ("height", "measured_r", "apex_ahead", "volume_ratio", "body_atr"):
        assert short[col] == pytest.approx(long_[col])

    o_long = wedge_outcomes(find_wedge_setups(candles, SPEC), candles["high"], candles["low"], candles["close"], SPEC)
    o_short = wedge_outcomes(shorts, mirrored["high"], mirrored["low"], mirrored["close"], SPEC)
    pd.testing.assert_frame_equal(o_long, o_short)


def test_a_close_back_inside_fails_the_breakout():
    # Candle 24 closes at 107, more than 0.2% under the line (108): the breakout failed
    candles = _candles(after=[107.0, 105.0, 104.0, 103.0], low={24: 106.9})
    setups = find_wedge_setups(candles, SPEC)
    assert setups.empty and setups.attrs["funnel"]["failed"] == 1


def test_a_breakout_that_never_comes_back_expires():
    candles = _candles(after=[113.0, 115.0, 117.0, 119.0, 121.0])  # lows stay far above the line
    setups = find_wedge_setups(candles, SPEC)
    assert setups.empty and setups.attrs["funnel"]["expired"] == 1


def test_the_trigger_may_come_one_candle_after_the_retest():
    # Candle 24 touches the line but closes down; candle 25 closes up near its high
    candles = _candles(after=[109.0, 110.2, 114.0, 118.0, 121.0], low={24: 108.05, 25: 108.6}, high={25: 110.3})
    s = find_wedge_setups(candles, SPEC).iloc[0]
    assert (s["retest_idx"], s["idx"], s["entry"]) == (24, 25, 110.2)
    atr = average_true_range(*(candles[k].to_numpy() for k in ("high", "low", "close")), 3)[25]
    assert s["line"] == pytest.approx(107.5)  # below the retest's low of 108.05
    assert s["stop"] == pytest.approx(107.5 - 0.5 * atr)


def test_a_breakout_without_volume_is_grade_b():
    s = find_wedge_setups(_candles(high={24: 110.6}, low={24: 108.1}, breakout_volume=100.0), SPEC).iloc[0]
    assert s["volume_ratio"] == pytest.approx(1.0) and s["grade"] == "B"


def test_wedges_never_look_ahead():
    rng = np.random.default_rng(7)
    n = 4000
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.002, (2, n)))
    candles = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + wick[0]),
                            "low": np.minimum(open_, close) * (1 - wick[1]), "close": close,
                            "volume": rng.lognormal(0, 0.5, n)},
                           index=pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC"))
    spec = WedgeSpec()
    full = find_wedge_setups(candles, spec)
    assert len(full) >= 5 and set(full["shape"]) <= {"rising wedge", "falling wedge", "triangle"}
    for cut in (int(full["idx"].iat[2]) + 1, int(full["idx"].iat[-1]), n // 2):
        part = find_wedge_setups(candles.iloc[:cut], spec)
        pd.testing.assert_frame_equal(_without_attrs(part), _without_attrs(full[full["idx"] < cut]))


def test_breakeven_stop_and_market_entries():
    def trade(high, low, close=None, be=(1.5, 100.1), at_close=False, max_hold=10):
        h, l = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
        c = np.asarray(close if close is not None else (h + l) / 2, dtype=float)
        return resolve_trade(h, l, c, 0, 100.0, 99.0, 1, 2.0, max_hold, at_close=at_close, breakeven=be)

    # Reaches 1.5R on candle 1, then comes back to the new stop on candle 2: out at break-even plus fees
    assert trade([100.5, 101.6, 101.0, 103.0], [99.5, 100.6, 100.05, 101.0]) == ("breakeven", pytest.approx(0.1), 2)
    # Without the move the same trade runs on to 2R
    assert trade([100.5, 101.6, 101.0, 103.0], [99.5, 100.6, 100.05, 101.0], be=None) == ("won", 2.0, 3)
    # The candle that reaches 1.5R also reaches 2R: a win
    assert trade([100.5, 102.2], [99.5, 100.6]) == ("won", 2.0, 1)
    # The candle that reaches 1.5R also hits the original stop: a loss, the stop had not moved yet
    assert trade([100.5, 101.6], [99.5, 98.9]) == ("lost", -1.0, 1)
    # After the move, the take-profit and the new stop in one candle: counted as the stop
    assert trade([100.5, 101.6, 102.5], [99.5, 100.6, 100.0])[0] == "breakeven"
    # A fill at the candle's close: that candle cannot stop the trade, the next one starts it
    assert trade([100.5, 101.0, 102.5], [98.0, 100.2, 100.5], be=None, at_close=True) == ("won", 2.0, 2)
    assert trade([100.5, 101.0, 102.5], [98.0, 100.2, 100.5], be=None) == ("lost", -1.0, 0)
    # Neither level within max_hold: closed at candle fill + max_hold
    assert trade([100.5, 101.0, 101.0], [99.5, 99.5, 99.5], [100.0, 100.4, 100.6], be=None, at_close=True,
                 max_hold=2) == ("timeout", pytest.approx(0.6), 2)


def test_wedge_costs_start_with_a_market_order():
    spec = WedgeSpec(maker_bps=2.0, taker_bps=5.0, slippage_bps=3.0)
    assert spec.round_trip("won") == pytest.approx(7e-4)  # taker in, maker take-profit out
    assert spec.round_trip("lost") == spec.round_trip("breakeven") == pytest.approx(13e-4)  # taker + slippage out
    assert spec.round_trip("timeout") == pytest.approx(10e-4)
    # The break-even stop covers both legs' fees, so such an exit loses only its slippage
    assert spec.breakeven_stop(100.0, 1) == pytest.approx(100.1) and spec.breakeven_stop(100.0, -1) == pytest.approx(99.9)
    risk = 0.01
    gross = (spec.breakeven_stop(100.0, 1) - 100.0) / (100.0 * risk)
    assert gross - spec.round_trip("breakeven") / risk == pytest.approx(-3e-4 / risk)


def test_wedge_backtest_and_report():
    spec = WedgeSpec(**{**SPEC.__dict__, "maker_bps": 0.0, "taker_bps": 0.0, "slippage_bps": 0.0})
    candles = _retested()
    won = backtest_wedge_candles(candles, spec, BacktestConfig(target=2.0, max_leverage=5.0))
    s = won["summary"]
    assert s["strategy"] == "wedge" and s["funnel"]["triggered"] == 1
    assert (s["trades"], s["wins"], s["total_net_r"]) == (1, 1, 2.0)
    trade = won["trades"].iloc[0]
    assert trade["shape"] == "falling wedge" and trade["hold_hours"] == 3.0  # entered at 24's close, won on 27
    assert trade["leverage"] == pytest.approx(0.01 / trade["stop_pct"] * 100)  # a 3.8% stop needs no leverage
    report = format_report("B-TEST_USDT", won)
    assert report.startswith("B-TEST_USDT · wedge breakouts on 1h · A+ setups · 2R target · 1% risk per trade, "
                             "at most 5x leverage\nmarket entries, stop never moved")
    assert "1 wedges, 1 broke out: 0 closed back inside, 0 found no retest in time, 1 triggered" in report
    assert "falling wedge" in report

    measured = backtest_wedge_candles(candles, spec, BacktestConfig(target="measured"))["summary"]
    assert (measured["losses"], measured["total_net_r"]) == (1, -1.0)
    assert measured["median_target_r"] == pytest.approx(round(19.2 / (110.4 - trade["stop"]), 2))
