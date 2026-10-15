"""Fair-value-gap setups: the rules that find them, what their orders did, and the models that judge them."""

import numpy as np
import pandas as pd
import pytest

from futurecast.engine.setups import live_view
from futurecast.features.setups import SetupSpec, find_setups, setup_features
from futurecast.labels.setups import break_even, expected_r, setup_outcomes, trade_cost_r
from futurecast.models.setup_model import SetupModel, SetupModelConfig, TargetModel

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
    assert 106.8 >= 103.15 + 3 * R > 106.0  # candle 15 reaches 3R ...
    assert (out["outcome_3"], out["win_3"], out["r_3"], out["end_3"]) == ("won", 1.0, 3.0, 15)
    # ... but not 4R, and candle 17 hits the stop
    assert (out["outcome_4"], out["win_4"], out["r_4"], out["end_4"]) == ("lost", 0.0, -1.0, 17)


def test_shorts_mirror_longs_exactly():
    ltf, htf = _a_plus_long()
    long_ = find_setups(ltf, htf, SPEC)
    short = find_setups(_mirror(ltf), _mirror(htf), SPEC)
    long_, short = long_[long_["side"] == 1], short[short["side"] == -1]
    assert len(short) == 1 and short.iloc[0]["grade"] == "A+" and short.iloc[0]["entry"] == pytest.approx(206 - 103.15)

    f_long = setup_features(long_, ltf["close"], SPEC)
    f_short = setup_features(short, _mirror(ltf)["close"], SPEC)
    assert f_long["setup_side"].iat[0] == 1 and f_short["setup_side"].iat[0] == -1
    pd.testing.assert_frame_equal(f_long.drop(columns="setup_side"), f_short.drop(columns="setup_side"),
                                  check_exact=False, rtol=1e-2)
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

    # In R, on a stop 0.5% from the entry; orders that never filled or trades still open have no cost yet
    cost = trade_cost_r(np.array(["won", "lost", "timeout", "", "open"]), 0.005, spec)
    np.testing.assert_allclose(cost[:3], [0.08, 0.2, 0.14])
    assert np.isnan(cost[3:]).all()

    # Break-even balances a win's net (3 - 0.08) against a stop's (1 + 0.2): cheaper wins lower it
    assert break_even(3.0, 1.0, 0.08, 0.2) == pytest.approx(1.2 / (2.92 + 1.2))
    assert break_even(3.0, 1.0, 0.08, 0.2) < break_even(3.0, 1.0, 0.2, 0.2) == pytest.approx(1.2 / 4)
    assert expected_r(break_even(3.0, 1.0, 0.08, 0.2), 3.0, 1.0, 0.08, 0.2) == pytest.approx(0.0)


def _synthetic_history(n, predictive, seed):
    """Filled setups whose wins depend on `signal` when predictive, with every second one graded A+."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(-1.4 + (1.6 * signal if predictive else 0))))
    win = (rng.random(n) < p).astype(float)
    idx = np.arange(n) * 10
    times = pd.date_range("2026-01-01", periods=n, freq="10min", tz="UTC")
    setups = pd.DataFrame({"idx": idx, "side": np.where(rng.random(n) < 0.5, 1, -1), "risk": 0.01,
                           "grade": np.where(np.arange(n) % 2 == 0, "A+", "B")}, index=times)
    outcomes = pd.DataFrame({"fill": "filled", "fill_idx": idx + 1, "order_end": -1,
                             "outcome_3": np.where(win == 1, "won", "lost"), "win_3": win,
                             "r_3": np.where(win == 1, 3.0, -1.0), "end_3": idx + 5}, index=times)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(0, 1, n)}, index=times)
    return X, setups, outcomes


@pytest.fixture(scope="module")
def fast_config():
    # On the 1% stops of _synthetic_history a win costs 0.04R, a stop 0.10R
    spec = SetupSpec(ltf_minutes=1, htf_minutes=15, targets=(3.0,), maker_bps=2.0, taker_bps=5.0, slippage_bps=3.0)
    return SetupModelConfig(spec=spec, max_rounds=120)


def test_models_earn_an_edge_only_when_setups_are_predictable(fast_config):
    X, setups, outcomes = _synthetic_history(3000, predictive=True, seed=1)
    model = SetupModel(fast_config)
    model.fit(X, setups, outcomes)
    tm = model.targets["3"]
    assert tm.status == "edge", tm.metrics
    assert tm.metrics["auc"] > 0.7 and tm.metrics["brier_skill_lower"] > 0
    assert tm.model_replay["trades"] >= 10 and tm.model_replay["avg_net_r"] > tm.rules_replay["avg_net_r"]
    assert tm.validated

    scored = model.score(X.iloc[-2:], setups.iloc[-2:])
    t = scored[-1]["targets"][scored[-1]["pick"]]
    assert t["probability_source"] == "model" and t["probability"] == t["model_probability"]
    p, lose = t["probability"], tm.loss_ratio + 0.1
    assert t["break_even"] == pytest.approx(lose / (3 - 0.04 + lose))
    assert t["expected_r"] == pytest.approx(p * (3 - 0.04) - (1 - p) * lose)

    X, setups, outcomes = _synthetic_history(3000, predictive=False, seed=2)
    noise = SetupModel(fast_config)
    noise.fit(X, setups, outcomes)
    tm = noise.targets["3"]
    assert tm.status == "no_edge"
    t = noise.score(X.iloc[-1:], setups.iloc[-1:])[0]["targets"][0]
    # Without an edge the rules' own win rate for the grade is served
    assert t["probability_source"] == "rules" and t["probability"] == tm.base_rates[setups["grade"].iat[-1]]


def test_too_few_setups_is_insufficient_data(fast_config):
    X, setups, outcomes = _synthetic_history(60, predictive=True, seed=3)
    model = SetupModel(fast_config)
    model.fit(X, setups, outcomes)
    assert model.targets["3"].status == "insufficient_data" and not model.targets["3"].models


def test_setup_model_persists(tmp_path, fast_config):
    X, setups, outcomes = _synthetic_history(1500, predictive=True, seed=4)
    model = SetupModel(fast_config)
    model.fit(X, setups, outcomes)
    path = tmp_path / "setup_model_x.joblib"
    model.save(path)
    loaded = SetupModel.load(path)
    assert loaded is not None
    assert loaded.score(X.iloc[-3:], setups.iloc[-3:]) == model.score(X.iloc[-3:], setups.iloc[-3:])


def _aligned(df, symbol="B-TEST_USDT"):
    return df.add_prefix(f"{symbol}_")


def test_live_view_signals_only_waiting_a_plus_setups_with_a_validated_record():
    ltf, htf = _a_plus_long()
    waiting = ltf.iloc[:11]  # the setup candle has just closed; the limit order is waiting

    view = live_view(_aligned(waiting), _aligned(htf), "B-TEST_USDT", None, None, spec=SPEC)
    long_ = view["setups"]["long"]
    assert long_["state"] == "waiting" and long_["grade"] == "A+"
    assert long_["checklist"] == {"htf_trend": True, "htf_zone": True, "ltf_choch": True, "ltf_gap": True}
    assert long_["probability"] is None  # rules only: no chance without a trained model
    assert view["setups"]["signal"] == "NEUTRAL"
    assert view["structure"]["htf"]["trend"] == "up" and view["structure"]["ltf"]["last_break"]["kind"] == "CHoCH"

    def model_with(rate_a_plus, validated):
        m = SetupModel(SetupModelConfig(spec=SPEC))
        m.targets = {key: TargetModel(target=T, status="no_edge", base_rates={"A+": rate_a_plus, "B": 0.1},
                                      validated=validated)
                     for key, T in (("3", 3.0), ("4", 4.0))}
        return m

    feats = pd.DataFrame(index=waiting.index)
    assert live_view(_aligned(waiting), _aligned(htf), "B-TEST_USDT", feats, model_with(0.6, True))["setups"]["signal"] == "LONG"
    # Below break-even, or a replay that did not make money: no signal
    assert live_view(_aligned(waiting), _aligned(htf), "B-TEST_USDT", feats, model_with(0.05, True))["setups"]["signal"] == "NEUTRAL"
    assert live_view(_aligned(waiting), _aligned(htf), "B-TEST_USDT", feats, model_with(0.6, False))["setups"]["signal"] == "NEUTRAL"

    # Once the order has filled it is too late to enter: shown as a trade in progress, no signal
    filled = live_view(_aligned(ltf.iloc[:14]), _aligned(htf), "B-TEST_USDT", pd.DataFrame(index=ltf.index[:14]),
                       model_with(0.6, True))["setups"]
    assert filled["long"]["state"] == "in_trade" and filled["signal"] == "NEUTRAL"
