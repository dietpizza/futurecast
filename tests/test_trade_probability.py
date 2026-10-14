"""Tests for take-profit/stop-loss labels and calibrated win-probability models."""

import numpy as np
import pandas as pd
import pytest

from futurecast.labels.barriers import (
    BarrierSpec,
    barrier_distances,
    break_even_probability,
    first_touch_labels,
)
from futurecast.models.trade_probability import (
    TradeModelConfig,
    TradeProbabilityModel,
    _purged_slices,
)


def _bars(highs, lows, closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="1min", tz="UTC")
    return pd.Series(closes, idx, dtype=float), pd.Series(highs, idx, dtype=float), pd.Series(lows, idx, dtype=float)


def test_first_touch_win_loss_timeout_and_same_bar():
    # Entry 100 on bar 0, TP/SL 1% → long TP 101 / SL 99, short TP 99 / SL 101
    cases = {
        "long_wins": ([100, 101.5, 100, 100], [100, 99.5, 100, 100], 1.0, 0.0),
        "short_wins": ([100, 100.5, 100, 100], [100, 98.5, 100, 100], 0.0, 1.0),
        "timeout": ([100, 100.5, 100.5, 100.5], [100, 99.5, 99.5, 99.5], 0.0, 0.0),
        "same_bar_both": ([100, 101.5, 100, 100], [100, 98.5, 100, 100], 0.0, 0.0),
        "tp_before_sl": ([100, 101.5, 100, 100], [100, 99.5, 98.0, 100], 1.0, 0.0),
    }
    for name, (highs, lows, want_long, want_short) in cases.items():
        close, high, low = _bars(highs, lows, [100.0] * 4)
        labels = first_touch_labels(close, high, low, tp=np.full(4, 0.01), sl=np.full(4, 0.01), horizon=3)
        assert labels["long_win"].iloc[0] == want_long, name
        assert labels["short_win"].iloc[0] == want_short, name
        # Outcomes that need bars beyond the data are unknown, never guessed
        assert labels.iloc[1:].isna().all().all(), name


def test_barriers_clear_costs_and_break_even():
    spec = BarrierSpec(tp_vol_mult=0.5, sl_vol_mult=0.5, cost_bps=10.0, min_tp_cost_mult=2.0)
    tp, sl = barrier_distances(np.array([1e-5, 0.02]), spec)
    assert tp[0] == pytest.approx(0.002)  # floored at 2x the 0.10% round-trip cost
    assert sl[0] == pytest.approx(0.002)  # stop keeps the 1:1 reward:risk
    assert tp[1] == pytest.approx(0.01)
    assert break_even_probability(0.01, 0.01, 0.001) == pytest.approx(0.55)


def test_purged_slices_are_ordered_and_gapped():
    cfg = TradeModelConfig()
    s = _purged_slices(10_000, 60, cfg)
    assert s["fit"].stop + 60 <= s["es"].start
    assert s["es"].stop + 60 <= s["calib"].start
    assert s["calib"].stop + 60 <= s["test"].start
    assert s["test"].stop == 10_000


def _synthetic_market(n, predictive, seed):
    rng = np.random.default_rng(seed)
    regime = np.zeros(n)
    for t in range(1, n):  # slow-moving state that sets the drift when predictive
        regime[t] = 0.98 * regime[t - 1] + rng.normal(0, 0.2)
    drift = 0.0006 * regime if predictive else np.zeros(n)
    ret = drift + rng.normal(0, 0.001, n)
    close = 100 * np.exp(np.cumsum(ret))
    wiggle = np.abs(rng.normal(0, 0.0003, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    features = pd.DataFrame({"regime": regime if predictive else rng.normal(0, 1, n),
                             "noise": rng.normal(0, 1, n)}, index=idx)
    return (features, pd.Series(close, idx), pd.Series(close * (1 + wiggle), idx),
            pd.Series(close * (1 - wiggle), idx))


@pytest.fixture(scope="module")
def fast_config():
    return TradeModelConfig(horizons={"15m": 15}, max_rounds=150)


def test_model_finds_real_edge_and_is_calibrated(fast_config):
    features, close, high, low = _synthetic_market(8000, predictive=True, seed=7)
    model = TradeProbabilityModel(fast_config)
    model.fit(features, close, high, low)

    for side in ("long", "short"):
        sm = model.sides["15m"][side]
        assert sm.status == "edge", sm.metrics
        assert sm.metrics["brier_skill"] > 0.05
        assert sm.metrics["auc"] > 0.65
        # Out-of-sample calibration: predicted and observed win rates line up per bin
        for b in sm.metrics["reliability"]:
            assert abs(b["predicted"] - b["observed"]) < 0.15, sm.metrics["reliability"]

    # A strongly positive regime should favour the long side
    up = features.copy()
    up.iloc[-1, up.columns.get_loc("regime")] = 3.0
    out = model.predict_latest(up, close)["15m"]
    assert out["long"]["probability"] > out["short"]["probability"]
    assert out["long"]["probability"] == out["long"]["model_probability"]


def test_model_without_edge_serves_base_rate(fast_config):
    features, close, high, low = _synthetic_market(8000, predictive=False, seed=11)
    model = TradeProbabilityModel(fast_config)
    model.fit(features, close, high, low)
    out = model.predict_latest(features, close)["15m"]

    for side in ("long", "short"):
        sm = model.sides["15m"][side]
        assert sm.metrics["brier_skill"] < 0.02  # nothing to learn from noise
        if sm.status != "edge":
            assert out[side]["probability"] == pytest.approx(sm.base_rate)
    if all(model.sides["15m"][s].status != "edge" for s in ("long", "short")):
        assert out["signal"] == "NEUTRAL"


def test_too_little_history_is_flagged_not_guessed():
    features, close, high, low = _synthetic_market(3000, predictive=True, seed=3)
    model = TradeProbabilityModel(TradeModelConfig(horizons={"4h": 240}, max_rounds=50))
    model.fit(features, close, high, low)
    for side in ("long", "short"):
        assert model.sides["4h"][side].status == "insufficient_data"


def test_save_load_roundtrip(tmp_path, fast_config):
    features, close, high, low = _synthetic_market(4000, predictive=True, seed=5)
    model = TradeProbabilityModel(fast_config)
    model.fit(features, close, high, low)
    path = tmp_path / "trade.joblib"
    model.save(path)
    loaded = TradeProbabilityModel.load(path)
    assert loaded is not None
    a = model.predict_latest(features, close)["15m"]
    b = loaded.predict_latest(features, close)["15m"]
    assert a["long"]["probability"] == pytest.approx(b["long"]["probability"])


def test_shrink_corrects_overconfidence_but_never_sharpens():
    from futurecast.models.trade_probability import _fit_shrink, shrink_toward

    rng = np.random.default_rng(1)
    base = 0.4
    true_p = np.clip(rng.normal(base, 0.08, 20_000), 0.05, 0.95)
    y = (rng.random(true_p.size) < true_p).astype(float)

    # Calibrated odds keep (almost) their full spread
    assert _fit_shrink(true_p, y, base) > 0.9
    # Odds stretched 2x away from the base rate get pulled back about halfway
    logit = lambda p: np.log(p / (1 - p))
    stretched = 1 / (1 + np.exp(-(logit(base) + 2.0 * (logit(true_p) - logit(base)))))
    slope = _fit_shrink(stretched, y, base)
    assert 0.35 < slope < 0.65
    assert np.mean(np.abs(shrink_toward(stretched, base, slope) - true_p)) < np.mean(np.abs(stretched - true_p))
    # Underconfident odds are left alone rather than sharpened
    assert shrink_toward(np.array([0.5]), base, 1.0)[0] == pytest.approx(0.5)
    assert _fit_shrink(base + 0.3 * (true_p - base), y, base) == pytest.approx(1.0, abs=1e-3)
