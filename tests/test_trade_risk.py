"""Tests for adverse-excursion labels and the model-based long/short trade risk."""

import numpy as np
import pandas as pd
import pytest

from futurecast.labels.barriers import adverse_excursion
from futurecast.models.trade_risk import TradeRiskConfig, TradeRiskModel, max_safe_leverage, risk_level


def _bars(highs, lows, closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="1min", tz="UTC")
    return pd.Series(closes, idx, dtype=float), pd.Series(highs, idx, dtype=float), pd.Series(lows, idx, dtype=float)


def test_adverse_excursion_per_side_and_unknown_tail():
    # Entry 100 on bar 0; the next three bars dip to 97 and spike to 102
    close, high, low = _bars([100, 101, 102, 100], [100, 99, 97, 99.5], [100, 100, 100, 100])
    mae = adverse_excursion(close, high, low, horizon=3)
    assert mae["long_mae"].iloc[0] == pytest.approx(0.03)
    assert mae["short_mae"].iloc[0] == pytest.approx(0.02)
    # Windows that run past the data are unknown, never guessed
    assert mae.iloc[1:].isna().all().all()

    # A side the price never moved against has zero excursion
    close, high, low = _bars([100, 103, 104], [100, 100, 101], [100, 102, 103])
    assert adverse_excursion(close, high, low, horizon=2)["long_mae"].iloc[0] == 0.0


def _volatile_market(n, predictive, seed):
    """Minute bars whose upcoming volatility is set by a slow `regime` state when predictive."""
    rng = np.random.default_rng(seed)
    regime = np.zeros(n)
    for t in range(1, n):
        regime[t] = 0.98 * regime[t - 1] + rng.normal(0, 0.2)
    vol = 0.001 * np.exp(1.0 * regime) if predictive else np.full(n, 0.001)
    ret = rng.normal(0, 1, n) * np.roll(vol, -1)  # bar t+1's size is known from the state at t
    close = 100 * np.exp(np.cumsum(ret))
    wiggle = np.abs(rng.normal(0, 0.0003, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    features = pd.DataFrame({"regime": regime if predictive else rng.normal(0, 1, n),
                             "noise": rng.normal(0, 1, n)}, index=idx)
    return (features, pd.Series(close, idx), pd.Series(close * (1 + wiggle), idx),
            pd.Series(close * (1 - wiggle), idx))


@pytest.fixture(scope="module")
def fast_config():
    return TradeRiskConfig(horizons={"15m": 15}, max_rounds=150)


def test_model_beats_volatility_baseline_and_covers_95(fast_config):
    features, close, high, low = _volatile_market(10_000, predictive=True, seed=7)
    model = TradeRiskModel(fast_config)
    model.fit(features, close, high, low)

    for side in ("long", "short"):
        m = model.sides["15m"][side].metrics
        assert model.sides["15m"][side].status == "edge", m
        assert m["pinball_skill_lower"] > 0
        assert 0.92 <= m["coverage"] <= 0.99, m  # the bound holds about 95% of the time on unseen data

    calm, wild = features.copy(), features.copy()
    calm.iloc[-1, calm.columns.get_loc("regime")] = -2.5
    wild.iloc[-1, wild.columns.get_loc("regime")] = 2.5
    out_calm = model.predict_latest(calm, close)["15m"]
    out_wild = model.predict_latest(wild, close)["15m"]
    for side in ("long", "short"):
        c, w = out_calm[side], out_wild[side]
        assert w["worst_move_pct"] > 1.5 * c["worst_move_pct"]
        assert w["score"] > c["score"]
        assert w["worst_move_pct"] == w["model_worst_move_pct"]  # proven models are served
        assert set(w["per_model_pct"]) == {"LightGBM", "XGBoost", "CatBoost"}


def test_model_without_signal_serves_volatility_baseline(fast_config):
    features, close, high, low = _volatile_market(8000, predictive=False, seed=11)
    model = TradeRiskModel(fast_config)
    model.fit(features, close, high, low)
    out = model.predict_latest(features, close)["15m"]

    for side in ("long", "short"):
        sm = model.sides["15m"][side]
        assert sm.metrics["pinball_skill"] < 0.05  # nothing to learn from noise
        assert 0.92 <= sm.metrics["coverage_baseline"] <= 0.99
        if sm.status != "edge":
            assert out[side]["worst_move_pct"] == pytest.approx(out[side]["baseline_worst_move_pct"])


def test_too_little_history_is_flagged_not_guessed():
    features, close, high, low = _volatile_market(3000, predictive=True, seed=3)
    model = TradeRiskModel(TradeRiskConfig(horizons={"4h": 240}, max_rounds=50))
    model.fit(features, close, high, low)
    for side in ("long", "short"):
        assert model.sides["4h"][side].status == "insufficient_data"


def test_leverage_keeps_liquidation_beyond_the_worst_move():
    cfg = TradeRiskConfig()  # 0.5% maintenance margin, 0.10% round-trip cost
    assert max_safe_leverage(0.01, cfg) == 20  # 1/20 - 0.6% = 4.4% clears a 1% move
    assert max_safe_leverage(0.05, cfg) == 10  # 1/10 - 0.6% = 9.4% clears 5%, 1/20 does not
    assert max_safe_leverage(0.20, cfg) == 3
    assert max_safe_leverage(0.995, cfg) == 1  # nothing clears it: the floor of the ladder
    assert [risk_level(s) for s in (10, 50, 75, 95)] == ["LOW", "MODERATE", "HIGH", "EXTREME"]


def test_save_load_roundtrip(tmp_path, fast_config):
    features, close, high, low = _volatile_market(4000, predictive=True, seed=5)
    model = TradeRiskModel(fast_config)
    model.fit(features, close, high, low)
    path = tmp_path / "risk.joblib"
    model.save(path)
    loaded = TradeRiskModel.load(path)
    assert loaded is not None
    a = model.predict_latest(features, close)["15m"]
    b = loaded.predict_latest(features, close)["15m"]
    assert a["long"]["worst_move_pct"] == pytest.approx(b["long"]["worst_move_pct"])
    assert a["short"]["score"] == b["short"]["score"]


def _regime_switching_market(n, seed):
    """Volatility jumps between calm and stormy stretches; features carry no hint of it."""
    rng = np.random.default_rng(seed)
    stormy = np.zeros(n, dtype=bool)
    for t in range(1, n):
        stormy[t] = stormy[t - 1] if rng.random() < 0.995 else not stormy[t - 1]
    vol = np.where(stormy, 0.004, 0.001)
    close = 100 * np.exp(np.cumsum(rng.normal(0, vol)))
    wiggle = np.abs(rng.normal(0, 0.3 * vol))
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    features = pd.DataFrame({"noise": rng.normal(0, 1, n), "noise2": rng.normal(0, 1, n)}, index=idx)
    return (features, pd.Series(close, idx), pd.Series(close * (1 + wiggle), idx),
            pd.Series(close * (1 - wiggle), idx))


def test_worst_move_holds_95_in_stormy_stretches_too():
    features, close, high, low = _regime_switching_market(12_000, seed=2)
    model = TradeRiskModel(TradeRiskConfig(horizons={"15m": 15}, max_rounds=100))
    model.fit(features, close, high, low)

    m = model.sides["15m"]["long"].metrics
    stormy, calm = m["coverage_by_regime"]["stormy"], m["coverage_by_regime"]["calm"]
    assert stormy["n_independent"] >= 30 and calm["n_independent"] >= 30, m
    # Trailing volatility lags when a storm starts: one overall multiplier lets too many moves through
    assert stormy["single_scale"] < 0.91, m
    # A multiplier per regime brings the stormy stretches back to ~95%, and gives back the calm surplus
    assert stormy["model"] >= stormy["single_scale"] + 0.03 and stormy["model"] >= 0.92, m
    assert calm["single_scale"] > calm["model"], m
    assert m["scale"][2] > m["scale"][0]  # more margin when stormy than when calm

    out = model.predict_latest(features, close)["15m"]
    assert out["regime"] in ("calm", "normal", "stormy") and 0.0 <= out["storm_chance"] <= 1.0
    # Serving uses the current regime's multiplier
    side = model.sides["15m"]["long"]
    X = features.reindex(columns=model.feature_names).iloc[[-1]].assign(risk_log_horizon_vol=np.log(0.01))
    calm_bound, storm_bound = (side.predict(X, 0.01, k)["model_worst_move"] for k in (0, 2))
    assert storm_bound / calm_bound == pytest.approx(side.scale[2] / side.scale[0])
