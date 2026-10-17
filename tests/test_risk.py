"""Risk inputs, targets, the walk-forward check and the score."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from deltarisk.config import load_config
from deltarisk.publisher.multi_manager import SymbolWorker
from deltarisk.stats.regime import WINDOW_BY_NAME, Window
from deltarisk.stats.risk import (
    HAR_COLUMNS,
    RiskSpec,
    WindowRiskModel,
    adverse_moves,
    coin_tier,
    exceed_chance,
    latest_reading,
    liquidation_move,
    risk_inputs,
    risk_score,
    split_positions,
    stop_move,
    walk_forward,
    window_data,
)

SYM, BENCH = "B-TEST_USDT", "B-BTC_USDT"


def _candles(n: int, minutes: int = 5, seed: int = 0, sigma: float = 0.002, sym: str = SYM) -> pd.DataFrame:
    """Random-walk candles whose highs and lows come from the path inside each candle."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, sigma / np.sqrt(5), (n, 5))
    path = 100 * np.exp(np.cumsum(steps.ravel())).reshape(n, 5)
    open_ = np.r_[100.0, path[:-1, -1]]
    idx = pd.date_range("2026-01-01", periods=n, freq=f"{minutes}min", tz="UTC")
    volume = rng.uniform(1e5, 2e5, n)
    return pd.DataFrame({
        f"{sym}_open": open_, f"{sym}_high": np.maximum(path.max(axis=1), open_),
        f"{sym}_low": np.minimum(path.min(axis=1), open_), f"{sym}_close": path[:, -1],
        f"{sym}_quote_volume": volume, f"{sym}_taker_buy_quote_volume": volume * rng.uniform(0.4, 0.6, n),
    }, index=idx)


def test_adverse_move_uses_the_next_h_candles_highs_and_lows():
    idx = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
    frame = pd.DataFrame({f"{SYM}_close": [100.0, 101, 99, 100, 100], f"{SYM}_high": [100.0, 103, 100, 101, 150],
                          f"{SYM}_low": [99.0, 98, 95, 99, 50]}, index=idx)
    out = adverse_moves(frame, SYM, 2)
    assert out["long"].iloc[0] == pytest.approx(0.05)  # lowest low of candles 1-2 is 95
    assert out["short"].iloc[0] == pytest.approx(0.03)  # highest high of candles 1-2 is 103
    assert out["long"].iloc[1] == pytest.approx(6 / 101)  # candle 4's low of 50 is outside the window
    assert out["short"].iloc[1] == pytest.approx(0.0)
    assert out["long"].iloc[-2:].isna().all() and out["short"].iloc[-2:].isna().all()


def test_inputs_never_use_later_candles():
    frame = pd.concat([_candles(3000), _candles(3000, sym=BENCH, seed=1)], axis=1)
    hourly = _candles(400, minutes=60, seed=2)
    hourly[f"{SYM}_funding_rate"] = np.random.default_rng(3).normal(1e-4, 5e-5, len(hourly))
    window = WINDOW_BY_NAME["1h"]
    before = risk_inputs(frame, SYM, window, BENCH, hourly)
    cut = 2500
    changed = frame.copy()
    changed.iloc[cut + 1:] *= 1.7
    hourly_changed = hourly.copy()
    later = hourly.index + pd.Timedelta(hours=1) > frame.index[cut] + pd.Timedelta(minutes=5)
    hourly_changed.loc[later, f"{SYM}_funding_rate"] = 0.01
    after = risk_inputs(changed, SYM, window, BENCH, hourly_changed)
    assert {"funding", "funding_pct", "market_rv", "taker_1", "to_funding"} <= set(before.columns)
    pd.testing.assert_frame_equal(before.iloc[: cut + 1], after.iloc[: cut + 1])
    assert not before.iloc[cut + 1:].equals(after.iloc[cut + 1:])
    assert before[list(HAR_COLUMNS)].iloc[-1].notna().all() and before["funding"].iloc[-1] > -1


def test_split_leaves_a_gap_and_tests_after_the_training_share():
    spec = RiskSpec(folds=4)
    positions = np.arange(10_000)
    folds = split_positions(positions, spec, gap=100)
    assert len(folds) == 4 and folds[0][1][0] == 6000
    for train, test in folds:
        assert train.max() < test[0] - 100 and train[0] == 0
    assert spec.gap(WINDOW_BY_NAME["1h"]) == 12 + 96 and spec.gap(WINDOW_BY_NAME["8h"]) == 8 + 8


def test_bad_case_covers_one_in_ten_on_a_random_walk():
    frame = _candles(9000)
    window = WINDOW_BY_NAME["30m"]
    data = window_data({"5m": frame}, SYM, window)
    spec = dataclasses.replace(RiskSpec(), folds=2, rounds=20)
    out = walk_forward(data, stop_move(5, 10), liquidation_move(10, 0.005), spec)
    for side in ("long", "short"):
        assert 0.07 <= out[side]["har"]["coverage"] <= 0.13
        assert 0.06 <= out[side]["lgbm"]["coverage"] <= 0.14
        assert out[side]["model"] in ("har", "lgbm")
        assert abs(out[side]["har"]["stop_hit_predicted"] - out[side]["stop_hit_rate"]) < 0.05
    model = WindowRiskModel(spec).fit(data, data.trainable)
    today = latest_reading(model, data, {"long": "lgbm", "short": "har"}, stop_move(5, 10), 0.095, 10)
    assert today["at"] == frame.index[-1]
    assert today["long"]["model"] == "lgbm" and 0 <= today["short"]["stop_hit_chance"] <= 1
    assert today["long"]["bad_case_margin_pct"] == pytest.approx(10 * today["long"]["bad_case_pct"])


def test_exceed_chance_reads_the_quantiles():
    alphas = (0.5, 0.9, 0.97)
    q = np.array([[1.0, 2.0, 3.0]])
    assert exceed_chance(q, alphas, np.array([2.0]))[0] == pytest.approx(0.1)
    assert exceed_chance(q, alphas, np.array([0.5]))[0] == pytest.approx(0.75)
    tail = exceed_chance(np.repeat(q, 3, axis=0), alphas, np.array([3.0, 4.0, 10.0]))
    assert tail[0] == pytest.approx(0.03) and tail[0] > tail[1] > tail[2] > 0


def test_stop_is_on_margin_and_scores_against_it():
    assert stop_move(5, 20) == pytest.approx(0.0025)
    assert stop_move(-10, 3) == pytest.approx(0.1 / 3)
    assert liquidation_move(20, 0.004) == pytest.approx(0.046)
    assert risk_score(0.426) == 43  # the score is the stop-hit chance in percent
    assert risk_score(1.2) == 100 and risk_score(-0.1) == 0
    assert risk_score(float("nan")) is None


def test_tiers():
    assert coin_tier("B-BTC_USDT") == "megacap" and coin_tier("B-XRP_USDT", 1e3) == "megacap"
    assert coin_tier("B-1000PEPE_USDT", 5e9) == "shitcoin"
    assert coin_tier("B-ZK_USDT", 20e6) == "shitcoin"
    assert coin_tier("B-INJ_USDT", 200e6) == "altcoin" and coin_tier("B-INJ_USDT") == "altcoin"


def test_offline_coins_have_every_chart_the_windows_use():
    worker = SymbolWorker(SYM, BENCH, "OTHER", load_config())
    worker.load_synthetic()
    charts = {w.interval for w in WINDOW_BY_NAME.values()}
    assert charts <= set(worker.candles) and all(len(worker.candles[iv]) for iv in charts)
    assert isinstance(WINDOW_BY_NAME["8h"], Window)


def test_models_retrain_after_a_day():
    worker = SymbolWorker(SYM, BENCH, "OTHER", load_config())
    assert not worker.needs_training()  # nothing loaded yet
    worker.load_synthetic()
    assert not worker.needs_training()
    later = worker.risk.trained_at + pd.Timedelta(hours=25)
    assert worker.needs_training(now=later)


def test_saved_models_are_reused_only_when_they_match(tmp_path):
    from deltarisk.stats.risk import CoinRisk, load_models, save_models

    worker = SymbolWorker(SYM, BENCH, "OTHER", load_config())
    worker.load_synthetic()
    path = tmp_path / "risk" / f"{SYM}.pkl"
    save_models(path, worker.risk)
    day = pd.Timedelta(hours=24)

    loaded = load_models(path, SYM, BENCH, day)
    assert isinstance(loaded, CoinRisk) and loaded.trained_at == worker.risk.trained_at
    assert set(loaded.windows) == set(worker.risk.windows)
    loaded.score(worker.candles)
    assert loaded.view(20, 0.004, 5)["1h"]["long"]["score"] == worker.risk.view(20, 0.004, 5)["1h"]["long"]["score"]

    assert load_models(path, SYM, "B-ETH_USDT", day) is None  # another benchmark
    assert load_models(path, "B-OTHER_USDT", BENCH, day) is None
    assert load_models(path, SYM, BENCH, pd.Timedelta(0)) is None  # too old
    assert load_models(tmp_path / "missing.pkl", SYM, BENCH, day) is None
    path.write_bytes(b"not a pickle")
    assert load_models(path, SYM, BENCH, day) is None


def test_a_quick_restart_reuses_the_saved_models(monkeypatch):
    from deltarisk.stats import risk as risk_module

    worker = SymbolWorker(SYM, BENCH, "OTHER", load_config())
    worker.load_synthetic()
    risk_module.save_models(worker._models_path(), worker.risk)  # offline runs don't save on their own

    restarted = SymbolWorker(SYM, BENCH, "OTHER", load_config())
    restarted.candles = worker.candles
    trained = []
    monkeypatch.setattr(SymbolWorker, "train", lambda self, **kw: trained.append(kw))
    restarted.prepare_models()
    assert trained == [] and restarted.status == "ready" and restarted.progress is None
    assert restarted.risk.trained_at == worker.risk.trained_at and restarted.risk_view["1h"]["long"]["score"] is not None


def test_exceed_chance_matches_the_row_by_row_reading():
    import math

    rng = np.random.default_rng(4)
    alphas = (0.25, 0.5, 0.75, 0.9, 0.97)
    q = np.sort(rng.uniform(0.01, 3.0, (500, 5)), axis=1)
    x = rng.uniform(-0.5, 5.0, 500)

    def one(qi, xi):
        if xi >= qi[-1]:
            scale = (qi[-1] - qi[-2]) / math.log((1 - alphas[-2]) / (1 - alphas[-1]))
            return (1 - alphas[-1]) * math.exp(-(xi - qi[-1]) / scale)
        return 1.0 - float(np.interp(xi, np.r_[0.0, qi], np.r_[0.0, alphas]))

    np.testing.assert_allclose(exceed_chance(q, alphas, x), [one(qi, xi) for qi, xi in zip(q, x)], atol=1e-12)
    assert exceed_chance(q[:1], alphas, 0.0)[0] == 1.0
