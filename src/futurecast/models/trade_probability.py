"""Calibrated long / short win probabilities for each forecast horizon.

A "win" is the take-profit being touched before the stop-loss (see labels.barriers).

Take-profit and stop are chosen per candle from a small menu of distances (LEVELS, in
units of horizon volatility: 1:1 tight and wide, 2:1 and 1:2 reward to risk). Every
level gets its own models below, and for the latest candle the level with the highest
expected profit after fees is served, so the odds shown always belong to the levels shown.
Levels with a proven edge are preferred; without one, the pick is by recent base rates.

For every horizon, side and level, LightGBM, XGBoost and CatBoost classifiers are fitted and
their average is mapped onto observed win frequencies with Platt scaling. History is
used strictly in time order, each slice separated from the next by one horizon so
overlapping trade windows cannot leak outcomes across a boundary:

    | fit | early-stop |  calibrate  |    test    |
    oldest ------------------------------------> newest

Validation: classifiers are fitted on the oldest data, calibrated on the next slice
and scored on the newest slice they never saw. The side earns "edge" only when the
lower end of a block-bootstrap interval of its Brier skill (vs. quoting the base rate)
is above zero on enough independent trades; otherwise the recent base rate is served.

Trading check: the test slice is also replayed as a strategy — on every candle pick the
proven level with the highest expected profit, take the trade whenever its calibrated odds
clear break-even, one position at a time — and a long/short signal is only ever raised for
sides whose replay made money after costs. Trying several levels gives luck more chances
to pass the edge test, so this replay of the pick-the-best rule is the gate that matters.

Out-of-sample shrink: calibrated odds tend to be overconfident on newer data, most of
all at the extremes where trades get taken. One slope in [0, 1] per side, learned on
the test slice, pulls probabilities toward the base rate in logit space. Test metrics
use cross-fitted slopes (learned on one half, applied to the other) so they stay honest;
the served probability uses the slope learned on the whole slice. It never sharpens.

Serving: once the procedure is validated it is rolled forward — classifiers are refitted
on everything before the newest slice and recalibrated on that newest slice — so live
probabilities reflect the current regime with the same fit → calibrate → apply spacing
that was tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from futurecast.labels.barriers import (
    BarrierSpec,
    barrier_distances,
    break_even_probability,
    first_touch_labels,
    horizon_volatility,
)

logger = logging.getLogger(__name__)

# Candle interval -> {horizon label: horizon length in candles}
HORIZONS_BY_INTERVAL: Dict[str, Dict[str, int]] = {
    "1m": {"5m": 5, "15m": 15, "1h": 60},
    "15m": {"4h": 16, "8h": 32},
}
# (take-profit, stop) distances tried per horizon and side, in units of horizon volatility
LEVELS: Tuple[Tuple[float, float], ...] = ((0.5, 0.5), (1.0, 0.5), (0.5, 1.0), (1.0, 1.0))
SIDES = ("long", "short")
MODEL_NAMES = ("LightGBM", "XGBoost", "CatBoost")
FORMAT_VERSION = 6

_EPS = 1e-6


@dataclass
class TradeModelConfig:
    """All lengths (horizons, barrier volatility window) are counted in candles of `interval`."""

    interval: str = "1m"
    barrier: BarrierSpec = field(default_factory=BarrierSpec)  # volatility window, cost and take-profit floor
    horizons: Dict[str, int] = field(default_factory=lambda: dict(HORIZONS_BY_INTERVAL["1m"]))
    levels: Tuple[Tuple[float, float], ...] = LEVELS
    train_frac: float = 0.6
    calib_frac: float = 0.2  # the test slice gets the remainder
    early_stop_frac: float = 0.2  # tail of a training slice used only for early stopping
    min_independent_test: int = 30  # non-overlapping test trades needed before judging skill
    min_class_count: int = 20  # every slice needs this many wins and losses
    min_signal_trades: int = 10  # replayed trades needed before a long/short signal is trusted
    bootstrap_rounds: int = 500
    skill_confidence: float = 0.9  # one-sided confidence that skill > 0
    max_rounds: int = 400
    random_seed: int = 42


# ───────────────────────── classifiers ─────────────────────────


def _fit_lightgbm(X_fit, y_fit, X_es, y_es, cfg: TradeModelConfig):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "seed": cfg.random_seed,
        "verbose": -1,
    }
    return lgb.train(
        params,
        lgb.Dataset(X_fit, label=y_fit),
        num_boost_round=cfg.max_rounds,
        valid_sets=[lgb.Dataset(X_es, label=y_es)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )


def _fit_xgboost(X_fit, y_fit, X_es, y_es, cfg: TradeModelConfig):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "tree_method": "hist",
        "seed": cfg.random_seed,
        "verbosity": 0,
    }
    return xgb.train(
        params,
        xgb.DMatrix(X_fit, label=y_fit),
        num_boost_round=cfg.max_rounds,
        evals=[(xgb.DMatrix(X_es, label=y_es), "es")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )


def _fit_catboost(X_fit, y_fit, X_es, y_es, cfg: TradeModelConfig):
    model = CatBoostClassifier(
        iterations=cfg.max_rounds,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        random_seed=cfg.random_seed,
        early_stopping_rounds=40,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(X_fit, y_fit, eval_set=(X_es, y_es), verbose=False)
    return model


_FITTERS = {"LightGBM": _fit_lightgbm, "XGBoost": _fit_xgboost, "CatBoost": _fit_catboost}


def _predict_raw(name: str, model: Any, X: pd.DataFrame) -> np.ndarray:
    if name == "LightGBM":
        return model.predict(X, num_iteration=model.best_iteration or None)
    if name == "XGBoost":
        best = getattr(model, "best_iteration", None)
        it_range = (0, best + 1) if best is not None else (0, 0)
        return model.predict(xgb.DMatrix(X), iteration_range=it_range)
    return model.predict_proba(X)[:, 1]


def _fit_all(X_fit, y_fit, X_es, y_es, horizon: int, cfg: TradeModelConfig) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for name in MODEL_NAMES:
        try:
            models[name] = _FITTERS[name](X_fit, y_fit, X_es, y_es, cfg)
        except Exception as err:  # one library failing should not sink the others
            logger.warning(f"{name} failed for horizon {horizon}: {err}")
    return models


# ───────────────────────── calibration ─────────────────────────


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


class PlattCalibrator:
    """Maps a raw score onto observed frequencies: p = sigmoid(a * logit(raw) + b)."""

    def __init__(self) -> None:
        self.lr: Optional[LogisticRegression] = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.lr = LogisticRegression(C=1.0)
        self.lr.fit(_logit(raw).reshape(-1, 1), y.astype(int))
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(_logit(np.asarray(raw, dtype=float)).reshape(-1, 1))[:, 1]


def _calibrate(models: Dict[str, Any], X_cal, y_cal):
    raw = {name: _predict_raw(name, m, X_cal) for name, m in models.items()}
    per_model = {name: PlattCalibrator().fit(r, y_cal) for name, r in raw.items()}
    ensemble = PlattCalibrator().fit(np.mean(np.vstack(list(raw.values())), axis=0), y_cal)
    return per_model, ensemble


def shrink_toward(p, base: float, slope: float):
    """sigmoid(logit(base) + slope * (logit(p) - logit(base))); slope 1 keeps p, 0 returns base."""
    centre = _logit(np.array([base]))[0]
    z = centre + slope * (_logit(np.asarray(p, dtype=float)) - centre)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_shrink(p: np.ndarray, y: np.ndarray, base: float) -> float:
    """Slope in [0, 1] that best maps probabilities onto realised outcomes (log loss)."""
    def loss(slope: float) -> float:
        q = np.clip(shrink_toward(p, base, slope), _EPS, 1.0 - _EPS)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    return float(minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded").x)


def _cross_fit_shrink(p: np.ndarray, y: np.ndarray, base: float) -> np.ndarray:
    """Shrunk probabilities where each half's slope is learned on the other half."""
    half = len(p) // 2
    first, second = slice(0, half), slice(half, len(p))
    out = np.empty_like(p)
    out[second] = shrink_toward(p[second], base, _fit_shrink(p[first], y[first], base))
    out[first] = shrink_toward(p[first], base, _fit_shrink(p[second], y[second], base))
    return out


def _ensemble_probability(models, ensemble_cal: PlattCalibrator, X) -> np.ndarray:
    raw = [_predict_raw(name, m, X) for name, m in models.items()]
    return ensemble_cal.transform(np.mean(np.vstack(raw), axis=0))


# ───────────────────────── evaluation ─────────────────────────


def _reliability(p: np.ndarray, y: np.ndarray, bins: int = 5) -> List[Dict[str, float]]:
    """Predicted vs observed win rate in equal-count bins of the predicted probability."""
    out = []
    for chunk in np.array_split(np.argsort(p), bins):
        if len(chunk):
            out.append({
                "predicted": round(float(p[chunk].mean()), 4),
                "observed": round(float(y[chunk].mean()), 4),
                "count": int(len(chunk)),
            })
    return out


def _bootstrap_skill_lower(p, y, base_rate, block: int, cfg: TradeModelConfig) -> float:
    """Lower confidence bound on Brier skill, resampling contiguous blocks of one horizon.

    Overlapping trade windows make neighbouring rows dependent; resampling whole
    horizon-length blocks keeps that dependence instead of overstating certainty.
    """
    n_blocks = len(y) // block
    if n_blocks < 2:
        return float("-inf")
    trim = n_blocks * block
    err_model = ((p[:trim] - y[:trim]) ** 2).reshape(n_blocks, block).sum(axis=1)
    err_base = ((base_rate - y[:trim]) ** 2).reshape(n_blocks, block).sum(axis=1)
    rng = np.random.default_rng(cfg.random_seed)
    picks = rng.integers(0, n_blocks, size=(cfg.bootstrap_rounds, n_blocks))
    skill = 1.0 - err_model[picks].sum(axis=1) / np.maximum(err_base[picks].sum(axis=1), 1e-12)
    return float(np.quantile(skill, 1.0 - cfg.skill_confidence))


def _replay_trades(p, break_even, pnl, y, horizon: int, cost: float) -> Dict[str, Any]:
    """Take the trade whenever p clears break-even, holding one position at a time."""
    trades = []
    t = 0
    while t < len(p):
        if p[t] >= break_even[t]:
            trades.append(t)
            t += horizon
        else:
            t += 1
    if not trades:
        return {"trades": 0, "win_rate": None, "avg_net_return_pct": None, "total_net_return_pct": None}
    idx = np.array(trades)
    net = pnl[idx] - cost
    return {
        "trades": int(len(idx)),
        "win_rate": round(float(y[idx].mean()), 4),
        "avg_probability": round(float(p[idx].mean()), 4),
        "avg_net_return_pct": round(float(net.mean() * 100), 4),
        "total_net_return_pct": round(float(net.sum() * 100), 4),
    }


# ───────────────────────── per horizon / side ─────────────────────────


@dataclass
class SideModel:
    """Production classifiers + calibrators for one horizon, side and level, with test evidence."""

    status: str  # "edge" | "no_edge" | "insufficient_data"
    base_rate: float  # recent win frequency, served when there is no proven edge
    level: Tuple[float, float] = LEVELS[0]  # (take-profit, stop) in units of horizon volatility
    loss_ratio: float = 1.0  # average non-win loss as a fraction of the stop distance
    shrink: float = 1.0  # out-of-sample overconfidence correction (1 = none)
    models: Dict[str, Any] = field(default_factory=dict)
    calibrators: Dict[str, PlattCalibrator] = field(default_factory=dict)
    ensemble_calibrator: Optional[PlattCalibrator] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Served probability plus model and per-model calibrated probabilities for the last row."""
        if not self.models:
            return {"probability": self.base_rate, "model_probability": None, "per_model": {}}
        raw = {name: _predict_raw(name, m, X) for name, m in self.models.items()}
        per_model = {
            name: float(shrink_toward(self.calibrators[name].transform(r)[-1:], self.base_rate, self.shrink)[0])
            for name, r in raw.items()
        }
        ens = self.ensemble_calibrator.transform(np.mean(np.vstack(list(raw.values())), axis=0))[-1:]
        model_p = float(shrink_toward(ens, self.base_rate, self.shrink)[0])
        served = model_p if self.status == "edge" else self.base_rate
        return {"probability": served, "model_probability": model_p, "per_model": per_model}


def _purged_slices(n: int, horizon: int, cfg: TradeModelConfig) -> Dict[str, slice]:
    train_end = int(n * cfg.train_frac)
    es_start = int(train_end * (1.0 - cfg.early_stop_frac))
    calib_end = int(n * (cfg.train_frac + cfg.calib_frac))
    return {
        "fit": slice(0, max(es_start - horizon, 0)),
        "es": slice(es_start, train_end),
        "calib": slice(train_end + horizon, calib_end),
        "test": slice(calib_end + horizon, n),
    }


def _rolled_slices(n: int, horizon: int, cfg: TradeModelConfig) -> Dict[str, slice]:
    """Same spacing as validation, shifted forward: fit on everything before the newest slice."""
    calib_end = int(n * (cfg.train_frac + cfg.calib_frac))
    es_start = int(calib_end * (1.0 - cfg.early_stop_frac))
    return {
        "fit": slice(0, max(es_start - horizon, 0)),
        "es": slice(es_start, calib_end),
        "calib": slice(calib_end + horizon, n),
    }


def expected_return(p, tp, sl, cost: float, loss_ratio: float = 1.0):
    """Expected net return per trade: a win earns tp - cost, a non-win loses loss_ratio * sl + cost."""
    loss = loss_ratio * np.asarray(sl, dtype=float)
    return np.asarray(p, dtype=float) * (np.asarray(tp, dtype=float) + loss) - (loss + cost)


def _loss_ratio(y: np.ndarray, pnl: np.ndarray, sl: np.ndarray) -> float:
    lost = y == 0
    if not lost.any():
        return 1.0
    return float(np.clip(np.mean(-pnl[lost] / sl[lost]), 0.0, 1.0))


def _fit_side(
    X: pd.DataFrame,
    y: np.ndarray,
    pnl: np.ndarray,
    tp: np.ndarray,
    sl: np.ndarray,
    horizon: int,
    cfg: TradeModelConfig,
    level: Tuple[float, float] = LEVELS[0],
) -> Tuple[SideModel, Optional[Dict[str, np.ndarray]]]:
    """Models for one level, plus its test-slice odds and outcomes (None when untestable)."""
    cost = cfg.barrier.cost
    s = _purged_slices(len(y), horizon, cfg)
    recent_base = float(np.mean(y[s["calib"].start:]))

    def enough(labels: np.ndarray) -> bool:
        wins = int(labels.sum())
        return wins >= cfg.min_class_count and len(labels) - wins >= cfg.min_class_count

    if not all(enough(y[s[k]]) for k in ("fit", "es", "calib", "test")):
        return SideModel(status="insufficient_data", base_rate=recent_base, level=level,
                         metrics={"n_rows": int(len(y)), "reason": "too few wins or losses in a slice"}), None

    # 1. Validate: fit on the oldest data, calibrate on the next slice, score on unseen newest data
    models = _fit_all(X.iloc[s["fit"]], y[s["fit"]], X.iloc[s["es"]], y[s["es"]], horizon, cfg)
    if not models:
        return SideModel(status="insufficient_data", base_rate=recent_base, level=level,
                         metrics={"reason": "no model trained"}), None
    per_model_cal, ens_cal = _calibrate(models, X.iloc[s["calib"]], y[s["calib"]])

    seen = slice(0, s["calib"].stop)
    loss_ratio = _loss_ratio(y[seen], pnl[seen], sl[seen])
    base_before = float(np.mean(y[seen]))

    test = s["test"]
    y_t, p_raw = y[test], _ensemble_probability(models, ens_cal, X.iloc[test])
    shrink = _fit_shrink(p_raw, y_t, base_before)
    p_t = _cross_fit_shrink(p_raw, y_t, base_before)
    be_t = break_even_probability(tp[test], sl[test], cost, loss_ratio)

    brier = float(np.mean((p_t - y_t) ** 2))
    brier_base = float(np.mean((base_before - y_t) ** 2))
    try:
        auc = float(roc_auc_score(y_t, p_t))
    except ValueError:
        auc = float("nan")
    replay = _replay_trades(p_t, be_t, pnl[test], y_t, horizon, cost)

    metrics: Dict[str, Any] = {
        "n_rows": int(len(y)),
        "n_test": int(len(y_t)),
        "n_independent": int(len(y_t) // horizon),
        "test_win_rate": round(float(y_t.mean()), 4),
        "base_rate": round(base_before, 4),
        "brier": round(brier, 5),
        "brier_base_rate": round(brier_base, 5),
        "brier_skill": round(1.0 - brier / brier_base, 4) if brier_base > 0 else 0.0,
        "brier_skill_lower": round(_bootstrap_skill_lower(p_t, y_t, base_before, horizon, cfg), 4),
        "auc": round(auc, 4),
        "loss_ratio": round(loss_ratio, 4),
        "shrink": round(shrink, 4),
        "brier_skill_unshrunk": round(1.0 - float(np.mean((p_raw - y_t) ** 2)) / brier_base, 4) if brier_base > 0 else 0.0,
        "replay": replay,
        "reliability": _reliability(p_t, y_t),
    }

    if metrics["n_independent"] < cfg.min_independent_test:
        status = "insufficient_data"
        metrics["reason"] = f"only {metrics['n_independent']} independent test trades (need {cfg.min_independent_test})"
    elif metrics["brier_skill_lower"] > 0 and auc > 0.5:
        status = "edge"
    else:
        status = "no_edge"
    test_arrays = {
        "p": p_t, "break_even": be_t, "pnl": pnl[test], "y": y_t,
        "expected": expected_return(p_t, tp[test], sl[test], cost, loss_ratio),
    }

    # 2. Roll forward: refit on everything but the newest slice, recalibrate on the newest slice
    r = _rolled_slices(len(y), horizon, cfg)
    if enough(y[r["fit"]]) and enough(y[r["es"]]) and enough(y[r["calib"]]):
        rolled = _fit_all(X.iloc[r["fit"]], y[r["fit"]], X.iloc[r["es"]], y[r["es"]], horizon, cfg)
        if rolled:
            models = rolled
            per_model_cal, ens_cal = _calibrate(models, X.iloc[r["calib"]], y[r["calib"]])
            loss_ratio = _loss_ratio(y, pnl, sl)

    return SideModel(
        status=status,
        base_rate=recent_base,
        level=level,
        loss_ratio=loss_ratio,
        shrink=shrink,
        models=models,
        calibrators=per_model_cal,
        ensemble_calibrator=ens_cal,
        metrics=metrics,
    ), test_arrays


@dataclass
class SideLevels:
    """Every take-profit / stop level for one horizon and side, and how picking among them did."""

    levels: List[SideModel]
    signal_validated: bool = False  # picking the best proven level made money on unseen data after costs
    replay: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Best status across levels: "edge" if any level has one."""
        for status in ("edge", "no_edge"):
            if any(lvl.status == status for lvl in self.levels):
                return status
        return "insufficient_data"


def _fit_levels(
    X: pd.DataFrame,
    labels: List[Dict[str, np.ndarray]],
    horizon: int,
    cfg: TradeModelConfig,
) -> SideLevels:
    """Fit every level for one side, then replay "trade the best proven level" on the test slice."""
    fitted = [
        _fit_side(X, lab["y"], lab["pnl"], lab["tp"], lab["sl"], horizon, cfg, level)
        for level, lab in zip(cfg.levels, labels)
    ]
    side = SideLevels(levels=[model for model, _ in fitted])
    proven = [t for model, t in fitted if model.status == "edge" and t is not None]
    if not proven:
        return side

    # All levels share the same test rows; on each one take the level with the highest expected return
    stack = {key: np.vstack([t[key] for t in proven]) for key in proven[0]}
    best = np.argmax(stack["expected"], axis=0)
    rows = np.arange(best.size)
    pick = {key: values[best, rows] for key, values in stack.items()}
    side.replay = _replay_trades(pick["p"], pick["break_even"], pick["pnl"], pick["y"], horizon, cfg.barrier.cost)
    side.signal_validated = (
        side.replay["trades"] >= cfg.min_signal_trades
        and side.replay["avg_net_return_pct"] is not None
        and side.replay["avg_net_return_pct"] > 0
    )
    return side


# ───────────────────────── public model ─────────────────────────


class TradeProbabilityModel:
    """Per-horizon calibrated long/short win probabilities for one symbol."""

    def __init__(self, config: Optional[TradeModelConfig] = None) -> None:
        self.config = config or TradeModelConfig()
        self.feature_names: List[str] = []
        self.sides: Dict[str, Dict[str, SideModel]] = {}
        self.trained_at: Optional[str] = None
        self.history_start: Optional[str] = None
        self.history_end: Optional[str] = None
        self.version = FORMAT_VERSION

    @property
    def is_trained(self) -> bool:
        return bool(self.sides)

    def _spec(self, level: Tuple[float, float]) -> BarrierSpec:
        return replace(self.config.barrier, tp_vol_mult=level[0], sl_vol_mult=level[1])

    def fit(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        on_step: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        """Fit every horizon/side/level. Features may contain NaN (handled natively by the boosters).

        `on_step(horizon, side)` is called before each horizon/side is fitted.
        """
        self.feature_names = list(features.columns)
        self.sides = {}
        summary: Dict[str, Any] = {}

        for h_key, horizon in self.config.horizons.items():
            sigma = horizon_volatility(close, horizon, self.config.barrier)
            per_level = []
            for level in self.config.levels:
                tp, sl = barrier_distances(sigma, self._spec(level))
                per_level.append((tp, sl, first_touch_labels(close, high, low, tp, sl, horizon)))

            known = per_level[0][2]["long_win"].notna().to_numpy()  # same rows for every level
            X = features.loc[known]
            self.sides[h_key] = {}
            for side in SIDES:
                if on_step is not None:
                    on_step(h_key, side)
                labels = [
                    {"y": out[f"{side}_win"].to_numpy()[known], "pnl": out[f"{side}_pnl"].to_numpy()[known],
                     "tp": tp[known], "sl": sl[known]}
                    for tp, sl, out in per_level
                ]
                side_levels = _fit_levels(X, labels, horizon, self.config)
                self.sides[h_key][side] = side_levels
                summary[f"{h_key}_{side}"] = {
                    "status": side_levels.status,
                    "signal_validated": side_levels.signal_validated,
                    "replay": side_levels.replay,
                    "levels": [
                        {"level": list(lvl.level), "status": lvl.status,
                         "brier_skill_lower": lvl.metrics.get("brier_skill_lower"), "auc": lvl.metrics.get("auc")}
                        for lvl in side_levels.levels
                    ],
                }
                logger.info(f"Trade model {h_key} {side}: {summary[f'{h_key}_{side}']}")

        self.trained_at = pd.Timestamp.now(tz="UTC").isoformat()
        self.history_start = str(features.index.min())
        self.history_end = str(features.index.max())
        return summary

    def predict_latest(self, features: pd.DataFrame, close: pd.Series) -> Dict[str, Any]:
        """Win probabilities, picked take-profit / stop and a trade signal for the latest bar, per horizon."""
        cost = self.config.barrier.cost
        X = features.reindex(columns=self.feature_names).iloc[[-1]]
        entry = float(close.iloc[-1])
        out: Dict[str, Any] = {}

        for h_key, horizon in self.config.horizons.items():
            sigma = float(horizon_volatility(close, horizon, self.config.barrier).iloc[-1])
            if not np.isfinite(sigma) or entry <= 0:
                continue

            sides_out: Dict[str, Any] = {}
            expected: Dict[str, float] = {}
            for side in SIDES:
                side_levels = self.sides.get(h_key, {}).get(side)
                if side_levels is None:
                    continue
                options = []
                for lvl in side_levels.levels:
                    tp, sl = (float(v) for v in barrier_distances(sigma, self._spec(lvl.level)))
                    pred = lvl.predict(X)
                    options.append((lvl, tp, sl, pred, float(expected_return(pred["probability"], tp, sl, cost, lvl.loss_ratio))))
                proven = [o for o in options if o[0].status == "edge"]
                lvl, tp, sl, pred, ev = max(proven or options, key=lambda o: o[4])
                sides_out[side] = {
                    **pred,
                    "break_even": round(float(break_even_probability(tp, sl, cost, lvl.loss_ratio)), 4),
                    "tp_pct": round(tp * 100, 4),
                    "sl_pct": round(sl * 100, 4),
                    "level": list(lvl.level),
                    "level_pick": "model" if proven else "base_rate",
                    "expected_return_pct": round(ev * 100, 4),
                    "status": lvl.status,
                    "signal_validated": side_levels.signal_validated,
                    "base_rate": lvl.base_rate,
                    "replay": side_levels.replay,
                    # the level's own replay is left out: signals are gated on `replay`, the pick-the-best rule
                    "metrics": {k: v for k, v in lvl.metrics.items() if k not in ("reliability", "replay")},
                    "reliability": lvl.metrics.get("reliability", []),
                    "levels": [
                        {"level": list(o[0].level), "tp_pct": round(o[1] * 100, 4), "sl_pct": round(o[2] * 100, 4),
                         "probability": o[3]["probability"], "expected_return_pct": round(o[4] * 100, 4),
                         "status": o[0].status}
                        for o in options
                    ],
                }
                if lvl.status == "edge" and side_levels.signal_validated:
                    expected[side] = ev
            if not sides_out:
                continue

            best = max(expected, key=expected.get) if expected else None
            signal = best.upper() if best is not None and expected[best] >= 0 else "NEUTRAL"

            # Barriers are percentages: apply them to the exchange's live price for levels
            out[h_key] = {
                "candle_interval": self.config.interval,
                "horizon_bars": horizon,
                "as_of": str(close.index[-1]),
                "signal": signal,
                "cost_pct": round(cost * 100, 4),
                **sides_out,
            }
        return out

    # ───────── persistence ─────────

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @staticmethod
    def load(path: str | Path) -> Optional["TradeProbabilityModel"]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            model = joblib.load(p)
        except Exception as err:
            logger.warning(f"Could not load trade model {p}: {err}")
            return None
        if getattr(model, "version", None) != FORMAT_VERSION:
            logger.info(f"Ignoring trade model {p}: format {getattr(model, 'version', None)} != {FORMAT_VERSION}")
            return None
        return model


def trade_model_path(models_dir: str | Path, symbol_clean: str) -> Path:
    return Path(models_dir) / f"trade_prob_{symbol_clean}.joblib"
