"""Win chances for fair-value-gap setups, learned from the coin's own history.

The rules (features.setups) find every setup; these models only judge them. For each
take-profit multiple in SetupSpec.targets, LightGBM, XGBoost and CatBoost classifiers learn
whether a filled setup reaches its take-profit before its stop (labels.setups), from the
setup's geometry, its 15m context, the time of day and the coin's market features at the
setup candle. Their average is mapped onto observed win rates with Platt scaling.

Data: every setup whose order filled and whose trade has resolved, in time order. Rows whose
trade was still running when the next slice starts are dropped from the earlier slice, so no
outcome leaks across a boundary:

    | fit | early-stop |  calibrate  |    test    |
    oldest ------------------------------------> newest

Baseline: the rules alone, i.e. the win rate of filled setups of the same grade (A+ or B) on
the slices before the test slice. The models earn "edge" only when, on the test slice, the
lower end of a block-bootstrap interval of their Brier skill against that baseline is above
zero and they rank winners better than the grade does (AUC), on at least `min_test_setups`
setups. Otherwise the rules' win rate is served.

Replays, after each trade's own costs (labels.setups.trade_cost_r: maker fees on a win, taker
plus slippage on the stop's exit), of every A+ setup from the start of the test slice, one at a
time (a setup blocks the next until its order expires or its trade resolves):
- rules: place every A+ order.
- models: place A+ orders whose calibrated chance clears break-even.
A target is "validated" when the replay behind the probability it serves (models with an
edge, else rules) made money on at least `min_signal_trades` trades. Signals need that.
Break-even and expected results charge every non-win the stop's costs (timeouts skip the
slippage, so this errs on the safe side).

Serving: the classifiers are then refitted on everything before the newest slice and
recalibrated on the newest slice, so live chances reflect the current market with the same
fit -> calibrate -> apply spacing that was tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from futurecast.features.setups import SetupSpec
from futurecast.labels.setups import break_even, expected_r, target_key, trade_cost_r

logger = logging.getLogger(__name__)

MODEL_NAMES = ("LightGBM", "XGBoost", "CatBoost")
GRADES = ("A+", "B")
FORMAT_VERSION = 3  # 3: per-leg maker/taker costs
_EPS = 1e-6


@dataclass
class SetupModelConfig:
    spec: SetupSpec = field(default_factory=SetupSpec)
    train_frac: float = 0.6
    calib_frac: float = 0.2  # the test slice gets the remainder
    early_stop_frac: float = 0.2  # tail of the training slice used only for early stopping
    min_class_count: int = 15  # every slice needs this many wins and losses
    min_grade_rows: int = 20  # filled setups a grade needs for its own win rate (else the overall rate)
    min_test_setups: int = 30  # filled test setups needed before judging skill
    min_signal_trades: int = 10  # replayed trades needed before a signal is trusted
    bootstrap_block: int = 5  # consecutive setups resampled together (setups cluster in busy hours)
    bootstrap_rounds: int = 500
    skill_confidence: float = 0.9  # one-sided confidence that skill > 0
    max_rounds: int = 300
    random_seed: int = 42


# ───────────────────────── classifiers ─────────────────────────


def _fit_lightgbm(X_fit, y_fit, X_es, y_es, cfg: SetupModelConfig):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 30,
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


def _fit_xgboost(X_fit, y_fit, X_es, y_es, cfg: SetupModelConfig):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 5,
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


def _fit_catboost(X_fit, y_fit, X_es, y_es, cfg: SetupModelConfig):
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


def _fit_all(X_fit, y_fit, X_es, y_es, cfg: SetupModelConfig) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for name in MODEL_NAMES:
        try:
            models[name] = _FITTERS[name](X_fit, y_fit, X_es, y_es, cfg)
        except Exception as err:  # one library failing should not sink the others
            logger.warning(f"{name} setup model failed: {err}")
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


# ───────────────────────── evaluation ─────────────────────────


def _reliability(p: np.ndarray, y: np.ndarray, bins: int = 5) -> List[Dict[str, float]]:
    """Predicted vs observed win rate in equal-count bins of the predicted probability."""
    out = []
    for chunk in np.array_split(np.argsort(p), bins):
        if len(chunk):
            out.append({"predicted": round(float(p[chunk].mean()), 4),
                        "observed": round(float(y[chunk].mean()), 4), "count": int(len(chunk))})
    return out


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def _bootstrap_skill_lower(p, y, base, cfg: SetupModelConfig) -> float:
    """Lower confidence bound on Brier skill against `base`, resampling blocks of consecutive setups."""
    block = cfg.bootstrap_block
    n_blocks = len(y) // block
    if n_blocks < 2:
        return float("-inf")
    trim = n_blocks * block
    err_model = ((p[:trim] - y[:trim]) ** 2).reshape(n_blocks, block).sum(axis=1)
    err_base = ((base[:trim] - y[:trim]) ** 2).reshape(n_blocks, block).sum(axis=1)
    rng = np.random.default_rng(cfg.random_seed)
    picks = rng.integers(0, n_blocks, size=(cfg.bootstrap_rounds, n_blocks))
    skill = 1.0 - err_model[picks].sum(axis=1) / np.maximum(err_base[picks].sum(axis=1), 1e-12)
    return float(np.quantile(skill, 1.0 - cfg.skill_confidence))


def _replay(take: np.ndarray, idx: np.ndarray, busy_until: np.ndarray, filled: np.ndarray,
            win: np.ndarray, net_r: np.ndarray) -> Dict[str, Any]:
    """Place the orders marked `take` in time order, one setup at a time."""
    free_from, orders, trades = -1, 0, []
    for i in np.flatnonzero(take):
        if idx[i] <= free_from:
            continue
        orders += 1
        free_from = busy_until[i]
        if filled[i]:
            trades.append(i)
    if not trades:
        return {"orders": orders, "trades": 0, "wins": 0, "win_rate": None, "avg_net_r": None, "total_net_r": None}
    t = np.array(trades)
    return {"orders": orders, "trades": int(len(t)), "wins": int(win[t].sum()),
            "win_rate": round(float(win[t].mean()), 4), "avg_net_r": round(float(net_r[t].mean()), 4),
            "total_net_r": round(float(net_r[t].sum()), 3)}


def _slices(idx: np.ndarray, end: np.ndarray, cfg: SetupModelConfig, rolled: bool = False) -> Dict[str, np.ndarray]:
    """Row positions per slice, dropping rows whose trade resolves after the next slice starts."""
    n = len(idx)
    calib_end = int(n * (cfg.train_frac + cfg.calib_frac))
    train_end = calib_end if rolled else int(n * cfg.train_frac)
    es_start = int(train_end * (1.0 - cfg.early_stop_frac))
    bounds = {"fit": (0, es_start), "es": (es_start, train_end)}
    if rolled:
        bounds["calib"] = (calib_end, n)
    else:
        bounds["calib"] = (train_end, calib_end)
        bounds["test"] = (calib_end, n)
    out = {}
    for name, (a, b) in bounds.items():
        rows = np.arange(a, b)
        if b < n:
            rows = rows[end[rows] < idx[b]]
        out[name] = rows
    return out


# ───────────────────────── per target ─────────────────────────


@dataclass
class TargetModel:
    """Classifiers for one take-profit multiple, with test evidence and both replays."""

    target: float
    status: str  # "edge" | "no_edge" | "insufficient_data"
    base_rates: Dict[str, float]  # win rate of filled setups per grade ("A+", "B")
    loss_ratio: float = 1.0  # average non-win loss in R (timeouts lose less than the stop)
    models: Dict[str, Any] = field(default_factory=dict)
    calibrators: Dict[str, PlattCalibrator] = field(default_factory=dict)
    ensemble_calibrator: Optional[PlattCalibrator] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    rules_replay: Dict[str, Any] = field(default_factory=dict)
    model_replay: Dict[str, Any] = field(default_factory=dict)
    validated: bool = False  # the replay behind the served probability made money after costs

    def rules_probability(self, grades: np.ndarray) -> np.ndarray:
        return np.array([self.base_rates.get(g, np.nan) for g in grades], dtype=float)

    def model_probability(self, X: pd.DataFrame) -> tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        if not self.models or self.ensemble_calibrator is None:
            return None, {}
        raw = {name: _predict_raw(name, m, X) for name, m in self.models.items()}
        per_model = {name: self.calibrators[name].transform(r) for name, r in raw.items()}
        return self.ensemble_calibrator.transform(np.mean(np.vstack(list(raw.values())), axis=0)), per_model


def _base_rates(y: np.ndarray, grades: np.ndarray, cfg: SetupModelConfig) -> Dict[str, float]:
    overall = float(y.mean()) if len(y) else float("nan")
    rates = {}
    for g in GRADES:
        mask = grades == g
        rates[g] = float(y[mask].mean()) if mask.sum() >= cfg.min_grade_rows else overall
    return rates


def _loss_ratio(y: np.ndarray, r: np.ndarray) -> float:
    lost = y == 0
    return float(np.clip(np.mean(-r[lost]), 0.0, 1.0)) if lost.any() else 1.0


class SetupModel:
    """Per-target calibrated win chances for one coin's setups."""

    def __init__(self, config: Optional[SetupModelConfig] = None) -> None:
        self.config = config or SetupModelConfig()
        self.feature_names: List[str] = []
        self.targets: Dict[str, TargetModel] = {}
        self.summary: Dict[str, Any] = {}
        self.backtests: Dict[str, Dict[str, Any]] = {}  # per target: the rules on the training history (backtest.py)
        self.trained_at: Optional[str] = None
        self.version = FORMAT_VERSION

    @property
    def spec(self) -> SetupSpec:
        return self.config.spec

    @property
    def is_trained(self) -> bool:
        return bool(self.targets)

    def fit(self, X: pd.DataFrame, setups: pd.DataFrame, outcomes: pd.DataFrame,
            on_step: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        """Fit every target. `X` holds one feature row per setup (NaN allowed), in time order.

        `on_step(target)` is called before each target is fitted.
        """
        self.feature_names = list(X.columns)
        self.targets = {}
        spec = self.spec
        filled = outcomes["fill"].to_numpy() == "filled"
        resolved = outcomes["fill"].to_numpy() != "pending"
        days = max((setups.index[-1] - setups.index[0]) / pd.Timedelta(days=1), 1e-9) if len(setups) else 0.0
        a_plus = setups["grade"].to_numpy() == "A+"
        summary: Dict[str, Any] = {
            "setups": int(len(setups)),
            "a_plus": int(a_plus.sum()),
            "days": round(float(days), 1),
            "setups_per_day": round(len(setups) / days, 1) if days else None,
            "a_plus_per_day": round(float(a_plus.sum()) / days, 1) if days else None,
            "fill_rate": round(float(filled[resolved].mean()), 4) if resolved.any() else None,
            "median_risk_pct": round(float(np.median(setups["risk"])) * 100, 4) if len(setups) else None,
        }
        for T in spec.targets:
            if on_step is not None:
                on_step(T)
            key = target_key(T)
            model = self._fit_target(T, X, setups, outcomes)
            self.targets[key] = model
            summary[key] = {"status": model.status, "validated": model.validated, "base_rates": model.base_rates,
                            "rules_replay": model.rules_replay, "model_replay": model.model_replay,
                            **{k: model.metrics.get(k) for k in ("n_filled", "n_test", "brier_skill",
                                                                 "brier_skill_lower", "auc", "auc_rules")}}
            logger.info(f"Setup model {T}R: {summary[key]}")
        self.summary = summary
        self.trained_at = pd.Timestamp.now(tz="UTC").isoformat()
        return summary

    def _fit_target(self, T: float, X: pd.DataFrame, setups: pd.DataFrame, outcomes: pd.DataFrame) -> TargetModel:
        cfg, spec, key = self.config, self.spec, target_key(T)
        idx = setups["idx"].to_numpy()
        grades = setups["grade"].to_numpy()
        risk = setups["risk"].to_numpy()
        win_cost, loss_cost = spec.round_trip("won") / risk, spec.round_trip("lost") / risk
        fill = outcomes["fill"].to_numpy()
        win = outcomes[f"win_{key}"].to_numpy()
        gross = outcomes[f"r_{key}"].to_numpy()
        end = outcomes[f"end_{key}"].to_numpy()
        filled = fill == "filled"
        rows = np.flatnonzero(filled & np.isfinite(win))
        y = win[rows]

        def enough(labels: np.ndarray) -> bool:
            wins = int(labels.sum())
            return wins >= cfg.min_class_count and len(labels) - wins >= cfg.min_class_count

        s = _slices(idx[rows], end[rows], cfg) if len(rows) else {}
        base = _base_rates(y, grades[rows], cfg)
        loss_ratio = _loss_ratio(y, gross[rows])

        # Every A+ setup from the start of the test period on, filled or not, with a known result
        test_start = idx[rows[s["test"][0]]] if len(rows) and len(s["test"]) else (idx[-1] + 1 if len(idx) else 0)
        period = np.flatnonzero((idx >= test_start) & (fill != "pending") & ~(filled & ~np.isfinite(win)))
        busy = np.where(filled, end, outcomes["order_end"].to_numpy())[period]
        net = (gross - trade_cost_r(outcomes[f"outcome_{key}"].to_numpy(), risk, spec))[period]

        def replay(take: np.ndarray) -> Dict[str, Any]:
            return _replay(take, idx[period], busy, filled[period], np.nan_to_num(win[period]), net)

        rules_replay = replay(grades[period] == "A+")

        def passes(r: Dict[str, Any]) -> bool:
            return r["trades"] >= cfg.min_signal_trades and r["avg_net_r"] is not None and r["avg_net_r"] > 0

        if not s or not all(enough(y[s[k]]) for k in ("fit", "es", "calib", "test")):
            return TargetModel(target=T, status="insufficient_data", base_rates=base, loss_ratio=loss_ratio,
                               metrics={"n_filled": int(len(rows)), "reason": "too few wins or losses in a slice"},
                               rules_replay=rules_replay, validated=passes(rules_replay))

        # 1. Validate: fit on the oldest setups, calibrate on the next ones, score on the newest
        models = _fit_all(X.iloc[rows[s["fit"]]], y[s["fit"]], X.iloc[rows[s["es"]]], y[s["es"]], cfg)
        if not models:
            return TargetModel(target=T, status="insufficient_data", base_rates=base, loss_ratio=loss_ratio,
                               metrics={"n_filled": int(len(rows)), "reason": "no model trained"},
                               rules_replay=rules_replay, validated=passes(rules_replay))
        per_model_cal, ens_cal = _calibrate(models, X.iloc[rows[s["calib"]]], y[s["calib"]])
        seen = np.concatenate([s["fit"], s["es"], s["calib"]])
        base_seen = _base_rates(y[seen], grades[rows[seen]], cfg)
        loss_seen = _loss_ratio(y[seen], gross[rows[seen]])

        def ensemble(rows_: np.ndarray) -> np.ndarray:
            raw = [_predict_raw(name, m, X.iloc[rows_]) for name, m in models.items()]
            return ens_cal.transform(np.mean(np.vstack(raw), axis=0))

        test = s["test"]
        y_t, p_t = y[test], ensemble(rows[test])
        b_t = np.array([base_seen[g] for g in grades[rows[test]]])
        brier, brier_rules = float(np.mean((p_t - y_t) ** 2)), float(np.mean((b_t - y_t) ** 2))
        auc, auc_rules = _auc(y_t, p_t), _auc(y_t, b_t)
        metrics: Dict[str, Any] = {
            "n_filled": int(len(rows)),
            "n_test": int(len(test)),
            "test_win_rate": round(float(y_t.mean()), 4),
            "brier": round(brier, 5),
            "brier_rules": round(brier_rules, 5),
            "brier_skill": round(1.0 - brier / brier_rules, 4) if brier_rules > 0 else 0.0,
            "brier_skill_lower": round(_bootstrap_skill_lower(p_t, y_t, b_t, cfg), 4),
            "auc": round(auc, 4),
            "auc_rules": round(auc_rules, 4) if np.isfinite(auc_rules) else None,
            "loss_ratio": round(loss_seen, 4),
            "reliability": _reliability(p_t, y_t),
        }
        if len(test) < cfg.min_test_setups:
            status = "insufficient_data"
            metrics["reason"] = f"only {len(test)} filled test setups (need {cfg.min_test_setups})"
        elif metrics["brier_skill_lower"] > 0 and auc > max(0.5, auc_rules if np.isfinite(auc_rules) else 0.5):
            status = "edge"
        else:
            status = "no_edge"

        p_period = ensemble(period)
        model_replay = replay((grades[period] == "A+") & (p_period >= break_even(T, loss_seen, win_cost[period], loss_cost[period])))
        validated = passes(model_replay if status == "edge" else rules_replay)

        # 2. Roll forward: refit on everything but the newest slice, recalibrate on the newest slice
        r = _slices(idx[rows], end[rows], cfg, rolled=True)
        if all(enough(y[r[k]]) for k in ("fit", "es", "calib")):
            rolled = _fit_all(X.iloc[rows[r["fit"]]], y[r["fit"]], X.iloc[rows[r["es"]]], y[r["es"]], cfg)
            if rolled:
                models = rolled
                per_model_cal, ens_cal = _calibrate(models, X.iloc[rows[r["calib"]]], y[r["calib"]])

        return TargetModel(target=T, status=status, base_rates=base, loss_ratio=loss_ratio, models=models,
                           calibrators=per_model_cal, ensemble_calibrator=ens_cal, metrics=metrics,
                           rules_replay=rules_replay, model_replay=model_replay, validated=validated)

    def score(self, X: pd.DataFrame, setups: pd.DataFrame) -> List[Dict[str, Any]]:
        """Win chance, break-even and expected result per target for each setup row, plus the pick.

        Each row: {"targets": [...], "pick": position in targets}. The pick is the target with the
        highest expected result among those with an edge, else among all.
        """
        X = X.reindex(columns=self.feature_names)
        grades = setups["grade"].to_numpy()
        risk = setups["risk"].to_numpy()
        win_cost, loss_cost = self.spec.round_trip("won") / risk, self.spec.round_trip("lost") / risk
        per_target = []
        for key, tm in self.targets.items():
            rules_p = tm.rules_probability(grades)
            model_p, per_model = tm.model_probability(X) if len(X) else (None, {})
            served = model_p if (tm.status == "edge" and model_p is not None) else rules_p
            per_target.append((key, tm, rules_p, model_p, per_model, served))

        out = []
        for i in range(len(setups)):
            targets = []
            for key, tm, rules_p, model_p, per_model, served in per_target:
                p = float(served[i])
                targets.append({
                    "target_r": tm.target,
                    "probability": p,
                    "probability_source": "model" if served is model_p else "rules",
                    "model_probability": float(model_p[i]) if model_p is not None else None,
                    "rules_probability": float(rules_p[i]),
                    "per_model": {name: float(v[i]) for name, v in per_model.items()},
                    "break_even": float(break_even(tm.target, tm.loss_ratio, win_cost[i], loss_cost[i])),
                    "expected_r": float(expected_r(p, tm.target, tm.loss_ratio, win_cost[i], loss_cost[i])),
                    "status": tm.status,
                    "validated": tm.validated,
                })
            proven = [k for k, t in enumerate(targets) if t["status"] == "edge"]
            pool = proven or list(range(len(targets)))
            out.append({"targets": targets, "pick": max(pool, key=lambda k: targets[k]["expected_r"]) if pool else None})
        return out

    def evidence(self) -> Dict[str, Any]:
        """Test results per target and history counts, for the dashboard."""
        return {
            "trained_at": self.trained_at,
            "entry_interval": self.spec.entry_interval,
            "context_interval": self.spec.context_interval,
            "summary": {k: v for k, v in self.summary.items() if not isinstance(v, dict)},
            "backtests": self.backtests,
            "targets": {
                key: {"target_r": tm.target, "status": tm.status, "validated": tm.validated,
                      "base_rates": tm.base_rates, "rules_replay": tm.rules_replay, "model_replay": tm.model_replay,
                      "metrics": tm.metrics}
                for key, tm in self.targets.items()
            },
        }

    # ───────── persistence ─────────

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @staticmethod
    def load(path: str | Path) -> Optional["SetupModel"]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            model = joblib.load(p)
        except Exception as err:
            logger.warning(f"Could not load setup model {p}: {err}")
            return None
        if getattr(model, "version", None) != FORMAT_VERSION:
            logger.info(f"Ignoring setup model {p}: format {getattr(model, 'version', None)} != {FORMAT_VERSION}")
            return None
        return model


def setup_model_path(models_dir: str | Path, symbol_clean: str) -> Path:
    return Path(models_dir) / f"setup_model_{symbol_clean}.joblib"
