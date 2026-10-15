"""Model-based long / short trade risk: how far price may move against a position.

For every horizon and side, LightGBM, XGBoost and CatBoost quantile regressors learn the
95th percentile of the adverse excursion: the worst move against a position entered at a
candle close, before the horizon ends (see labels.barriers.adverse_excursion). The target
is measured in units of the horizon volatility known at entry, so it stays comparable
between calm and wild regimes.

History is used in time order with the same purged slices as the win probabilities
(models.trade_probability):

    | fit | early-stop |  calibrate  |    test    |
    oldest ------------------------------------> newest

Calibration: one multiplier per model, and one for their average, scales the predicted
bound so exactly 95% of calibration-slice moves stay inside it (split conformal).

Validation: the baseline is volatility alone — the 95th percentile of the normalised
excursion on everything seen before the test slice, times current volatility. On the
newest, unseen slice the model must beat that baseline on pinball (quantile) loss with
90% block-bootstrap confidence while still covering close to 95% of moves. Otherwise
the baseline is served, like the win probabilities fall back to their base rate.

Serving: the procedure is rolled forward (refit on everything before the newest slice,
recalibrated on the newest slice). Per side it reports the 95% worst move, a 0-100 score
ranking that move against this coin's own newest slice, and the highest leverage whose
liquidation price stays beyond it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor

from futurecast.labels.barriers import BarrierSpec, adverse_excursion, horizon_volatility
from futurecast.models.trade_probability import (
    HORIZONS_BY_INTERVAL,
    MODEL_NAMES,
    SIDES,
    _purged_slices,
    _rolled_slices,
)

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
_MIN_BOUND = 1e-3  # floor on a predicted bound (volatility units) before scaling
_SIGMA_COL = "risk_log_horizon_vol"  # the target's own scale, known at entry, given to every model


@dataclass
class TradeRiskConfig:
    """All lengths (horizons, volatility window) are counted in candles of `interval`."""

    interval: str = "1m"
    barrier: BarrierSpec = field(default_factory=BarrierSpec)  # volatility window and trading cost
    horizons: Dict[str, int] = field(default_factory=lambda: dict(HORIZONS_BY_INTERVAL["1m"]))
    quantile: float = 0.95  # share of moves the worst-move bound must contain
    coverage_tolerance: float = 0.03  # test coverage may fall this far below `quantile`
    train_frac: float = 0.6
    calib_frac: float = 0.2  # the test slice gets the remainder
    early_stop_frac: float = 0.2
    min_slice_rows: int = 200
    min_independent_test: int = 30  # non-overlapping test windows needed before judging skill
    bootstrap_rounds: int = 500
    skill_confidence: float = 0.9
    max_rounds: int = 400
    random_seed: int = 42
    maintenance_margin: float = 0.005  # share of position value the exchange keeps at liquidation
    leverage_ladder: Tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    reference_points: int = 201  # stored quantiles of recent bounds, used to score "now vs usual"


# ───────────────────────── quantile regressors ─────────────────────────


def _fit_lightgbm(X_fit, z_fit, X_es, z_es, cfg: TradeRiskConfig):
    params = {
        "objective": "quantile",
        "alpha": cfg.quantile,
        "metric": "quantile",
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
        lgb.Dataset(X_fit, label=z_fit),
        num_boost_round=cfg.max_rounds,
        valid_sets=[lgb.Dataset(X_es, label=z_es)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )


def _fit_xgboost(X_fit, z_fit, X_es, z_es, cfg: TradeRiskConfig):
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": cfg.quantile,
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
        xgb.DMatrix(X_fit, label=z_fit),
        num_boost_round=cfg.max_rounds,
        evals=[(xgb.DMatrix(X_es, label=z_es), "es")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )


def _fit_catboost(X_fit, z_fit, X_es, z_es, cfg: TradeRiskConfig):
    model = CatBoostRegressor(
        iterations=cfg.max_rounds,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=5.0,
        loss_function=f"Quantile:alpha={cfg.quantile}",
        random_seed=cfg.random_seed,
        early_stopping_rounds=40,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(X_fit, z_fit, eval_set=(X_es, z_es), verbose=False)
    return model


_FITTERS = {"LightGBM": _fit_lightgbm, "XGBoost": _fit_xgboost, "CatBoost": _fit_catboost}


def _predict(name: str, model: Any, X: pd.DataFrame) -> np.ndarray:
    if name == "LightGBM":
        raw = model.predict(X, num_iteration=model.best_iteration or None)
    elif name == "XGBoost":
        best = getattr(model, "best_iteration", None)
        it_range = (0, best + 1) if best is not None else (0, 0)
        raw = model.predict(xgb.DMatrix(X), iteration_range=it_range)
    else:
        raw = model.predict(X)
    return np.maximum(np.asarray(raw, dtype=float), _MIN_BOUND)


def _fit_all(X_fit, z_fit, X_es, z_es, horizon: int, cfg: TradeRiskConfig) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for name in MODEL_NAMES:
        try:
            models[name] = _FITTERS[name](X_fit, z_fit, X_es, z_es, cfg)
        except Exception as err:  # one library failing should not sink the others
            logger.warning(f"{name} risk model failed for horizon {horizon}: {err}")
    return models


# ───────────────────────── calibration & evaluation ─────────────────────────


def _conformal_scale(bound: np.ndarray, z: np.ndarray, q: float) -> float:
    """Multiplier that makes `scale * bound` contain exactly a share `q` of the observed z."""
    return float(np.quantile(z / bound, q))


def _calibrate(models: Dict[str, Any], X, z, q: float) -> Tuple[Dict[str, float], float]:
    raw = {name: _predict(name, m, X) for name, m in models.items()}
    per_model = {name: _conformal_scale(r, z, q) for name, r in raw.items()}
    ensemble = _conformal_scale(np.mean(np.vstack(list(raw.values())), axis=0), z, q)
    return per_model, ensemble


def _ensemble_bound(models: Dict[str, Any], scale: float, X) -> np.ndarray:
    return scale * np.mean(np.vstack([_predict(name, m, X) for name, m in models.items()]), axis=0)


def _pinball(bound: np.ndarray, z: np.ndarray, q: float) -> np.ndarray:
    diff = z - bound
    return np.maximum(q * diff, (q - 1.0) * diff)


def _bootstrap_skill_lower(loss_model, loss_base, block: int, cfg: TradeRiskConfig) -> float:
    """Lower confidence bound on pinball skill, resampling contiguous horizon-length blocks."""
    n_blocks = len(loss_model) // block
    if n_blocks < 2:
        return float("-inf")
    trim = n_blocks * block
    model_blocks = loss_model[:trim].reshape(n_blocks, block).sum(axis=1)
    base_blocks = loss_base[:trim].reshape(n_blocks, block).sum(axis=1)
    rng = np.random.default_rng(cfg.random_seed)
    picks = rng.integers(0, n_blocks, size=(cfg.bootstrap_rounds, n_blocks))
    skill = 1.0 - model_blocks[picks].sum(axis=1) / np.maximum(base_blocks[picks].sum(axis=1), 1e-12)
    return float(np.quantile(skill, 1.0 - cfg.skill_confidence))


def _reference(bounds: np.ndarray, cfg: TradeRiskConfig) -> np.ndarray:
    return np.quantile(bounds, np.linspace(0.0, 1.0, cfg.reference_points))


# ───────────────────────── per horizon / side ─────────────────────────


@dataclass
class SideRisk:
    """Production quantile models + conformal scales for one horizon and side, with test evidence."""

    status: str  # "edge" | "no_edge" | "insufficient_data"
    baseline_z: float  # recent 95th percentile of the normalised excursion, served without edge
    scale: float = 1.0  # conformal multiplier for the models' average
    models: Dict[str, Any] = field(default_factory=dict)
    model_scales: Dict[str, float] = field(default_factory=dict)
    reference: Optional[np.ndarray] = None  # served bounds (fraction of price) over the newest slice
    metrics: Dict[str, Any] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame, sigma: float) -> Dict[str, Any]:
        """Served worst-move bound plus model, baseline and per-model bounds for the last row."""
        baseline = self.baseline_z * sigma
        if not self.models:
            return {"worst_move": baseline, "model_worst_move": None, "baseline_worst_move": baseline, "per_model": {}}
        raw = {name: float(_predict(name, m, X)[-1]) for name, m in self.models.items()}
        per_model = {name: self.model_scales[name] * r * sigma for name, r in raw.items()}
        model_move = self.scale * float(np.mean(list(raw.values()))) * sigma
        served = model_move if self.status == "edge" else baseline
        return {"worst_move": served, "model_worst_move": model_move, "baseline_worst_move": baseline,
                "per_model": per_model}


def _fit_side(X: pd.DataFrame, z: np.ndarray, sigma: np.ndarray, horizon: int, cfg: TradeRiskConfig) -> SideRisk:
    q = cfg.quantile
    s = _purged_slices(len(z), horizon, cfg)
    recent = slice(s["calib"].start, len(z))
    recent_base = float(np.quantile(z[recent], q)) if len(z[recent]) else float("nan")

    if any(s[k].stop - s[k].start < cfg.min_slice_rows for k in ("fit", "es", "calib", "test")):
        return SideRisk(status="insufficient_data", baseline_z=recent_base,
                        metrics={"n_rows": int(len(z)), "reason": "too few rows in a slice"})

    # 1. Validate: fit on the oldest data, calibrate on the next slice, score on unseen newest data
    models = _fit_all(X.iloc[s["fit"]], z[s["fit"]], X.iloc[s["es"]], z[s["es"]], horizon, cfg)
    if not models:
        return SideRisk(status="insufficient_data", baseline_z=recent_base, metrics={"reason": "no model trained"})
    model_scales, scale = _calibrate(models, X.iloc[s["calib"]], z[s["calib"]], q)
    base_before = float(np.quantile(z[: s["calib"].stop], q))

    test = s["test"]
    z_t, sig_t = z[test], sigma[test]
    bound_t = _ensemble_bound(models, scale, X.iloc[test])
    loss_model, loss_base = _pinball(bound_t, z_t, q), _pinball(np.full_like(z_t, base_before), z_t, q)
    skill = 1.0 - float(loss_model.sum() / max(loss_base.sum(), 1e-12))
    metrics: Dict[str, Any] = {
        "n_rows": int(len(z)),
        "n_test": int(len(z_t)),
        "n_independent": int(len(z_t) // horizon),
        "quantile": q,
        "coverage": round(float(np.mean(z_t <= bound_t)), 4),
        "coverage_baseline": round(float(np.mean(z_t <= base_before)), 4),
        "pinball_skill": round(skill, 4),
        "pinball_skill_lower": round(_bootstrap_skill_lower(loss_model, loss_base, horizon, cfg), 4),
        "avg_bound_pct": round(float(np.mean(bound_t * sig_t)) * 100, 4),
        "avg_bound_baseline_pct": round(float(np.mean(base_before * sig_t)) * 100, 4),
        "scale": round(scale, 4),
    }

    if metrics["n_independent"] < cfg.min_independent_test:
        status = "insufficient_data"
        metrics["reason"] = f"only {metrics['n_independent']} independent test windows (need {cfg.min_independent_test})"
    elif metrics["pinball_skill_lower"] > 0 and metrics["coverage"] >= q - cfg.coverage_tolerance:
        status = "edge"
    else:
        status = "no_edge"

    # 2. Roll forward: refit on everything but the newest slice, recalibrate on the newest slice
    r = _rolled_slices(len(z), horizon, cfg)
    if all(r[k].stop - r[k].start >= cfg.min_slice_rows for k in ("fit", "es", "calib")):
        rolled = _fit_all(X.iloc[r["fit"]], z[r["fit"]], X.iloc[r["es"]], z[r["es"]], horizon, cfg)
        if rolled:
            models = rolled
            model_scales, scale = _calibrate(models, X.iloc[r["calib"]], z[r["calib"]], q)

    newest = r["calib"]
    if status == "edge":
        served = _ensemble_bound(models, scale, X.iloc[newest]) * sigma[newest]
    else:
        served = recent_base * sigma[newest]

    return SideRisk(
        status=status,
        baseline_z=recent_base,
        scale=scale,
        models=models,
        model_scales=model_scales,
        reference=_reference(served, cfg) if len(served) else None,
        metrics=metrics,
    )


def risk_level(score: float) -> str:
    if score >= 90:
        return "EXTREME"
    if score >= 70:
        return "HIGH"
    if score >= 35:
        return "MODERATE"
    return "LOW"


def max_safe_leverage(worst_move: float, cfg: TradeRiskConfig) -> int:
    """Highest leverage on the ladder whose liquidation price stays beyond the worst move."""
    safe = [lev for lev in cfg.leverage_ladder
            if 1.0 / lev - cfg.maintenance_margin - cfg.barrier.cost > worst_move]
    return max(safe) if safe else min(cfg.leverage_ladder)


# ───────────────────────── public model ─────────────────────────


class TradeRiskModel:
    """Per-horizon long/short worst-move bounds, risk scores and leverage limits for one symbol."""

    def __init__(self, config: Optional[TradeRiskConfig] = None) -> None:
        self.config = config or TradeRiskConfig()
        self.feature_names: List[str] = []
        self.sides: Dict[str, Dict[str, SideRisk]] = {}
        self.trained_at: Optional[str] = None
        self.version = FORMAT_VERSION

    @property
    def is_trained(self) -> bool:
        return bool(self.sides)

    def fit(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        on_step: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        """Fit every horizon/side. Features may contain NaN (handled natively by the boosters).

        `on_step(horizon, side)` is called before each horizon/side is fitted.
        """
        self.feature_names = list(features.columns)
        self.sides = {}
        summary: Dict[str, Any] = {}

        for h_key, horizon in self.config.horizons.items():
            sigma = horizon_volatility(close, horizon, self.config.barrier).to_numpy()
            excursion = adverse_excursion(close, high, low, horizon)
            usable = excursion["long_mae"].notna().to_numpy() & np.isfinite(sigma) & (sigma > 0)
            X = features.loc[usable].assign(**{_SIGMA_COL: np.log(sigma[usable])})
            self.sides[h_key] = {}
            for side in SIDES:
                if on_step is not None:
                    on_step(h_key, side)
                z = excursion[f"{side}_mae"].to_numpy()[usable] / sigma[usable]
                side_risk = _fit_side(X, z, sigma[usable], horizon, self.config)
                self.sides[h_key][side] = side_risk
                m = side_risk.metrics
                summary[f"{h_key}_{side}"] = {
                    "status": side_risk.status,
                    "coverage": m.get("coverage"),
                    "pinball_skill": m.get("pinball_skill"),
                    "pinball_skill_lower": m.get("pinball_skill_lower"),
                    "n_independent": m.get("n_independent"),
                }
                logger.info(f"Risk model {h_key} {side}: {summary[f'{h_key}_{side}']}")

        self.trained_at = pd.Timestamp.now(tz="UTC").isoformat()
        return summary

    def predict_latest(self, features: pd.DataFrame, close: pd.Series) -> Dict[str, Any]:
        """Worst-move bounds, risk scores and leverage limits for the latest bar, per horizon."""
        cfg = self.config
        latest = features.reindex(columns=self.feature_names).iloc[[-1]]
        out: Dict[str, Any] = {}

        for h_key, horizon in cfg.horizons.items():
            sigma = float(horizon_volatility(close, horizon, cfg.barrier).iloc[-1])
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            X = latest.assign(**{_SIGMA_COL: np.log(sigma)})
            sides_out: Dict[str, Any] = {}
            for side in SIDES:
                model = self.sides.get(h_key, {}).get(side)
                if model is None or not np.isfinite(model.baseline_z):
                    continue
                pred = model.predict(X, sigma)
                worst = pred["worst_move"]
                ref = model.reference
                score = float(np.searchsorted(ref, worst, side="right")) / len(ref) * 100 if ref is not None else 50.0
                score = int(np.clip(round(score), 1, 99))
                sides_out[side] = {
                    "worst_move_pct": round(worst * 100, 4),
                    "model_worst_move_pct": round(pred["model_worst_move"] * 100, 4) if pred["model_worst_move"] is not None else None,
                    "baseline_worst_move_pct": round(pred["baseline_worst_move"] * 100, 4),
                    "per_model_pct": {k: round(v * 100, 4) for k, v in pred["per_model"].items()},
                    "score": score,
                    "level": risk_level(score),
                    "max_leverage": max_safe_leverage(worst, cfg),
                    "status": model.status,
                    "metrics": model.metrics,
                }
            if len(sides_out) < len(SIDES):
                continue

            ratio = sides_out["long"]["worst_move_pct"] / max(sides_out["short"]["worst_move_pct"], 1e-9)
            bias = "LONG_RISKIER" if ratio >= 1.25 else ("SHORT_RISKIER" if ratio <= 0.8 else "BALANCED")
            out[h_key] = {
                "candle_interval": cfg.interval,
                "horizon_bars": horizon,
                "as_of": str(close.index[-1]),
                "quantile": cfg.quantile,
                "bias": bias,
                "risk_ratio": round(ratio, 3),
                **sides_out,
            }
        return out

    # ───────── persistence ─────────

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @staticmethod
    def load(path: str | Path) -> Optional["TradeRiskModel"]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            model = joblib.load(p)
        except Exception as err:
            logger.warning(f"Could not load risk model {p}: {err}")
            return None
        if getattr(model, "version", None) != FORMAT_VERSION:
            logger.info(f"Ignoring risk model {p}: format {getattr(model, 'version', None)} != {FORMAT_VERSION}")
            return None
        return model


def risk_model_path(models_dir: str | Path, symbol_clean: str) -> Path:
    return Path(models_dir) / f"trade_risk_{symbol_clean}.joblib"
