"""Multi-Model Cross Verification Suite: LightGBM, XGBoost, CatBoost and Consensus Ensemble."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from futurecast.config import LightGBMConfig
from futurecast.engine.calibration import SignalCalibrator
from futurecast.schema import ModelForecastItem, CrossVerificationSummary, SignalState
from futurecast.validation.metrics import compute_regression_metrics

logger = logging.getLogger(__name__)


class LightGBMAlphaModel:
    """LightGBM booster with Huber/L1 residual loss."""

    def __init__(self, config: Optional[LightGBMConfig] = None) -> None:
        self.name = "LightGBM"
        self.config = config or LightGBMConfig()
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> lgb.Booster:
        self.feature_names = list(X_train.columns)
        params: Dict[str, Any] = {
            "objective": self.config.objective,
            "metric": self.config.metric,
            "boosting_type": self.config.boosting_type,
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.num_leaves,
            "max_depth": self.config.max_depth,
            "min_child_samples": self.config.min_child_samples,
            "subsample": self.config.subsample,
            "subsample_freq": self.config.subsample_freq,
            "colsample_bytree": self.config.colsample_bytree,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "n_jobs": self.config.n_jobs,
            "verbose": -1,
        }

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        valid_sets = [train_data]
        valid_names = ["train"]
        callbacks = []

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=self.feature_names)
            valid_sets.append(val_data)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds, verbose=False))

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBM model not trained or loaded.")
        cols = [c for c in self.feature_names if c in X.columns]
        if len(cols) != len(self.feature_names):
            # Align columns if missing
            aligned_X = pd.DataFrame(index=X.index)
            for c in self.feature_names:
                aligned_X[c] = X[c] if c in X.columns else 0.0
            return self.model.predict(aligned_X)
        return self.model.predict(X[self.feature_names])

    def save(self, file_path: str | Path) -> None:
        if self.model is None:
            return
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(p))

    def load(self, file_path: str | Path) -> bool:
        p = Path(file_path)
        if not p.exists():
            return False
        self.model = lgb.Booster(model_file=str(p))
        self.feature_names = self.model.feature_name()
        return True


class XGBoostAlphaModel:
    """XGBoost booster with pseudo-Huber regression loss."""

    def __init__(self) -> None:
        self.name = "XGBoost"
        self.model: Optional[xgb.Booster] = None
        self.feature_names: List[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> xgb.Booster:
        self.feature_names = list(X_train.columns)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        evals = [(dtrain, "train")]

        params: Dict[str, Any] = {
            "objective": "reg:pseudohubererror",
            "eval_metric": "mae",
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "verbosity": 0,
        }

        early_stopping_rounds = None
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, "val"))
            early_stopping_rounds = 30

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBoost model not trained or loaded.")
        aligned_X = pd.DataFrame(index=X.index)
        for c in self.feature_names:
            aligned_X[c] = X[c] if c in X.columns else 0.0
        dmatrix = xgb.DMatrix(aligned_X[self.feature_names], feature_names=self.feature_names)
        return self.model.predict(dmatrix)

    def save(self, file_path: str | Path) -> None:
        if self.model is None:
            return
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(p))

    def load(self, file_path: str | Path) -> bool:
        p = Path(file_path)
        if not p.exists():
            return False
        booster = xgb.Booster()
        booster.load_model(str(p))
        self.model = booster
        self.feature_names = booster.feature_names or []
        return True


class CatBoostAlphaModel:
    """CatBoost Regressor with MAE loss to reject crypto microstructure flash wicks."""

    def __init__(self) -> None:
        self.name = "CatBoost"
        self.model: Optional[CatBoostRegressor] = None
        self.feature_names: List[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> CatBoostRegressor:
        self.feature_names = list(X_train.columns)
        eval_set = (X_val, y_val) if (X_val is not None and y_val is not None) else None

        self.model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.03,
            depth=5,
            loss_function="MAE",
            eval_metric="MAE",
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=False,
            early_stopping_rounds=30 if eval_set is not None else None,
        )
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CatBoost model not trained or loaded.")
        aligned_X = pd.DataFrame(index=X.index)
        for c in self.feature_names:
            aligned_X[c] = X[c] if c in X.columns else 0.0
        return self.model.predict(aligned_X[self.feature_names])

    def save(self, file_path: str | Path) -> None:
        if self.model is None:
            return
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(p))

    def load(self, file_path: str | Path) -> bool:
        p = Path(file_path)
        if not p.exists():
            return False
        model = CatBoostRegressor()
        model.load_model(str(p))
        self.model = model
        self.feature_names = list(model.feature_names_) if hasattr(model, "feature_names_") and model.feature_names_ else []
        return True


class MultiModelEnsemble:
    """Orchestrates multi-model training, inference, and cross-verification."""

    def __init__(self, lgb_config: Optional[LightGBMConfig] = None) -> None:
        self.lgb_config = lgb_config
        self.models: Dict[str, Any] = {
            "LightGBM": LightGBMAlphaModel(lgb_config),
            "XGBoost": XGBoostAlphaModel(),
            "CatBoost": CatBoostAlphaModel(),
        }
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.is_loaded = False

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Train LightGBM, XGBoost, and CatBoost and score them.

        X_val/y_val drive early stopping. When X_test/y_test are given, metrics come from
        that untouched slice; scoring on the early-stopping slice would flatter the models.
        """
        self.metrics = {}
        X_score, y_score = (X_test, y_test) if X_test is not None and y_test is not None else (X_val, y_val)
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name} model...")
                model.train(X_train, y_train, X_val, y_val)
                if X_score is not None and y_score is not None:
                    preds = model.predict(X_score)
                    m = compute_regression_metrics(preds, y_score)
                    self.metrics[name] = m
                    logger.info(f"{name} -> Hit Rate: {m['hit_rate']*100:.1f}%, IC: {m['spearman_ic']:+.4f}")
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")

        self.is_loaded = any(m.model is not None for m in self.models.values())
        return self.metrics

    def predict_all(self, X: pd.DataFrame) -> Dict[str, float]:
        """Point predictions from every model that is actually trained (missing ones are omitted)."""
        preds: Dict[str, float] = {}
        for name, model in self.models.items():
            if model.model is not None:
                try:
                    preds[name] = float(model.predict(X)[-1])
                except Exception as e:
                    logger.debug(f"Prediction failed for {name}: {e}")
        return preds

    def get_cross_verification(
        self,
        latest_features: pd.DataFrame,
        beta: float,
        calibrator: SignalCalibrator,
    ) -> Tuple[float, float, SignalState, float, Dict[str, ModelForecastItem], CrossVerificationSummary]:
        """Perform multi-model cross-verification and return consensus and per-model diagnostics."""
        raw_preds = self.predict_all(latest_features)
        if not raw_preds:
            empty = CrossVerificationSummary(
                concordance=0.0, consensus_signal=SignalState.NEUTRAL, models_count=0,
                bullish_count=0, bearish_count=0, neutral_count=0, dispersion_bps=0.0,
            )
            return 0.0, 0.0, SignalState.NEUTRAL, 50.0, {}, empty

        model_items: Dict[str, ModelForecastItem] = {}
        signals: List[SignalState] = []
        alphas: List[float] = []
        z_scores: List[float] = []

        for name, raw_p in raw_preds.items():
            exp_bps, z, sig, conf = calibrator.calibrate(raw_pred=raw_p, beta_hedge_ratio=beta, key=name)
            m_stats = self.metrics.get(name, {})
            item = ModelForecastItem(
                model_name=name,
                expected_residual_alpha_bps=exp_bps,
                z_score=z,
                signal=sig,
                confidence_percentile=conf,
                hit_rate=m_stats.get("hit_rate"),
                spearman_ic=m_stats.get("spearman_ic"),
            )
            model_items[name] = item
            signals.append(sig)
            alphas.append(exp_bps)
            z_scores.append(z)

        # Cross-verification consensus
        consensus_alpha = float(np.mean(alphas))
        consensus_z = float(np.mean(z_scores))
        dispersion = float(np.std(alphas)) if len(alphas) > 1 else 0.0

        bullish_cnt = sum(1 for s in signals if s == SignalState.BULLISH_RESIDUAL)
        bearish_cnt = sum(1 for s in signals if s == SignalState.BEARISH_RESIDUAL)
        neutral_cnt = sum(1 for s in signals if s == SignalState.NEUTRAL)
        n_models = len(signals)

        # Determine consensus signal & concordance
        if bullish_cnt > bearish_cnt and bullish_cnt >= n_models / 2:
            consensus_sig = SignalState.BULLISH_RESIDUAL
            concordance = bullish_cnt / n_models
        elif bearish_cnt > bullish_cnt and bearish_cnt >= n_models / 2:
            consensus_sig = SignalState.BEARISH_RESIDUAL
            concordance = bearish_cnt / n_models
        else:
            consensus_sig = SignalState.NEUTRAL
            concordance = neutral_cnt / n_models if neutral_cnt > 0 else 0.5

        # Average confidence
        confs = [item.confidence_percentile for item in model_items.values()]
        avg_conf = float(np.mean(confs)) if confs else 50.0

        summary = CrossVerificationSummary(
            concordance=concordance,
            consensus_signal=consensus_sig,
            models_count=n_models,
            bullish_count=bullish_cnt,
            bearish_count=bearish_cnt,
            neutral_count=neutral_cnt,
            dispersion_bps=dispersion,
        )

        return consensus_alpha, consensus_z, consensus_sig, avg_conf, model_items, summary

    def save_models(self, directory: str | Path, symbol_clean: str, meta: Optional[Dict[str, Any]] = None) -> None:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        if meta is not None:
            with open(p / f"residual_meta_{symbol_clean}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        # Save each model
        if self.models["LightGBM"].model is not None:
            self.models["LightGBM"].save(p / f"lightgbm_{symbol_clean}_residual_alpha.txt")
        if self.models["XGBoost"].model is not None:
            self.models["XGBoost"].save(p / f"xgboost_{symbol_clean}_residual_alpha.json")
        if self.models["CatBoost"].model is not None:
            self.models["CatBoost"].save(p / f"catboost_{symbol_clean}_residual_alpha.cbm")

        # Save metrics
        if self.metrics:
            metrics_file = p / f"metrics_{symbol_clean}.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(self.metrics, f, indent=2)

    def load_models(self, directory: str | Path, symbol_clean: str, bar_interval: Optional[str] = None) -> bool:
        """Load this symbol's own models. Never borrows another coin's weights.

        With `bar_interval`, models are only loaded if they were trained on that candle
        interval (models trained on other candles would see differently scaled features).
        """
        p = Path(directory)
        if bar_interval is not None:
            meta_file = p / f"residual_meta_{symbol_clean}.json"
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                meta = {}
            if meta.get("bar_interval") != bar_interval:
                logger.info(f"No {bar_interval} residual models for {symbol_clean}; needs training")
                self.is_loaded = False
                return False

        files = {
            "LightGBM": p / f"lightgbm_{symbol_clean}_residual_alpha.txt",
            "XGBoost": p / f"xgboost_{symbol_clean}_residual_alpha.json",
            "CatBoost": p / f"catboost_{symbol_clean}_residual_alpha.cbm",
        }
        loaded_any = False
        for name, path in files.items():
            if path.exists() and self.models[name].load(path):
                loaded_any = True

        metrics_file = p / f"metrics_{symbol_clean}.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except Exception:
                self.metrics = {}

        self.is_loaded = loaded_any
        return loaded_any
