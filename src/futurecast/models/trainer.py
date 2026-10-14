"""LightGBM model training, evaluation, and feature importance reporting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

from futurecast.config import LightGBMConfig
from futurecast.validation.metrics import compute_regression_metrics, compute_spearman_ic

logger = logging.getLogger(__name__)


def ic_eval(preds: np.ndarray, train_data: lgb.Dataset) -> Tuple[str, float, bool]:
    """Custom LightGBM evaluation metric for Spearman Rank IC (higher is better)."""
    labels = train_data.get_label()
    ic = compute_spearman_ic(preds, labels)
    return "spearman_ic", ic, True


class ModelTrainer:
    """Trains and serializes regularized LightGBM models for residual alpha prediction."""

    def __init__(self, config: LightGBMConfig | None = None) -> None:
        self.config = config or LightGBMConfig()
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> lgb.Booster:
        """Train LightGBM with early stopping and regularization constraints."""
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
            "verbose": self.config.verbose,
        }

        if custom_params:
            params.update(custom_params)

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
            raise RuntimeError("Model has not been trained or loaded yet.")
        return self.model.predict(X[self.feature_names])

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        preds = self.predict(X)
        return compute_regression_metrics(preds, y)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        importances = self.model.feature_importance(importance_type=importance_type)
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)

    def save(self, file_path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info(f"Model saved successfully to {path}")

    def load(self, file_path: str | Path) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = lgb.Booster(model_file=str(path))
        self.feature_names = self.model.feature_name()
        logger.info(f"Model loaded successfully from {path} ({len(self.feature_names)} features)")
