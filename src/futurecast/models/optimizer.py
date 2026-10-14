"""Optuna Bayesian hyperparameter optimization over purged walk-forward splits."""

from __future__ import annotations

import logging
from typing import Any, Dict
import numpy as np
import optuna
import pandas as pd

from futurecast.config import LightGBMConfig, OptunaConfig
from futurecast.models.trainer import ModelTrainer
from futurecast.validation.metrics import compute_spearman_ic
from futurecast.validation.purged_cv import PurgedWalkForwardCV

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """Optimizes LightGBM hyperparameters to maximize Mean Walk-Forward Spearman IC."""

    def __init__(
        self,
        cv: PurgedWalkForwardCV,
        optuna_config: OptunaConfig | None = None,
        base_lgb_config: LightGBMConfig | None = None,
    ) -> None:
        self.cv = cv
        self.optuna_config = optuna_config or OptunaConfig()
        self.base_lgb_config = base_lgb_config or LightGBMConfig()

    def optimize(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> Dict[str, Any]:
        """Run Optuna study."""
        # Align features and target
        common_idx = features.dropna().index.intersection(target.dropna().index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]

        splits = list(self.cv.split(X))
        if not splits:
            raise ValueError("No valid walk-forward splits found for hyperparameter optimization.")

        def objective(trial: optuna.Trial) -> float:
            suggested_params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 100, 500, step=50),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.85),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            }

            fold_ics = []
            for split in splits:
                X_tr = X.iloc[split.train_indices]
                y_tr = y.iloc[split.train_indices]
                X_te = X.iloc[split.test_indices]
                y_te = y.iloc[split.test_indices]

                trainer = ModelTrainer(self.base_lgb_config)
                trainer.train(
                    X_tr,
                    y_tr,
                    X_val=X_te,
                    y_val=y_te,
                    custom_params=suggested_params,
                )

                preds = trainer.predict(X_te)
                ic = compute_spearman_ic(preds, y_te)
                fold_ics.append(ic)

            # Maximize mean Spearman IC across out-of-sample folds
            return float(np.mean(fold_ics))

        study = optuna.create_study(
            study_name=self.optuna_config.study_name,
            direction="maximize",
        )
        study.optimize(
            objective,
            n_trials=self.optuna_config.n_trials,
            timeout=self.optuna_config.timeout_seconds,
            show_progress_bar=False,
        )

        logger.info(f"Optimization complete. Best Trial Mean IC: {study.best_value:.4f}")
        logger.info(f"Best Parameters: {study.best_params}")
        return {
            "best_params": study.best_params,
            "best_ic": study.best_value,
        }
