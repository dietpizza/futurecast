"""Statistical and quantitative evaluation metrics for financial forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_spearman_ic(y_pred: np.ndarray | pd.Series, y_true: np.ndarray | pd.Series) -> float:
    """Calculate the Information Coefficient (Spearman Rank Correlation)."""
    valid = ~(np.isnan(y_pred) | np.isnan(y_true))
    if np.sum(valid) < 10:
        return 0.0
    corr, _ = spearmanr(y_pred[valid], y_true[valid])
    return float(corr) if not np.isnan(corr) else 0.0


def compute_hit_rate(y_pred: np.ndarray | pd.Series, y_true: np.ndarray | pd.Series) -> float:
    """Directional accuracy: percentage of times sign(y_pred) == sign(y_true)."""
    valid = ~(np.isnan(y_pred) | np.isnan(y_true))
    if np.sum(valid) == 0:
        return 0.5
    correct = np.sign(y_pred[valid]) == np.sign(y_true[valid])
    return float(np.mean(correct))


def compute_regression_metrics(
    y_pred: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute comprehensive regression and alpha evaluation metrics."""
    valid = ~(np.isnan(y_pred) | np.isnan(y_true))
    yp = np.asarray(y_pred)[valid]
    yt = np.asarray(y_true)[valid]

    if len(yp) < 5:
        return {"spearman_ic": 0.0, "rmse": 0.0, "mae": 0.0, "hit_rate": 0.5}

    ic = compute_spearman_ic(yp, yt)
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae = float(np.mean(np.abs(yp - yt)))
    hit_rate = compute_hit_rate(yp, yt)

    return {
        "spearman_ic": ic,
        "rmse": rmse,
        "mae": mae,
        "hit_rate": hit_rate,
    }


def compute_theoretical_long_short_sharpe(
    predictions: pd.Series,
    realized_residual: pd.Series,
    threshold: float = 0.0,
    annualization_factor: float = np.sqrt(365 * 24 * 4),  # 15m intervals per year
) -> float:
    """Evaluate theoretical friction-free Sharpe ratio of an alpha signal:
    
    Position = sign(predictions) if |predictions| > threshold else 0
    Return = Position * realized_residual
    """
    signal = np.where(predictions > threshold, 1.0, np.where(predictions < -threshold, -1.0, 0.0))
    strat_ret = signal * realized_residual
    mean_ret = strat_ret.mean()
    std_ret = strat_ret.std()

    if std_ret == 0 or np.isnan(std_ret):
        return 0.0

    return float((mean_ret / std_ret) * annualization_factor)
