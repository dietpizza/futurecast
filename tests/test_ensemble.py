"""Tests for Multi-Model Cross-Verification Suite (LightGBM, XGBoost, CatBoost)."""

import numpy as np
import pandas as pd
import pytest

from futurecast.engine.calibration import SignalCalibrator
from futurecast.models.ensemble import (
    MultiModelEnsemble,
    LightGBMAlphaModel,
    XGBoostAlphaModel,
    CatBoostAlphaModel,
)
from futurecast.schema import SignalState


@pytest.fixture
def sample_dataset():
    np.random.seed(42)
    n = 200
    features = {
        "doge_ret_1m": np.random.normal(0, 0.002, n),
        "doge_ret_15m": np.random.normal(0, 0.005, n),
        "beta_btc_1440m": np.random.normal(1.2, 0.1, n),
        "volume_zscore_60m": np.random.normal(0, 1.0, n),
        "parkinson_vol_60m": np.random.uniform(0.01, 0.05, n),
    }
    X = pd.DataFrame(features)
    # Synthetic target correlated with features
    y = pd.Series(
        0.4 * X["doge_ret_15m"] - 0.2 * X["volume_zscore_60m"] * 0.001 + np.random.normal(0, 0.001, n),
        name="target",
    )
    return X, y


def test_individual_models_train_and_predict(sample_dataset):
    X, y = sample_dataset
    split = 150
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_val, y_val = X.iloc[split:], y.iloc[split:]

    # 1. LightGBM
    lgb_m = LightGBMAlphaModel()
    lgb_m.train(X_train, y_train, X_val, y_val)
    preds_lgb = lgb_m.predict(X_val)
    assert len(preds_lgb) == len(X_val)

    # 2. XGBoost
    xgb_m = XGBoostAlphaModel()
    xgb_m.train(X_train, y_train, X_val, y_val)
    preds_xgb = xgb_m.predict(X_val)
    assert len(preds_xgb) == len(X_val)

    # 3. CatBoost
    cb_m = CatBoostAlphaModel()
    cb_m.train(X_train, y_train, X_val, y_val)
    preds_cb = cb_m.predict(X_val)
    assert len(preds_cb) == len(X_val)


def test_multi_model_ensemble_cross_verification(sample_dataset, tmp_path):
    X, y = sample_dataset
    split = 150
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_val, y_val = X.iloc[split:], y.iloc[split:]

    ensemble = MultiModelEnsemble()
    metrics = ensemble.train_all(X_train, y_train, X_val, y_val)

    assert "LightGBM" in metrics
    assert "XGBoost" in metrics
    assert "CatBoost" in metrics
    assert "hit_rate" in metrics["LightGBM"]
    assert "spearman_ic" in metrics["LightGBM"]

    # Test point prediction
    latest_feat = X.iloc[[-1]]
    calibrator = SignalCalibrator(zscore_window=60, signal_threshold_z=0.5)

    alpha, z, sig, conf, model_items, cv_summary = ensemble.get_cross_verification(
        latest_features=latest_feat,
        beta=1.2,
        calibrator=calibrator,
    )

    assert isinstance(alpha, float)
    assert isinstance(z, float)
    assert sig in (SignalState.BULLISH_RESIDUAL, SignalState.BEARISH_RESIDUAL, SignalState.NEUTRAL)
    assert 0.0 <= conf <= 100.0

    # Verify per-model diagnostics
    assert len(model_items) == 3
    assert "LightGBM" in model_items
    assert "XGBoost" in model_items
    assert "CatBoost" in model_items

    for m_name, item in model_items.items():
        assert item.model_name == m_name
        assert item.signal in (SignalState.BULLISH_RESIDUAL, SignalState.BEARISH_RESIDUAL, SignalState.NEUTRAL)
        assert item.hit_rate is not None

    # Verify cross-verification summary
    assert cv_summary.models_count == 3
    assert 0.0 <= cv_summary.concordance <= 1.0
    assert cv_summary.bullish_count + cv_summary.bearish_count + cv_summary.neutral_count == 3

    # Test model save & load roundtrip
    ensemble.save_models(tmp_path, "b_doge_usdt")
    loaded_ensemble = MultiModelEnsemble()
    loaded_ok = loaded_ensemble.load_models(tmp_path, "b_doge_usdt")
    assert loaded_ok is True
    assert loaded_ensemble.models["LightGBM"].model is not None
    assert loaded_ensemble.models["XGBoost"].model is not None
    assert loaded_ensemble.models["CatBoost"].model is not None
