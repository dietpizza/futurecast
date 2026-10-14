"""Tests for feature engineering, zero-leakage, and stationarity."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
from futurecast.ingestion.binance import generate_synthetic_data
from futurecast.ingestion.storage import align_cross_asset_data
from futurecast.features.pipeline import FeaturePipeline


def test_feature_pipeline_computes_all_features():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    # Generate 1500 bars so 1440m windows warm up
    periods = 1600
    doge = generate_synthetic_data("DOGEUSDT", start=start, periods=periods, base_price=0.15, volatility=0.6, seed=1)
    btc = generate_synthetic_data("BTCUSDT", start=start, periods=periods, base_price=65000.0, volatility=0.4, seed=2)
    eth = generate_synthetic_data("ETHUSDT", start=start, periods=periods, base_price=2500.0, volatility=0.5, seed=3)

    aligned = align_cross_asset_data(doge, {"BTCUSDT": btc, "ETHUSDT": eth})
    pipeline = FeaturePipeline()
    features = pipeline.generate_features(aligned)

    # Check key features exist
    assert "DOGEUSDT_ret_1m" in features.columns
    assert "beta_btc_1440m" in features.columns
    assert "beta_residual_momentum_15m" in features.columns
    assert "volume_zscore_60m" in features.columns
    assert "parkinson_vol_60m" in features.columns
    assert "funding_rate_dev_3d" in features.columns
    assert "dist_to_extremes_60m" in features.columns
    assert "vwap_basis_4h_bps" in features.columns
    assert "upper_wick_ratio" in features.columns

    # Verify no infinite values
    assert not np.isinf(features.dropna().values).any()

    # Verify that the final rows have fully valid numeric values after warmup
    assert not features.iloc[-1].isna().any()


def test_feature_zero_leakage():
    """Ensure that modifying future bars does NOT change historical feature values."""
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    periods = 200
    doge1 = generate_synthetic_data("DOGEUSDT", start=start, periods=periods, base_price=0.15, volatility=0.6, seed=1)
    btc1 = generate_synthetic_data("BTCUSDT", start=start, periods=periods, base_price=65000.0, volatility=0.4, seed=2)
    eth1 = generate_synthetic_data("ETHUSDT", start=start, periods=periods, base_price=2500.0, volatility=0.5, seed=3)

    aligned1 = align_cross_asset_data(doge1, {"BTCUSDT": btc1, "ETHUSDT": eth1})
    pipeline = FeaturePipeline()
    feats1 = pipeline.generate_features(aligned1)

    # Mutate only the last 10 bars in a new dataset
    aligned2 = aligned1.copy()
    aligned2.iloc[-10:, aligned2.columns.get_loc("DOGEUSDT_close")] *= 5.0
    feats2 = pipeline.generate_features(aligned2)

    # The features up to index -11 MUST be identical (strictly backward-looking)
    sub1 = feats1.iloc[:-11].dropna()
    sub2 = feats2.iloc[:-11].dropna()
    np.testing.assert_allclose(sub1.values, sub2.values, rtol=1e-5, atol=1e-7)
