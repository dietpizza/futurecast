"""Tests for feature engineering, zero-leakage, and stationarity."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
from futurecast.ingestion.synthetic import generate_synthetic_data
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


def test_shitcoin_prediction_strategy_boundary_and_signals():
    """Verify that shitcoin prediction strategy applies strictly to shitcoins and calculates all 6 techniques."""
    from futurecast.features.shitcoin import is_high_beta_shitcoin, evaluate_shitcoin_microstructure

    # 1. Isolation boundary
    # Shitcoins / meme tokens
    for sym in ["B-PEPE_USDT", "PEPEUSDT", "B-DOGE_USDT", "DOGEUSDT", "B-FLOKI_USDT", "B-BONK_USDT", "B-SHIB_USDT"]:
        assert is_high_beta_shitcoin(sym, sector="MEME") is True
        assert is_high_beta_shitcoin(sym) is True

    # Big-volume major tokens MUST NOT be identified as shitcoins
    for sym in ["BTCUSDT", "B-BTC_USDT", "SOLUSDT", "B-SOL_USDT", "ETHUSDT", "B-ETH_USDT", "AVAXUSDT"]:
        assert is_high_beta_shitcoin(sym, sector="MAJOR" if "BTC" in sym else "L1") is False

    # 2. Evaluation on major coins returns is_shitcoin_strategy_active = False
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    btc = generate_synthetic_data("B-BTC_USDT", start=start, periods=60, base_price=65000.0, volatility=0.4)
    sol = generate_synthetic_data("B-SOL_USDT", start=start, periods=60, base_price=180.0, volatility=0.5)
    aligned_major = align_cross_asset_data(sol, {"B-BTC_USDT": btc})

    major_eval = evaluate_shitcoin_microstructure(aligned_major, target_symbol="B-SOL_USDT", sector="L1")
    assert major_eval["is_shitcoin_strategy_active"] is False
    assert "shitcoin_alpha_boost_bps" not in major_eval

    # 3. Evaluation on shitcoins returns all 6 quant metrics
    pepe = generate_synthetic_data("B-PEPE_USDT", start=start, periods=60, base_price=0.00001, volatility=0.8)
    doge = generate_synthetic_data("B-DOGE_USDT", start=start, periods=60, base_price=0.15, volatility=0.6)
    aligned_shitcoin = align_cross_asset_data(pepe, {"B-DOGE_USDT": doge})

    sc_eval = evaluate_shitcoin_microstructure(
        aligned_shitcoin, target_symbol="B-PEPE_USDT", sector="MEME", leader_symbol="B-DOGE_USDT"
    )
    assert sc_eval["is_shitcoin_strategy_active"] is True
    assert "volume_clock_burst" in sc_eval
    assert "shallow_book_skew" in sc_eval
    assert "sector_leader_spillover" in sc_eval
    assert "cvd_divergence" in sc_eval
    assert "squeeze_flush_risk" in sc_eval
    assert "cross_exchange_lead_lag_bps" in sc_eval
    assert "shitcoin_regime" in sc_eval
    assert "shitcoin_alpha_boost_bps" in sc_eval
