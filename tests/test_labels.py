"""Tests for residual alpha target generation."""

from datetime import datetime, timezone
from futurecast.ingestion.synthetic import generate_synthetic_data
from futurecast.ingestion.storage import align_cross_asset_data
from futurecast.features.pipeline import FeaturePipeline
from futurecast.labels.residual_alpha import compute_residual_alpha_target


def test_residual_alpha_target_calculation():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    periods = 300
    doge = generate_synthetic_data("DOGEUSDT", start=start, periods=periods, base_price=0.15, volatility=0.6, seed=1)
    btc = generate_synthetic_data("BTCUSDT", start=start, periods=periods, base_price=65000.0, volatility=0.4, seed=2)
    eth = generate_synthetic_data("ETHUSDT", start=start, periods=periods, base_price=2500.0, volatility=0.5, seed=3)

    aligned = align_cross_asset_data(doge, {"BTCUSDT": btc, "ETHUSDT": eth})
    pipeline = FeaturePipeline()
    features = pipeline.generate_features(aligned)

    target = compute_residual_alpha_target(
        aligned_df=aligned,
        features_df=features,
        target_symbol="DOGEUSDT",
        benchmark_symbol="BTCUSDT",
        horizon_minutes=15,
    )

    assert target.name == "target_residual_alpha_15m"
    assert len(target) == len(aligned)
    # The last 15 bars must be NaN because forward returns are undefined at the end of the series
    assert target.iloc[-15:].isna().all()
    # Prior bars should have finite numbers where features are valid
    valid_slice = target.iloc[100:-15]
    assert valid_slice.notna().any()
