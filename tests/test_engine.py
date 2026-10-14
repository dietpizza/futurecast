"""Tests for the air-gapped PureForecastEngine, calibration, and drift monitor."""

from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from futurecast.config import load_config
from futurecast.engine.calibration import SignalCalibrator
from futurecast.engine.drift import DriftMonitor
from futurecast.engine.forecaster import PureForecastEngine
from futurecast.features.pipeline import FeaturePipeline
from futurecast.ingestion.binance import generate_synthetic_data
from futurecast.ingestion.storage import align_cross_asset_data
from futurecast.labels.residual_alpha import compute_residual_alpha_target
from futurecast.models.trainer import ModelTrainer
from futurecast.publisher.broadcaster import SignalBroadcaster
from futurecast.schema import SignalState


def test_signal_calibrator_thresholds():
    calibrator = SignalCalibrator(zscore_window=100, signal_threshold_z=1.5)

    # Warm up with near-zero noise
    for _ in range(50):
        calibrator.calibrate(raw_pred=np.random.normal(0, 0.001), beta_hedge_ratio=1.0)

    # Inject extreme positive residual alpha
    _, z_pos, state_pos, _ = calibrator.calibrate(raw_pred=0.05, beta_hedge_ratio=1.2)
    assert z_pos > 1.5
    assert state_pos == SignalState.BULLISH_RESIDUAL

    # Inject extreme negative residual alpha
    _, z_neg, state_neg, _ = calibrator.calibrate(raw_pred=-0.05, beta_hedge_ratio=1.2)
    assert z_neg < -1.5
    assert state_neg == SignalState.BEARISH_RESIDUAL


def test_drift_monitor_realization():
    monitor = DriftMonitor(horizon_bars=15)
    t0 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)

    monitor.record_forecast(
        timestamp=t0,
        predicted_residual=0.005,
        target_price=0.15,
        bench_price=65000.0,
        beta=1.5,
    )

    # 10 minutes later: not yet realized
    t1 = t0 + timedelta(minutes=10)
    rec1 = monitor.update_with_current_bar(t1, current_target_price=0.151, current_bench_price=65100.0)
    assert rec1 is None

    # 15 minutes later: realized!
    t2 = t0 + timedelta(minutes=15)
    rec2 = monitor.update_with_current_bar(t2, current_target_price=0.153, current_bench_price=65200.0)
    assert rec2 is not None
    assert rec2.predicted == 0.005
    assert len(monitor.completed_history) == 1


def test_pure_forecast_engine_air_gapped_guarantee(tmp_path):
    cfg = load_config("config")
    cfg.forecasting.output_dir = str(tmp_path / "forecasts")

    # Train a minimal fast model on synthetic data
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    periods = 400
    doge = generate_synthetic_data("DOGEUSDT", start=start, periods=periods, base_price=0.15, volatility=0.6, seed=1)
    btc = generate_synthetic_data("BTCUSDT", start=start, periods=periods, base_price=65000.0, volatility=0.4, seed=2)
    eth = generate_synthetic_data("ETHUSDT", start=start, periods=periods, base_price=2500.0, volatility=0.5, seed=3)

    aligned = align_cross_asset_data(doge, {"BTCUSDT": btc, "ETHUSDT": eth})
    pipeline = FeaturePipeline()
    features = pipeline.generate_features(aligned)
    target = compute_residual_alpha_target(aligned, features, horizon_minutes=15)

    valid_idx = features.dropna().index.intersection(target.dropna().index)
    X = features.loc[valid_idx]
    y = target.loc[valid_idx]

    trainer = ModelTrainer(cfg.model)
    trainer.train(X, y)

    broadcaster = SignalBroadcaster(output_dir=tmp_path / "forecasts")
    engine = PureForecastEngine(
        config=cfg,
        trainer=trainer,
        feature_pipeline=pipeline,
        broadcaster=broadcaster,
        min_buffer_size=60,
    )

    # Feed data to engine
    engine.feed_bars(aligned.iloc[:100])
    event = engine.feed_bars(aligned.iloc[[101]])

    assert event is not None
    # CRITICAL: Verify safety field
    assert event.execution_disabled is True
    assert event.symbol == "DOGEUSDT"
    assert event.benchmark_symbol == "BTCUSDT"

    # Verify audit file was written
    audit_files = list((tmp_path / "forecasts").glob("*.jsonl"))
    assert len(audit_files) > 0
