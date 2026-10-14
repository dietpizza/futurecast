"""Air-gapped online forecasting runtime engine."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict, Optional, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

from futurecast.config import AppConfig
from futurecast.engine.calibration import SignalCalibrator
from futurecast.engine.drift import DriftMonitor
from futurecast.features.pipeline import FeaturePipeline
from futurecast.ingestion.factory import get_symbol_sector
from futurecast.models.trainer import ModelTrainer
from futurecast.schema import ForecastEvent, ModelForecastItem, CrossVerificationSummary, SignalState

if TYPE_CHECKING:
    from futurecast.models.ensemble import MultiModelEnsemble
    from futurecast.publisher.broadcaster import SignalBroadcaster

logger = logging.getLogger(__name__)


class PureForecastEngine:
    """Core runtime engine for real-time inference, signal calibration, and drift tracking.
    
    Permits ZERO order execution or trading interaction.
    """

    def __init__(
        self,
        config: AppConfig,
        trainer: Union[ModelTrainer, MultiModelEnsemble],
        feature_pipeline: FeaturePipeline,
        broadcaster: Optional[Any] = None,
        min_buffer_size: int = 1440,
        ensemble: Optional[MultiModelEnsemble] = None,
    ) -> None:
        # Enforce zero-execution policy
        config.enforce_zero_execution_policy()
        self.config = config
        self.trainer = trainer
        self.ensemble = ensemble
        self.feature_pipeline = feature_pipeline
        if broadcaster is None:
            from futurecast.publisher.broadcaster import SignalBroadcaster
            self.broadcaster = SignalBroadcaster(output_dir=config.forecasting.output_dir)
        else:
            self.broadcaster = broadcaster
        self.calibrator = SignalCalibrator(
            zscore_window=config.forecasting.zscore_window_minutes,
            signal_threshold_z=config.forecasting.signal_threshold_z,
        )
        self.drift_monitor = DriftMonitor(
            horizon_bars=config.features.target.horizon_min
        )
        self.min_buffer_size = min_buffer_size
        self._bar_buffer = pd.DataFrame()

    def feed_bars(self, aligned_df_slice: pd.DataFrame) -> Optional[ForecastEvent]:
        """Ingest new aligned 1-minute bars and generate forecast on the latest closed bar."""
        if aligned_df_slice.empty:
            return None

        # Append to sliding buffer
        if self._bar_buffer.empty:
            self._bar_buffer = aligned_df_slice.copy()
        else:
            self._bar_buffer = pd.concat([self._bar_buffer, aligned_df_slice])
            self._bar_buffer = self._bar_buffer[~self._bar_buffer.index.duplicated(keep="last")]

        # Ensure buffer doesn't grow unbounded
        max_buffer = self.min_buffer_size * 2
        if len(self._bar_buffer) > max_buffer:
            self._bar_buffer = self._bar_buffer.iloc[-max_buffer:]

        if len(self._bar_buffer) < self.min_buffer_size:
            logger.info(f"Warming up buffer: {len(self._bar_buffer)}/{self.min_buffer_size} bars...")
            return None

        # Compute point-in-time features on buffer
        features = self.feature_pipeline.generate_features(self._bar_buffer)
        latest_features = features.iloc[[-1]]

        # Current prices and timestamp
        latest_timestamp = self._bar_buffer.index[-1]
        target_symbol = self.config.assets.target_asset
        bench_symbol = self.config.assets.benchmark_assets[0]

        curr_target_price = float(self._bar_buffer[f"{target_symbol}_close"].iloc[-1])
        curr_bench_price = float(self._bar_buffer[f"{bench_symbol}_close"].iloc[-1])

        # Current point-in-time rolling beta
        beta_col = "beta_btc_1440m"
        beta = float(latest_features[beta_col].iloc[0]) if beta_col in latest_features else 1.0
        if np.isnan(beta):
            beta = 1.0

        # Update drift monitor for completed horizons
        self.drift_monitor.update_with_current_bar(
            current_time=latest_timestamp,
            current_target_price=curr_target_price,
            current_bench_price=curr_bench_price,
        )

        # Predict residual alpha: check if multi-model ensemble is present
        if hasattr(self.trainer, "get_cross_verification"):
            expected_bps, z_score, signal_state, conf_pct, model_items, cv_summary = (
                self.trainer.get_cross_verification(
                    latest_features=latest_features,
                    beta=beta,
                    calibrator=self.calibrator,
                )
            )
            raw_pred = expected_bps / 10000.0
        elif self.ensemble is not None:
            expected_bps, z_score, signal_state, conf_pct, model_items, cv_summary = (
                self.ensemble.get_cross_verification(
                    latest_features=latest_features,
                    beta=beta,
                    calibrator=self.calibrator,
                )
            )
            raw_pred = expected_bps / 10000.0
        else:
            raw_pred = float(self.trainer.predict(latest_features)[0])
            expected_bps, z_score, signal_state, conf_pct = self.calibrator.calibrate(
                raw_pred=raw_pred,
                beta_hedge_ratio=beta,
            )
            model_items = {
                "LightGBM": ModelForecastItem(
                    model_name="LightGBM",
                    expected_residual_alpha_bps=expected_bps,
                    z_score=z_score,
                    signal=signal_state,
                    confidence_percentile=conf_pct,
                )
            }
            cv_summary = CrossVerificationSummary(
                concordance=1.0,
                consensus_signal=signal_state,
                models_count=1,
                bullish_count=1 if signal_state == SignalState.BULLISH_RESIDUAL else 0,
                bearish_count=1 if signal_state == SignalState.BEARISH_RESIDUAL else 0,
                neutral_count=1 if signal_state == SignalState.NEUTRAL else 0,
                dispersion_bps=0.0,
            )

        # Microstructure context
        ctx = {}
        for col in ["volume_zscore_60m", "relative_volume_ratio_15m", "funding_rate_bps", "oi_delta_15m_pct"]:
            if col in latest_features.columns:
                val = latest_features[col].iloc[0]
                ctx[col] = float(val) if not np.isnan(val) else 0.0

        sector = get_symbol_sector(target_symbol)

        event = ForecastEvent(
            timestamp=latest_timestamp,
            symbol=target_symbol,
            benchmark_symbol=bench_symbol,
            current_price=curr_target_price,
            benchmark_price=curr_bench_price,
            expected_residual_alpha_15m_bps=expected_bps,
            z_score=z_score,
            beta_hedge_ratio=beta,
            signal=signal_state,
            confidence_percentile=conf_pct,
            microstructure_context=ctx,
            execution_disabled=True,
            model_predictions=model_items,
            cross_verification=cv_summary,
            sector=sector,
        )

        # Enqueue for future forward drift tracking
        self.drift_monitor.record_forecast(
            timestamp=latest_timestamp,
            predicted_residual=raw_pred,
            target_price=curr_target_price,
            bench_price=curr_bench_price,
            beta=beta,
        )

        # Broadcast event
        self.broadcaster.broadcast(event)
        return event
