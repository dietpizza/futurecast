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
from futurecast.features.shitcoin import is_high_beta_shitcoin, evaluate_shitcoin_microstructure
from futurecast.ingestion.factory import get_symbol_sector
from futurecast.schema import ForecastEvent, SignalState

if TYPE_CHECKING:
    from futurecast.models.ensemble import LightGBMAlphaModel, MultiModelEnsemble
    from futurecast.publisher.broadcaster import SignalBroadcaster

logger = logging.getLogger(__name__)


class PureForecastEngine:
    """Core runtime engine for real-time inference, signal calibration, and drift tracking.
    
    Permits ZERO order execution or trading interaction.
    """

    def __init__(
        self,
        config: AppConfig,
        trainer: Union[LightGBMAlphaModel, MultiModelEnsemble],
        feature_pipeline: FeaturePipeline,
        broadcaster: Optional[Any] = None,
        min_buffer_size: int = 1440,
        ensemble: Optional[MultiModelEnsemble] = None,
        max_buffer_size: Optional[int] = None,
        bar_minutes: int = 1,
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
        self.max_buffer_size = max_buffer_size or min_buffer_size * 2
        self.bar_minutes = bar_minutes
        self._bar_buffer = pd.DataFrame()
        # Latest state, kept so the order book can be refreshed between candles
        self.last_features: Optional[pd.DataFrame] = None
        self.last_event: Optional[ForecastEvent] = None

    def reset(self) -> None:
        """Drop buffered candles and z-score history (e.g. after a data gap or retrain)."""
        self._bar_buffer = pd.DataFrame()
        self.calibrator.reset()
        self.last_features = None
        self.last_event = None

    def refresh_orderbook(self, orderbook_depth: Optional[Dict[str, Any]]) -> Optional[ForecastEvent]:
        """Attach a fresh order book to the latest event, without a new candle."""
        ev = self.last_event
        if ev is None:
            return None
        ev.orderbook_depth = orderbook_depth or {}
        return ev

    def feed_bars(
        self,
        aligned_df_slice: pd.DataFrame,
        orderbook_depth: Optional[Dict[str, Any]] = None,
    ) -> Optional[ForecastEvent]:
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
        if len(self._bar_buffer) > self.max_buffer_size:
            self._bar_buffer = self._bar_buffer.iloc[-self.max_buffer_size:]

        if len(self._bar_buffer) < self.min_buffer_size:
            logger.info(f"Warming up buffer: {len(self._bar_buffer)}/{self.min_buffer_size} bars...")
            return None

        # Compute point-in-time features on buffer
        features = self.feature_pipeline.generate_features(self._bar_buffer)
        latest_features = features.iloc[[-1]]
        self.last_features = features

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

        # Predict residual alpha with multi-model cross-verification
        ensemble = self.ensemble if self.ensemble is not None else self.trainer
        expected_bps, z_score, signal_state, conf_pct, model_items, cv_summary = (
            ensemble.get_cross_verification(
                latest_features=latest_features,
                beta=beta,
                calibrator=self.calibrator,
            )
        )
        raw_pred = expected_bps / 10000.0

        # Microstructure context
        ctx = {}
        for col in ["volume_zscore_60m", "relative_volume_ratio_15m", "funding_rate_bps", "oi_delta_15m_pct"]:
            if col in latest_features.columns:
                val = latest_features[col].iloc[0]
                ctx[col] = float(val) if not np.isnan(val) else 0.0

        sector = get_symbol_sector(target_symbol)

        # High-Beta Shitcoin Strategy (strictly only for shitcoins; big vol coins unaffected)
        if is_high_beta_shitcoin(target_symbol, sector):
            shitcoin_metrics = evaluate_shitcoin_microstructure(
                df=self._bar_buffer,
                target_symbol=target_symbol,
                sector=sector,
                leader_symbol=bench_symbol,
            )
            ctx.update(shitcoin_metrics)
            alpha_boost = shitcoin_metrics.get("shitcoin_alpha_boost_bps", 0.0)
            if alpha_boost != 0.0:
                expected_bps = round(expected_bps + alpha_boost, 2)
                # Directional confirmation
                if expected_bps > 15.0 and z_score > 0.8:
                    signal_state = SignalState.BULLISH_RESIDUAL
                elif expected_bps < -15.0 and z_score < -0.8:
                    signal_state = SignalState.BEARISH_RESIDUAL
            if shitcoin_metrics.get("shitcoin_regime") == "DORMANT_CHOP":
                expected_bps = round(expected_bps * 0.5, 2)
                if abs(expected_bps) < 12.0:
                    signal_state = SignalState.NEUTRAL
        else:
            ctx["is_shitcoin_strategy_active"] = False

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
            orderbook_depth=orderbook_depth or {},
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
        self.last_event = event
        return event
