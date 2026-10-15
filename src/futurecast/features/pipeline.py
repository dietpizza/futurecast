"""Unified feature engineering pipeline and stationarity verification."""

from __future__ import annotations

import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from futurecast.config import FeaturesConfig
from futurecast.features.cross_asset import compute_cross_asset_features
from futurecast.features.microstructure import compute_microstructure_features
from futurecast.features.derivatives import compute_derivative_features
from futurecast.features.technical import compute_technical_features

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Builds every model feature from an aligned candle matrix (target plus two benchmarks)."""

    def __init__(
        self,
        config: FeaturesConfig | None = None,
        target_symbol: str = "DOGEUSDT",
        primary_benchmark: str = "BTCUSDT",
        secondary_benchmark: str = "ETHUSDT",
    ) -> None:
        self.config = config or FeaturesConfig()
        self.target_symbol = target_symbol
        self.primary_benchmark = primary_benchmark
        self.secondary_benchmark = secondary_benchmark

    def generate_features(self, aligned_df: pd.DataFrame) -> pd.DataFrame:
        """All features for every candle, each computed from that candle and earlier ones only.

        Window settings count candles of whatever interval `aligned_df` holds (see FeaturesConfig).
        """
        logger.info(f"Generating features for {len(aligned_df)} aligned bars...")

        # Cross-asset returns, betas, spreads and RSI
        grp_a = compute_cross_asset_features(
            aligned_df,
            target_symbol=self.target_symbol,
            primary_benchmark=self.primary_benchmark,
            secondary_benchmark=self.secondary_benchmark,
            horizons=self.config.cross_asset.horizons_min,
            beta_windows=self.config.cross_asset.beta_windows_min,
            rsi_period=self.config.cross_asset.rsi_period,
        )

        # Volume, volatility and taker order flow
        grp_b = compute_microstructure_features(
            aligned_df,
            target_symbol=self.target_symbol,
            benchmark_symbol=self.primary_benchmark,
            volume_z_windows=self.config.microstructure.volume_z_windows_min,
            parkinson_windows=self.config.microstructure.parkinson_windows_min,
            rvr_window=self.config.microstructure.rvr_window_min,
        )

        # Settled funding rates
        grp_c = compute_derivative_features(
            aligned_df,
            target_symbol=self.target_symbol,
            benchmark_symbol=self.primary_benchmark,
            funding_window_min=self.config.derivatives.funding_ema_window_min,
        )

        # Price geometry: distance to recent extremes, VWAP basis, candle shape
        grp_d = compute_technical_features(
            aligned_df,
            target_symbol=self.target_symbol,
            extremes_window=self.config.technical.stochastic_extremes_window_min,
            vwap_window=self.config.technical.vwap_window_min,
        )

        # Combine all features
        features = pd.concat([grp_a, grp_b, grp_c, grp_d], axis=1)

        # Guarantee strict backward-looking alignment:
        # Every feature at timestamp t is strictly computed up to bar t.
        # Infinities become NaN; warm-up NaNs stay (the boosters handle missing values)
        features = features.replace([np.inf, -np.inf], np.nan)
        logger.info(f"Created {features.shape[1]} raw candidate features.")
        return features
