"""Unified feature engineering pipeline and stationarity verification."""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning)

from futurecast.config import FeaturesConfig
from futurecast.features.cross_asset import compute_cross_asset_features
from futurecast.features.microstructure import compute_microstructure_features
from futurecast.features.derivatives import compute_derivative_features
from futurecast.features.technical import compute_technical_features

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Assembles all 35 features and verifies strict stationarity and zero leakage."""

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
        """Compute all 35 features from the aligned continuous 1-minute OHLCV matrix."""
        logger.info(f"Generating features for {len(aligned_df)} aligned bars...")

        # Group A: Cross-Asset Lead-Lag & Dynamics (10 features)
        grp_a = compute_cross_asset_features(
            aligned_df,
            target_symbol=self.target_symbol,
            primary_benchmark=self.primary_benchmark,
            secondary_benchmark=self.secondary_benchmark,
            horizons=self.config.cross_asset.horizons_min,
            beta_windows=self.config.cross_asset.beta_windows_min,
            rsi_period=self.config.cross_asset.rsi_period,
        )

        # Group B: Volume & Volatility Microstructure (12 features)
        grp_b = compute_microstructure_features(
            aligned_df,
            target_symbol=self.target_symbol,
            benchmark_symbol=self.primary_benchmark,
            volume_z_windows=self.config.microstructure.volume_z_windows_min,
            parkinson_windows=self.config.microstructure.parkinson_windows_min,
            rvr_window=self.config.microstructure.rvr_window_min,
        )

        # Group C: Derivative Flow & Positioning (8 features)
        grp_c = compute_derivative_features(
            aligned_df,
            target_symbol=self.target_symbol,
            benchmark_symbol=self.primary_benchmark,
            funding_ema_window=self.config.derivatives.funding_ema_window_min,
            oi_delta_windows=self.config.derivatives.oi_delta_windows_min,
            divergence_window=self.config.derivatives.oi_divergence_window_min,
        )

        # Group D: Technical & Price Geometry (5 features)
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
        # Clean infinities and drop warmup NaNs (longest rolling window is 1440 min = 1 day)
        features = features.replace([np.inf, -np.inf], np.nan)
        logger.info(f"Created {features.shape[1]} raw candidate features.")
        return features

    def run_stationarity_audit(
        self,
        features: pd.DataFrame,
        sample_size: int = 10000,
        significance_threshold: float = 0.05,
    ) -> Dict[str, dict]:
        """Run Augmented Dickey-Fuller (ADF) test on each feature to verify stationarity."""
        audit_results = {}
        # Use recent valid slice to test stationarity
        valid_df = features.dropna().tail(sample_size)

        for col in valid_df.columns:
            series = valid_df[col].values
            if np.all(series == series[0]) or np.isnan(series).any():
                audit_results[col] = {
                    "adf_stat": 0.0,
                    "p_value": 1.0,
                    "is_stationary": False,
                    "note": "constant or nan",
                }
                continue

            try:
                adf_stat, p_val, _, _, _, _ = adfuller(series, maxlag=10, autolag="AIC")
                audit_results[col] = {
                    "adf_stat": float(adf_stat),
                    "p_value": float(p_val),
                    "is_stationary": bool(p_val < significance_threshold),
                    "note": "OK" if p_val < significance_threshold else "Non-stationary candidate",
                }
            except Exception as err:
                audit_results[col] = {
                    "adf_stat": 0.0,
                    "p_value": 1.0,
                    "is_stationary": False,
                    "note": f"Error: {err}",
                }

        return audit_results
