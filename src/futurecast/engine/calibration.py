"""Forecast calibration, rolling z-score standardization, and signal classification."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict
import numpy as np
from scipy.stats import norm

from futurecast.schema import SignalState


class SignalCalibrator:
    """Calibrates raw regression predictions into standardized z-scores and regime signals."""

    def __init__(
        self,
        zscore_window: int = 1440,
        signal_threshold_z: float = 1.5,
    ) -> None:
        self.zscore_window = zscore_window
        self.signal_threshold_z = signal_threshold_z
        self._histories: Dict[str, Deque[float]] = {}

    def reset(self) -> None:
        self._histories.clear()

    def calibrate(
        self,
        raw_pred: float,
        beta_hedge_ratio: float,
        key: str = "default",
    ) -> tuple[float, float, SignalState, float]:
        """Convert raw predicted residual return into:

        Returns:
            (expected_bps, z_score, signal_state, confidence_percentile)

        Each `key` (one per model) keeps its own history, so one model's predictions
        never shift another model's z-score. The "confidence percentile" is
        normal_cdf(|z|): how unusual this prediction is versus the model's own recent
        predictions, not a probability that the trade works.
        """
        history = self._histories.setdefault(key, deque(maxlen=self.zscore_window))
        history.append(raw_pred)
        expected_bps = float(raw_pred * 10000.0)

        if len(history) < 30:
            # Insufficient calibration history, default to neutral
            return expected_bps, 0.0, SignalState.NEUTRAL, 50.0

        hist_arr = np.array(history)
        mean = float(np.mean(hist_arr))
        std = float(np.std(hist_arr))

        if std < 1e-8 or np.isnan(std):
            z_score = 0.0
        else:
            z_score = float((raw_pred - mean) / std)

        # Confidence percentile from standard normal CDF
        conf_pct = float(norm.cdf(abs(z_score)) * 100.0)

        # Signal state determination
        if z_score > self.signal_threshold_z:
            state = SignalState.BULLISH_RESIDUAL
        elif z_score < -self.signal_threshold_z:
            state = SignalState.BEARISH_RESIDUAL
        else:
            state = SignalState.NEUTRAL

        return expected_bps, z_score, state, conf_pct
