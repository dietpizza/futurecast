"""Online model drift monitoring and forward-testing performance auditor."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import logging
from typing import NamedTuple, Optional
import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class PendingForecast(NamedTuple):
    timestamp: datetime
    predicted_residual: float
    target_price_at_t: float
    bench_price_at_t: float
    beta_at_t: float
    horizon_bars: int = 15


class RealizedRecord(NamedTuple):
    timestamp: datetime
    predicted: float
    realized: float


class DriftMonitor:
    """Tracks real-time forecast accuracy without trading.
    
    Stores pending forecasts and, as new bars close 15 minutes later, calculates
    the realized residual alpha to track rolling Spearman IC and error metrics.
    """

    def __init__(self, horizon_bars: int = 15, max_history: int = 1440) -> None:
        self.horizon_bars = horizon_bars
        self.pending_queue: deque[PendingForecast] = deque()
        self.completed_history: deque[RealizedRecord] = deque(maxlen=max_history)

    def record_forecast(
        self,
        timestamp: datetime,
        predicted_residual: float,
        target_price: float,
        bench_price: float,
        beta: float,
    ) -> None:
        """Enqueue new forecast awaiting future price realization."""
        self.pending_queue.append(
            PendingForecast(
                timestamp=timestamp,
                predicted_residual=predicted_residual,
                target_price_at_t=target_price,
                bench_price_at_t=bench_price,
                beta_at_t=beta,
                horizon_bars=self.horizon_bars,
            )
        )

    def update_with_current_bar(
        self,
        current_time: datetime,
        current_target_price: float,
        current_bench_price: float,
    ) -> Optional[RealizedRecord]:
        """Check if any pending forecast has reached its 15-minute horizon and evaluate it."""
        if not self.pending_queue:
            return None

        # Check the oldest forecast
        oldest = self.pending_queue[0]
        # In 1m bars, 15m difference:
        time_elapsed_min = (current_time - oldest.timestamp).total_seconds() / 60.0

        if time_elapsed_min >= self.horizon_bars:
            f = self.pending_queue.popleft()
            
            # Realized log returns
            r_target = np.log(current_target_price / f.target_price_at_t)
            r_bench = np.log(current_bench_price / f.bench_price_at_t)

            # Realized residual alpha
            realized_alpha = r_target - (f.beta_at_t * r_bench)

            record = RealizedRecord(
                timestamp=f.timestamp,
                predicted=f.predicted_residual,
                realized=realized_alpha,
            )
            self.completed_history.append(record)
            return record

        return None

    def get_rolling_metrics(self, window: int = 240) -> dict[str, float]:
        """Compute rolling Spearman IC, RMSE, and hit rate over completed forecasts."""
        if len(self.completed_history) < 20:
            return {"rolling_ic": 0.0, "rolling_rmse": 0.0, "sample_count": len(self.completed_history)}

        recent = list(self.completed_history)[-window:]
        preds = np.array([r.predicted for r in recent])
        reals = np.array([r.realized for r in recent])

        ic, _ = spearmanr(preds, reals)
        rmse = float(np.sqrt(np.mean((preds - reals) ** 2)))
        hit_rate = float(np.mean(np.sign(preds) == np.sign(reals)))

        res = {
            "rolling_ic": float(ic) if not np.isnan(ic) else 0.0,
            "rolling_rmse": rmse,
            "hit_rate": hit_rate,
            "sample_count": len(recent),
        }

        if res["rolling_ic"] < -0.05:
            logger.warning(f"MODEL DRIFT WARNING: Rolling IC dropped to {res['rolling_ic']:.3f}")

        return res
