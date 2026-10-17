"""The funding rate at the latest closed candle, read against the coin's own recent rates.

`funding_view` takes aligned frames (interval -> frame with `{symbol}_funding_rate` etc. columns,
indexed by candle open time) and returns None when the candles carry no funding rate.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from futurecast.features.setups import interval_minutes

FUNDING_CHART = "1h"
FUNDING_DAYS = 90


def percentile_of(history: np.ndarray, value: float) -> Optional[float]:
    """Share of `history` below `value`, ties counted half, in percent."""
    history = history[np.isfinite(history)]
    if not len(history) or not np.isfinite(value):
        return None
    return float(100.0 * (np.sum(history < value) + 0.5 * np.sum(history == value)) / len(history))


def funding(rates: pd.Series, per_day: int, days: int = FUNDING_DAYS) -> Optional[Dict[str, float]]:
    """The latest settled funding rate and its percentile among the rates in force over `days`."""
    r = rates.astype(float).to_numpy()[-days * per_day:]
    r = r[np.isfinite(r)]
    if not len(r):
        return None
    return {"rate_pct": 100.0 * float(r[-1]), "mean_pct": 100.0 * float(r.mean()),
            "percentile": percentile_of(r[:-1], r[-1]) if len(r) > 1 else None, "days": days}


def funding_view(candles: Dict[str, pd.DataFrame], symbol: str) -> Optional[Dict[str, float]]:
    frame = candles.get(FUNDING_CHART)
    if frame is None or frame.empty or f"{symbol}_funding_rate" not in frame:
        return None
    return funding(frame[f"{symbol}_funding_rate"], 1440 // interval_minutes(FUNDING_CHART))
