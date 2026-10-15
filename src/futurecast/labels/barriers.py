"""Volatility scale and adverse-excursion labels for the trade-risk model (models.trade_risk).

A position is entered at the close of bar t and held for at most `horizon` bars. Its adverse
excursion is the worst move against it in that window, measured in units of the volatility
known at entry so it stays comparable between a sleepy major and a volatile meme coin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class BarrierSpec:
    """Volatility window and trading cost."""

    vol_window: int = 240  # trailing candles used to estimate volatility (4 hours of 1m candles)
    vol_min_periods: int = 60
    cost_bps: float = 10.0  # assumed round-trip fees + slippage

    @property
    def cost(self) -> float:
        return self.cost_bps / 10_000.0


def horizon_volatility(close: pd.Series, horizon: int, spec: BarrierSpec) -> pd.Series:
    """Expected standard deviation of the log return over `horizon` bars (trailing data only)."""
    log_ret = np.log(close.clip(lower=1e-12)).diff()
    sigma_1m = log_ret.rolling(spec.vol_window, min_periods=spec.vol_min_periods).std()
    return sigma_1m * np.sqrt(horizon)


def adverse_excursion(close: pd.Series, high: pd.Series, low: pd.Series, horizon: int) -> pd.DataFrame:
    """Worst move against a position entered at the close of bar t, over bars t+1 .. t+horizon.

    Columns long_mae / short_mae are fractions of entry (0 when price never went against
    the side); NaN where the window runs past the data.
    """
    n = len(close)
    cols = {name: np.full(n, np.nan) for name in ("long_mae", "short_mae")}
    m = n - horizon

    if m > 0:
        entry = close.to_numpy(dtype=float)[:m]
        fut_low = sliding_window_view(low.to_numpy(dtype=float)[1:], horizon)[:m].min(axis=1)
        fut_high = sliding_window_view(high.to_numpy(dtype=float)[1:], horizon)[:m].max(axis=1)
        known = np.isfinite(entry) & (entry > 0)
        safe = np.where(known, entry, 1.0)
        cols["long_mae"][:m] = np.where(known, np.maximum(0.0, 1.0 - fut_low / safe), np.nan)
        cols["short_mae"][:m] = np.where(known, np.maximum(0.0, fut_high / safe - 1.0), np.nan)

    return pd.DataFrame(cols, index=close.index)
