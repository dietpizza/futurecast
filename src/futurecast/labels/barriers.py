"""Take-profit / stop-loss trade outcome labels.

A trade is entered at the close of bar t and held for at most `horizon` bars:

- A long wins if price touches entry * (1 + tp) before entry * (1 - sl).
- A short wins if price touches entry * (1 - tp) before entry * (1 + sl).
- Touching neither before the horizon ends (timeout) is not a win.
- Touching both inside the same 1-minute bar is ambiguous; it counts as a loss.

Barrier distances scale with recent volatility, so a "win" means a comparable
move for a sleepy major and a volatile meme coin, and they are floored so a
take-profit always clears round-trip trading costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class BarrierSpec:
    """Where the take-profit and stop-loss sit relative to entry."""

    tp_vol_mult: float = 0.5  # take-profit distance in units of expected horizon volatility
    sl_vol_mult: float = 0.5  # stop-loss distance in the same units
    vol_window: int = 240  # trailing 1m bars used to estimate volatility
    vol_min_periods: int = 60
    cost_bps: float = 10.0  # assumed round-trip fees + slippage
    min_tp_cost_mult: float = 2.0  # take-profit is never closer than this many round-trip costs

    @property
    def cost(self) -> float:
        return self.cost_bps / 10_000.0


def horizon_volatility(close: pd.Series, horizon: int, spec: BarrierSpec) -> pd.Series:
    """Expected standard deviation of the log return over `horizon` bars (trailing data only)."""
    log_ret = np.log(close.clip(lower=1e-12)).diff()
    sigma_1m = log_ret.rolling(spec.vol_window, min_periods=spec.vol_min_periods).std()
    return sigma_1m * np.sqrt(horizon)


def barrier_distances(sigma_h, spec: BarrierSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Take-profit and stop-loss distances as fractions of the entry price.

    The stop keeps the configured reward:risk ratio when the take-profit is floored.
    """
    sigma = np.asarray(sigma_h, dtype=float)
    tp = np.maximum(spec.tp_vol_mult * sigma, spec.min_tp_cost_mult * spec.cost)
    sl = tp * (spec.sl_vol_mult / spec.tp_vol_mult)
    return tp, sl


def break_even_probability(tp, sl, cost: float, loss_ratio: float = 1.0):
    """Win rate at which expected profit after costs is zero.

    A win earns tp - cost. A non-win loses loss_ratio * sl + cost on average, where
    loss_ratio is measured from history: 1.0 means every non-win hit the stop, lower
    values mean some trades timed out closer to entry.
    """
    tp = np.asarray(tp, dtype=float)
    loss = loss_ratio * np.asarray(sl, dtype=float)
    return (loss + cost) / (tp + loss)


def _first_touch(mask: np.ndarray) -> np.ndarray:
    """Index of the first True per row, or the row length when there is none."""
    first = mask.argmax(axis=1)
    first[~mask.any(axis=1)] = mask.shape[1]
    return first


def first_touch_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    tp,
    sl,
    horizon: int,
) -> pd.DataFrame:
    """Long/short outcomes per entry bar; NaN where the outcome is not yet known.

    Columns: long_win / short_win (1.0 or 0.0) and long_pnl / short_pnl, the gross
    return as a fraction of entry: +tp on a win, -sl when stopped (or ambiguous),
    and the move to the horizon's close on a timeout.
    `tp` and `sl` are per-bar distances (fractions of entry) known at entry time.
    """
    n = len(close)
    cols = {name: np.full(n, np.nan) for name in ("long_win", "short_win", "long_pnl", "short_pnl")}
    m = n - horizon

    if m > 0:
        closes = close.to_numpy(dtype=float)
        entry = closes[:m]
        exit_close = closes[horizon:horizon + m]
        tp_m = np.asarray(tp, dtype=float)[:m]
        sl_m = np.asarray(sl, dtype=float)[:m]

        # Row t holds bars t+1 .. t+horizon (views, no copies)
        fut_high = sliding_window_view(high.to_numpy(dtype=float)[1:], horizon)[:m]
        fut_low = sliding_window_view(low.to_numpy(dtype=float)[1:], horizon)[:m]

        long_tp = _first_touch(fut_high >= (entry * (1.0 + tp_m))[:, None])
        long_sl = _first_touch(fut_low <= (entry * (1.0 - sl_m))[:, None])
        short_tp = _first_touch(fut_low <= (entry * (1.0 - tp_m))[:, None])
        short_sl = _first_touch(fut_high >= (entry * (1.0 + sl_m))[:, None])

        drift = exit_close / entry - 1.0
        known = np.isfinite(entry) & (entry > 0) & np.isfinite(tp_m) & np.isfinite(sl_m)

        for side, tp_bar, sl_bar, timeout_pnl in (
            ("long", long_tp, long_sl, drift),
            ("short", short_tp, short_sl, -drift),
        ):
            win = tp_bar < sl_bar
            stopped = (sl_bar <= tp_bar) & (sl_bar < horizon)
            pnl = np.where(win, tp_m, np.where(stopped, -sl_m, timeout_pnl))
            cols[f"{side}_win"][:m] = np.where(known, win.astype(float), np.nan)
            cols[f"{side}_pnl"][:m] = np.where(known, pnl, np.nan)

    return pd.DataFrame(cols, index=close.index)
