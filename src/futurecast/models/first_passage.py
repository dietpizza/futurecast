"""Volatility-only odds of hitting the take-profit before the stop: an absorbing Markov chain.

Price is a continuous-time Markov chain on a lattice between the stop (-b) and the
take-profit (+a), stepping up or down with equal rates and no drift; the two barriers are
absorbing. With distances measured in units of the expected horizon volatility, the chain's
probability of being absorbed at +a within the horizon is

    P(win) = expm(Q)[start, take-profit]

where Q is the lattice generator scaled so one horizon has unit variance. It depends only on
x = a / horizon_vol and the stop / take-profit ratio, so a table per ratio is built once
(about 150 matrix exponentials) and interpolated. As the lattice gets finer this converges to
Brownian motion touching a barrier, which is what the high / low based win labels measure.

This is the win probability a trader with no view on direction would quote, knowing only
how volatile the coin is about to be (models.regimes forecasts that).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.linalg import expm

_CELLS_PER_TP = 40  # lattice cells between entry and take-profit
_X_MIN, _X_MAX, _X_POINTS = 0.05, 8.0, 160


def _absorb_at_top(x: float, ratio: float) -> float:
    """P(chain from 0 is absorbed at +x before -ratio*x within unit variance)."""
    up = _CELLS_PER_TP
    down = int(round(ratio * _CELLS_PER_TP))
    n = up + down + 1  # lattice nodes from -b to +a; the two ends absorb
    delta = x / _CELLS_PER_TP
    rate = 0.5 / delta ** 2  # each direction: total variance rate 1 per unit time
    Q = np.zeros((n, n))
    idx = np.arange(1, n - 1)
    Q[idx, idx - 1] = rate
    Q[idx, idx + 1] = rate
    Q[idx, idx] = -2 * rate
    return float(expm(Q)[down, n - 1])


@lru_cache(maxsize=None)
def _table(ratio: float) -> np.ndarray:
    xs = np.geomspace(_X_MIN, _X_MAX, _X_POINTS)
    return np.array([_absorb_at_top(x, ratio) for x in xs])


def win_probability(x, ratio: float) -> np.ndarray:
    """No-direction odds that price touches +x before -ratio*x within the horizon.

    `x` is the take-profit distance in units of the horizon volatility (array or scalar);
    `ratio` is stop / take-profit distance. Below the table, both barriers are certain to be
    reached, so the odds are the gambler's-ruin ratio / (1 + ratio); far above it they are 0.
    """
    ratio = round(float(ratio), 6)
    x = np.asarray(x, dtype=float)
    table = _table(ratio)
    log_xs = np.log(np.geomspace(_X_MIN, _X_MAX, _X_POINTS))
    out = np.interp(np.log(np.clip(x, _X_MIN, _X_MAX)), log_xs, table)
    out = np.where(x <= _X_MIN, ratio / (1.0 + ratio), out)
    out = np.where(x >= _X_MAX, 0.0, out)
    return np.where(np.isfinite(x), out, np.nan)
