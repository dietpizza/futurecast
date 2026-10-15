"""Volatility regimes: a hidden Markov model of a coin's candle-to-candle log returns.

Three hidden states, calm / normal / stormy, each emitting zero-mean Gaussian returns with
its own volatility, and a transition matrix that says how long each state tends to last and
where it goes next. Fitted with EM (Baum-Welch) on the oldest REGIME_FIT_FRAC of history only,
which lies inside every model's fit slice, so no calibration or test candle shapes it.

Only *filtered* probabilities are ever used: P(state at t | returns up to t). Smoothed
probabilities and Viterbi paths look at later candles and would leak the future into
features, baselines and calibration.

Uses (all in models.trade_risk):
- labels: the most likely state now (per-regime calibration of the worst-move bound).
- storm_chance: chance of spending at least one candle in the stormy state within a horizon.

The forward and backward passes are run as prefix products of per-candle matrices with a
doubling scan, so a 100k-candle history takes ~17 batched 3x3 matrix products per pass
instead of a Python loop per candle.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

REGIME_NAMES = ("calm", "normal", "stormy")
REGIME_FIT_FRAC = 0.4  # oldest share of history the regimes are fitted on
_MIN_TRANS = 1e-8


def log_returns(close: pd.Series) -> np.ndarray:
    """Candle-to-candle log returns, 0 for the first candle and anything non-finite."""
    r = np.diff(np.log(np.clip(close.to_numpy(dtype=float), 1e-12, None)), prepend=np.nan)
    return np.where(np.isfinite(r), r, 0.0)


def _prefix_products(M: np.ndarray) -> np.ndarray:
    """P[t] proportional to M[0] @ M[1] @ ... @ M[t], each rescaled to a max entry of 1."""
    P = M.copy()
    d = 1
    while d < len(P):
        prod = P[:-d] @ P[d:]
        prod /= np.maximum(prod.max(axis=(1, 2), keepdims=True), 1e-300)
        P[d:] = prod
        d *= 2
    return P


def _normalize_rows(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(a.sum(axis=-1, keepdims=True), 1e-300)


class VolatilityRegimes:
    """Gaussian hidden Markov model of volatility regimes, states ordered calm -> stormy."""

    def __init__(self, sigma: np.ndarray, trans: np.ndarray, fitted_rows: int = 0) -> None:
        self.sigma = np.asarray(sigma, dtype=float)  # per-candle return std of each state
        self.trans = np.asarray(trans, dtype=float)  # trans[i, j] = P(next state j | state i)
        self.fitted_rows = fitted_rows
        self.log_likelihood: Optional[float] = None

    @property
    def n_states(self) -> int:
        return len(self.sigma)

    @property
    def stationary(self) -> np.ndarray:
        """Long-run share of time in each state (the starting belief of every filter)."""
        vals, vecs = np.linalg.eig(self.trans.T)
        v = np.abs(np.real(vecs[:, np.argmin(np.abs(vals - 1.0))]))
        return v / v.sum()

    # ───────── inference ─────────

    def _emissions(self, r: np.ndarray) -> np.ndarray:
        """Per-candle state likelihoods, rescaled so the likeliest state is 1 (T x K)."""
        logp = -0.5 * (r[:, None] / self.sigma) ** 2 - np.log(self.sigma)
        return np.exp(logp - logp.max(axis=1, keepdims=True))

    def _forward(self, b: np.ndarray) -> np.ndarray:
        alpha = np.empty_like(b)
        alpha[0] = _normalize_rows(self.stationary * b[0])
        if len(b) > 1:
            P = _prefix_products(self.trans[None, :, :] * b[1:, None, :])
            alpha[1:] = _normalize_rows(np.einsum("k,tkj->tj", alpha[0], P))
        return alpha

    def _backward(self, b: np.ndarray) -> np.ndarray:
        beta = np.ones_like(b)
        if len(b) > 1:
            M = self.trans[None, :, :] * b[1:, None, :]  # M[s-1] moves from candle s-1 to s
            P = _prefix_products(M[::-1].transpose(0, 2, 1))  # reversed, transposed: suffix products
            beta[:-1] = P.sum(axis=1)[::-1]
            beta /= beta.max(axis=1, keepdims=True)
        return beta

    def filter(self, returns: np.ndarray) -> np.ndarray:
        """P(state at t | returns up to t) for every candle (T x K). Never looks ahead."""
        r = np.asarray(returns, dtype=float)
        if len(r) == 0:
            return np.empty((0, self.n_states))
        return self._forward(self._emissions(r))

    def filter_close(self, close: pd.Series) -> np.ndarray:
        return self.filter(log_returns(close))

    @staticmethod
    def labels(probs: np.ndarray) -> np.ndarray:
        return np.argmax(probs, axis=1)

    def storm_chance(self, probs: np.ndarray, horizon: int) -> np.ndarray:
        """Chance the coin is stormy now or at any candle within the next `horizon`, per row of `probs`."""
        storm = self.n_states - 1
        absorbing = self.trans.copy()  # once stormy, stay stormy: the mass there is "reached by now"
        absorbing[storm] = 0.0
        absorbing[storm, storm] = 1.0
        return (probs @ np.linalg.matrix_power(absorbing, horizon))[:, storm]

    # ───────── fitting ─────────

    @classmethod
    def fit(cls, returns: np.ndarray, n_states: int = 3, max_iter: int = 60, tol: float = 1e-7) -> "VolatilityRegimes":
        """Baum-Welch on `returns`; states come out sorted from calm to stormy."""
        r = np.asarray(returns, dtype=float)
        var_floor = max(float(np.var(r)) * 1e-4, 1e-16)
        # Start from volatility quantiles of a short rolling window, sticky transitions
        rolling = pd.Series(r).abs().rolling(30, min_periods=1).mean().to_numpy()
        edges = np.quantile(rolling, np.linspace(0, 1, n_states + 1))
        bucket = np.clip(np.searchsorted(edges, rolling, side="right") - 1, 0, n_states - 1)
        sigma = np.array([np.sqrt(max(np.mean(r[bucket == k] ** 2) if np.any(bucket == k) else np.var(r), var_floor))
                          for k in range(n_states)])
        trans = np.full((n_states, n_states), 0.03 / max(n_states - 1, 1))
        np.fill_diagonal(trans, 0.97)
        model = cls(np.sort(sigma), trans, fitted_rows=len(r))

        prev_ll = -np.inf
        for _ in range(max_iter):
            b = model._emissions(r)
            alpha, beta = model._forward(b), model._backward(b)
            gamma = _normalize_rows(alpha * beta)
            xi = alpha[:-1, :, None] * model.trans[None] * (b[1:] * beta[1:])[:, None, :]
            xi /= np.maximum(xi.sum(axis=(1, 2), keepdims=True), 1e-300)

            # Log-likelihood from the one-step predictions (emissions were rescaled per candle)
            pred = np.vstack([model.stationary, alpha[:-1] @ model.trans])
            logp = -0.5 * (r[:, None] / model.sigma) ** 2 - np.log(model.sigma)
            ll = float(np.sum(np.log(np.maximum((pred * b).sum(axis=1), 1e-300)) + logp.max(axis=1)))
            model.log_likelihood = ll

            trans = xi.sum(axis=0)
            trans = np.maximum(trans / np.maximum(trans.sum(axis=1, keepdims=True), 1e-300), _MIN_TRANS)
            model.trans = _normalize_rows(trans)
            model.sigma = np.sqrt(np.maximum((gamma * r[:, None] ** 2).sum(axis=0) / np.maximum(gamma.sum(axis=0), 1e-12),
                                             var_floor))
            if ll - prev_ll < tol * abs(ll):
                break
            prev_ll = ll

        order = np.argsort(model.sigma)
        model.sigma = model.sigma[order]
        model.trans = model.trans[np.ix_(order, order)]
        return model

    @classmethod
    def fit_on_history(cls, close: pd.Series) -> "VolatilityRegimes":
        """Fit on the oldest REGIME_FIT_FRAC of `close` only."""
        r = log_returns(close)
        return cls.fit(r[: max(int(len(r) * REGIME_FIT_FRAC), 2)])

    def summary(self) -> dict:
        names = REGIME_NAMES if self.n_states == len(REGIME_NAMES) else tuple(f"state_{k}" for k in range(self.n_states))
        stay = np.diag(self.trans)
        return {
            name: {"sigma_pct": round(float(s) * 100, 5), "share": round(float(p), 4),
                   "mean_run_candles": round(float(1.0 / max(1.0 - st, 1e-9)), 1)}
            for name, s, p, st in zip(names, self.sigma, self.stationary, stay)
        }


def regime_name(label: int, n_states: int = 3) -> str:
    return REGIME_NAMES[label] if n_states == len(REGIME_NAMES) else f"state_{label}"
