"""Volatility regimes (hidden Markov model)."""

import numpy as np
import pytest

from futurecast.models.regimes import VolatilityRegimes

SIGMA = np.array([0.0005, 0.0012, 0.004])
TRANS = np.array([[0.995, 0.004, 0.001], [0.006, 0.99, 0.004], [0.004, 0.016, 0.98]])


def _switching_returns(n, seed=0):
    rng = np.random.default_rng(seed)
    state = np.empty(n, dtype=int)
    state[0] = 1
    for t in range(1, n):
        state[t] = rng.choice(3, p=TRANS[state[t - 1]])
    return rng.normal(0, SIGMA[state]), state


def test_scan_filter_matches_a_step_by_step_filter():
    r, _ = _switching_returns(600)
    model = VolatilityRegimes(SIGMA, TRANS)
    b = model._emissions(r)
    alpha = model.stationary * b[0]
    expected = [alpha / alpha.sum()]
    for t in range(1, len(r)):
        alpha = (expected[-1] @ TRANS) * b[t]
        expected.append(alpha / alpha.sum())
    np.testing.assert_allclose(model.filter(r), np.array(expected), atol=1e-12)


def test_filtered_regimes_never_look_ahead():
    r, _ = _switching_returns(1000, seed=1)
    model = VolatilityRegimes(SIGMA, TRANS)
    later_storm = r.copy()
    later_storm[600:] *= 10
    np.testing.assert_allclose(model.filter(r)[:600], model.filter(later_storm)[:600])


def test_fit_recovers_regimes_and_orders_them_calm_to_stormy():
    r, state = _switching_returns(30_000, seed=2)
    model = VolatilityRegimes.fit(r)
    np.testing.assert_allclose(model.sigma, SIGMA, rtol=0.1)
    np.testing.assert_allclose(np.diag(model.trans), np.diag(TRANS), atol=0.01)
    assert np.mean(model.labels(model.filter(r)) == state) > 0.9

    calm, stormy = np.eye(3)[[0, 2]]
    chance = model.storm_chance(np.vstack([calm, stormy]), 60)
    assert 0 < chance[0] < 0.5 and chance[1] == pytest.approx(1.0)

