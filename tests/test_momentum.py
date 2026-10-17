"""Momentum statistics: the move's z, the event study and its intervals, persistence tests, the signal rule."""

import math

import numpy as np
import pandas as pd
import pytest

from futurecast.stats.momentum import (
    Horizon,
    MomentumSpec,
    analyze_horizon,
    long_run_variance,
    momentum_signal,
    momentum_view,
    variance_ratio,
)

SYM, BENCH = "B-TEST_USDT", "B-BTC_USDT"
H1 = Horizon("1h", "15m", 4)
H4 = Horizon("4h", "15m", 16)
N = 17_280  # 180 days of 15m candles, as the dashboard keeps


def _candles(returns, bench_returns=None, buy_share=0.5):
    """Aligned 15m candles from one-candle log returns of the coin (and its benchmark)."""
    idx = pd.date_range("2026-01-01", periods=len(returns), freq="15min", tz="UTC", name="timestamp")
    close = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({f"{SYM}_close": close, f"{SYM}_high": close * 1.001, f"{SYM}_low": close * 0.999,
                       f"{SYM}_quote_volume": 1000.0, f"{SYM}_taker_buy_quote_volume": 1000.0 * buy_share}, index=idx)
    if bench_returns is not None:
        df[f"{BENCH}_close"] = 100 * np.exp(np.cumsum(bench_returns))
    return df


def _ar1(phi, sigma, seed, n=N):
    """One-candle returns where each carries `phi` of the previous one: momentum above 0, reversal below."""
    e = np.random.default_rng(seed).normal(0, sigma, n)
    r = np.empty(n)
    r[0] = e[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    return r


def test_overlapping_windows_get_the_right_standard_error():
    # Means of overlapping 16-candle sums of independent returns: the variance of the mean is h^2 / N
    h, n = 16, 200_000
    e = np.random.default_rng(0).normal(size=n + h)
    sums = np.convolve(e, np.ones(h), "valid")[:n]
    se = math.sqrt(long_run_variance(sums - sums.mean(), h - 1)) / n
    assert se == pytest.approx(h / math.sqrt(n), rel=0.05)
    u = np.array([1.0, -2.0, 0.5])
    assert long_run_variance(u, 0) == pytest.approx(u @ u)


def test_variance_ratio_tells_trending_from_reverting_from_random():
    def vr(phi, seed):
        return variance_ratio(np.r_[0, np.cumsum(_ar1(phi, 0.004, seed))], 16)

    trending, reverting, random = vr(0.2, 1), vr(-0.2, 2), vr(0.0, 3)
    # AR(1) theory: VR(q) = 1 + 2 sum_k (1 - k/q) phi^k
    theory = 1 + 2 * sum((1 - k / 16) * 0.2 ** k for k in range(1, 16))
    assert trending["vr"] == pytest.approx(theory, rel=0.1) and trending["p"] < 1e-6
    assert reverting["vr"] < 0.8 and reverting["p"] < 1e-6
    assert random["vr"] == pytest.approx(1.0, abs=0.1) and random["p"] > 0.05


def test_momentum_raises_a_long_continuation():
    r = _ar1(0.5, 0.005, seed=4)
    r[-4:] = 0.01  # a strong last hour: top fifth
    h = analyze_horizon(_candles(r), SYM, H1)

    a = h["after"]
    assert a["bucket"] == 5 and a["z_high"] is None
    assert a["mean_pct"] > MomentumSpec().round_trip_pct and a["ci_low_pct"] > 0
    assert a["older_mean_pct"] > 0 and a["newer_mean_pct"] > 0 and a["hit_rate"] > a["all_hit_rate"]
    assert (h["signal"], h["reason"], h["kind"]) == ("LONG", "edge", "continuation")
    p = h["persistence"]
    assert p["character"] == "trending" and p["ic"] > 0 and p["ic_p"] < 0.01


def test_mean_reversion_raises_a_short_reversal():
    r = _ar1(-0.5, 0.015, seed=5)
    r[-4:] = 0.02
    h = analyze_horizon(_candles(r), SYM, H1)

    assert h["after"]["bucket"] == 5 and h["after"]["ci_high_pct"] < 0
    assert (h["signal"], h["reason"], h["kind"]) == ("SHORT", "edge", "reversal")
    assert h["persistence"]["character"] == "mean_reverting" and h["persistence"]["ic"] < 0


def test_a_random_walk_raises_nothing():
    r = np.random.default_rng(6).normal(0, 0.004, N)
    r[-16:] = 0.004  # even after a strong move
    h = analyze_horizon(_candles(r), SYM, H4)

    assert h["after"]["bucket"] == 5
    assert h["signal"] == "NEUTRAL" and h["reason"] in ("noise", "fees")
    assert h["persistence"]["character"] == "random"


def test_the_move_is_scaled_by_trailing_volatility_only():
    r = np.random.default_rng(7).normal(0, 0.003, 4_000)
    candles = _candles(r)
    k = 3_000
    spec = MomentumSpec()
    window = spec.vol_window(H4)
    h = analyze_horizon(candles.iloc[:k], SYM, H4, spec)

    logp = np.log(candles[f"{SYM}_close"].to_numpy())
    move = logp[k - 1] - logp[k - 1 - 16]
    vol = np.std(np.diff(logp[:k])[-window:], ddof=1)
    assert h["z"] == pytest.approx(move / (vol * 4))
    assert h["return_pct"] == pytest.approx(100 * math.expm1(move))
    assert h["typical_move_pct"] == pytest.approx(100 * vol * 4)


def test_versus_benchmark_takes_out_beta():
    rng = np.random.default_rng(8)
    bench = rng.normal(0, 0.003, 3_000)
    coin = 1.5 * bench + rng.normal(0, 0.0005, 3_000)
    coin[-16:] = 1.5 * bench[-16:]  # the last 4h moved exactly with beta
    vs = analyze_horizon(_candles(coin, bench), SYM, H4, benchmark=BENCH)["versus"]

    assert vs["beta"] == pytest.approx(1.5, abs=0.05)
    assert vs["excess_pct"] == pytest.approx(0.0, abs=0.1)
    assert vs["return_pct"] == pytest.approx(100 * math.expm1(bench[-16:].sum()))


def test_taker_flow_against_its_history():
    candles = _candles(np.random.default_rng(9).normal(0, 0.003, 2_000))
    candles.iloc[-16:, candles.columns.get_loc(f"{SYM}_taker_buy_quote_volume")] = 800.0
    flow = analyze_horizon(candles, SYM, H4)["flow"]
    assert flow["buy_pct"] == pytest.approx(80.0)
    assert flow["percentile"] > 99


def test_signal_rule():
    spec = MomentumSpec()
    edge = {"bucket": 5, "buckets": 5, "windows": 100.0, "mean_pct": 0.5, "ci_low_pct": 0.2, "ci_high_pct": 0.8,
            "older_mean_pct": 0.4, "newer_mean_pct": 0.6}
    assert momentum_signal(edge, spec) == ("LONG", "edge")
    assert momentum_signal(None, spec) == ("NEUTRAL", "few")
    assert momentum_signal({**edge, "bucket": 3}, spec) == ("NEUTRAL", "flat")
    assert momentum_signal({**edge, "windows": 29.0}, spec) == ("NEUTRAL", "few")
    assert momentum_signal({**edge, "ci_low_pct": -0.1}, spec) == ("NEUTRAL", "noise")
    assert momentum_signal({**edge, "mean_pct": 0.15, "ci_low_pct": 0.1, "ci_high_pct": 0.2}, spec) == ("NEUTRAL", "fees")
    assert momentum_signal({**edge, "older_mean_pct": -0.1}, spec) == ("NEUTRAL", "unstable")
    short = {**edge, "bucket": 1, "mean_pct": -0.5, "ci_low_pct": -0.8, "ci_high_pct": -0.2,
             "older_mean_pct": -0.4, "newer_mean_pct": -0.6}
    assert momentum_signal(short, spec) == ("SHORT", "edge")


def test_view_covers_the_charts_on_hand():
    candles = _candles(np.random.default_rng(10).normal(0, 0.003, 3_000))
    view = momentum_view({"15m": candles, "1h": pd.DataFrame()}, SYM)
    assert set(view["horizons"]) == {"1h", "4h"}
    assert view["round_trip_pct"] == pytest.approx(2 * MomentumSpec().taker_bps / 100)
    short = analyze_horizon(candles.iloc[:10], SYM, H4)
    assert (short["signal"], short["reason"]) == ("NEUTRAL", "few") and "z" not in short
