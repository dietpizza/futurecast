"""Momentum statistics on closed candles: how strong the latest move is, what followed moves
like it on this coin, and whether the coin's price trends or mean-reverts. No fitted models.

Each horizon (MomentumSpec.horizons: 1h and 4h on 15m candles, 1d and 3d on 1h candles) is h
candles of its chart. With log price p and σ_t the standard deviation of one-candle log
returns over the trailing `vol_days` (at least 5 horizons):

- Move: z_t = (p_t - p_{t-h}) / (σ_t √h), the last h candles' return in units of a typical
  h-candle move. Under a random walk z is roughly standard normal; `percentile` ranks the
  latest z against every earlier one on the chart.
- After similar moves (an event study): every past z is put in a quintile; for the quintile the
  latest move falls in, the simple return over the following h candles: count, mean, median,
  share up, and a 95% interval of the mean. Windows h candles long overlap, so the interval sums
  the autocovariances of the first h - 1 lags (Hansen-Hodrick standard errors, long_run_variance).
  The mean is also given for the older and the newer half of the history, and over every window
  (the drift the quintile is compared to).
- Persistence: the information coefficient (correlation of z with the next h candles'
  volatility-scaled return, with the same overlap-corrected t of its slope) and the Lo-MacKinlay variance ratio
  VR(h) with its heteroskedasticity-robust z: above 1 returns keep their direction within the
  horizon (trending), below 1 they reverse (mean-reverting).
- Signal: LONG or SHORT only when the latest move is outside the middle quintile and, after
  moves in its quintile, the next h candles averaged more than a market round trip's fees
  (2 x taker) in one direction, with the 95% interval excluding zero, both halves of the
  history agreeing in sign, and at least `min_windows` non-overlapping windows behind it.
  Otherwise NEUTRAL, with `reason` saying which check failed.

Positions count candles, not clock time; gaps the aligner dropped are stepped over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from futurecast.features.setups import SetupSpec, interval_minutes

LONG, SHORT, NEUTRAL = "LONG", "SHORT", "NEUTRAL"
Z95 = 1.959963984540054  # two-sided 95% normal quantile
SIGNIFICANT = 0.05


@dataclass(frozen=True)
class Horizon:
    """A window of `candles` candles of the `interval` chart (4h = 16 candles of 15m)."""

    name: str
    interval: str
    candles: int


@dataclass(frozen=True)
class MomentumSpec:
    horizons: Tuple[Horizon, ...] = field(default=(
        Horizon("1h", "15m", 4), Horizon("4h", "15m", 16), Horizon("1d", "1h", 24), Horizon("3d", "1h", 72),
    ))
    vol_days: float = 14.0  # trailing window of the volatility that scales moves
    quantiles: int = 5  # buckets of past z for the event study
    min_windows: int = 30  # non-overlapping windows (count / h) a quintile needs to raise a signal
    taker_bps: float = SetupSpec.taker_bps  # per leg: a momentum trade enters and exits at market

    @property
    def round_trip_pct(self) -> float:
        return 2.0 * self.taker_bps / 100.0

    def vol_window(self, horizon: Horizon) -> int:
        return max(int(round(self.vol_days * 1440 / interval_minutes(horizon.interval))), 5 * horizon.candles)


# ───────── statistics ─────────

def p_value(z: float) -> float:
    """Two-sided p-value of a standard normal statistic."""
    return math.erfc(abs(z) / math.sqrt(2.0)) if np.isfinite(z) else float("nan")


def long_run_variance(u: np.ndarray, lag: int) -> float:
    """Variance of Σu for a mean-zero series whose terms up to `lag` apart are correlated, as
    when windows overlap (windows h candles long: lag h - 1). Autocovariances are summed with
    equal weights (Hansen-Hodrick), which is unbiased for that overlap, where Newey-West's
    Bartlett weights understate it by about a third; Newey-West is the fallback when the sum
    comes out non-positive."""
    lag = min(lag, len(u) - 1)
    s0 = float(u @ u)
    cov = [float(u[k:] @ u[:-k]) for k in range(1, lag + 1)]
    hansen_hodrick = s0 + 2.0 * sum(cov)
    if hansen_hodrick > 0:
        return hansen_hodrick
    return s0 + 2.0 * sum((1.0 - k / (lag + 1.0)) * c for k, c in enumerate(cov, start=1))


def lagged(x: np.ndarray, h: int) -> np.ndarray:
    """x[t - h] at position t, NaN for the first h."""
    out = np.full(len(x), np.nan)
    if 0 < h < len(x):
        out[h:] = x[:-h]
    return out


def lead(x: np.ndarray, h: int) -> np.ndarray:
    """x[t + h] at position t, NaN for the last h."""
    out = np.full(len(x), np.nan)
    if 0 < h < len(x):
        out[:-h] = x[h:]
    return out


def rolling_vol(log_price: np.ndarray, window: int) -> np.ndarray:
    """Standard deviation of one-candle log returns over the trailing `window` candles (half
    of it at the start of the history)."""
    returns = pd.Series(np.diff(log_price, prepend=np.nan))
    return returns.rolling(window, min_periods=max(window // 2, 2)).std().to_numpy()


def percentile_of(history: np.ndarray, value: float) -> Optional[float]:
    """Share of `history` below `value`, ties counted half, in percent."""
    history = history[np.isfinite(history)]
    if not len(history) or not np.isfinite(value):
        return None
    return float(100.0 * (np.sum(history < value) + 0.5 * np.sum(history == value)) / len(history))


def variance_ratio(log_price: np.ndarray, q: int) -> Dict[str, float]:
    """Lo-MacKinlay variance ratio of overlapping q-candle returns and its heteroskedasticity-
    robust z (Lo & MacKinlay 1988, z*): VR = Var(q-candle return) / (q Var(one-candle return))."""
    x = log_price[np.isfinite(log_price)]
    n = len(x) - 1
    if q < 2 or n < 4 * q:
        return {"vr": float("nan"), "z": float("nan"), "p": float("nan")}
    r = np.diff(x)
    mu = r.mean()
    d = r - mu
    var1 = float(d @ d) / (n - 1)
    rq = x[q:] - x[:-q] - q * mu
    m = q * (n - q + 1) * (1.0 - q / n)
    vr = float(rq @ rq) / m / var1
    a = d * d
    total = float(a.sum()) ** 2
    theta = sum((2.0 * (q - j) / q) ** 2 * n * float(a[j:] @ a[:-j]) / total for j in range(1, q))
    z = math.sqrt(n) * (vr - 1.0) / math.sqrt(theta) if theta > 0 else float("nan")
    return {"vr": vr, "z": z, "p": p_value(z)}


def information_coefficient(x: np.ndarray, y: np.ndarray, lag: int) -> Dict[str, float]:
    """Correlation of x with y, and the t of the slope of y on x with errors correlated up to
    `lag` apart (long_run_variance)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10 * max(lag, 1):
        return {"ic": float("nan"), "t": float("nan"), "p": float("nan"), "n": int(len(x))}
    xc, yc = x - x.mean(), y - y.mean()
    sxx, syy = float(xc @ xc), float(yc @ yc)
    if sxx <= 0 or syy <= 0:
        return {"ic": float("nan"), "t": float("nan"), "p": float("nan"), "n": int(len(x))}
    beta = float(xc @ yc) / sxx
    var_beta = long_run_variance(xc * (yc - beta * xc), lag) / sxx ** 2
    t = beta / math.sqrt(var_beta) if var_beta > 0 else float("nan")
    return {"ic": float(xc @ yc) / math.sqrt(sxx * syy), "t": t, "p": p_value(t), "n": int(len(x))}


def event_study(z: np.ndarray, forward: np.ndarray, h: int, now: float, quantiles: int = 5) -> Optional[Dict[str, Any]]:
    """What followed past moves in the same quantile of z as `now`.

    `forward[t]` is the simple return over the h candles after t. Returns None when z or the
    history is missing. Means and the interval are in percent.
    """
    ok = np.isfinite(z) & np.isfinite(forward)
    zs, ys = z[ok], forward[ok]
    if not np.isfinite(now) or len(ys) < 10 * quantiles:
        return None
    edges = np.quantile(zs, np.linspace(0.0, 1.0, quantiles + 1)[1:-1])
    bucket = np.searchsorted(edges, zs, side="right")
    b = int(np.searchsorted(edges, now, side="right"))
    inside = bucket == b
    n = int(inside.sum())
    if n < 2:
        return None
    sample = ys[inside]
    mean = float(sample.mean())
    se = math.sqrt(long_run_variance(np.where(inside, ys - mean, 0.0), h - 1)) / n
    t = mean / se if se > 0 else float("nan")
    half = len(ys) // 2
    older, newer = ys[:half][inside[:half]], ys[half:][inside[half:]]
    return {
        "bucket": b + 1, "buckets": quantiles,
        "z_low": float(edges[b - 1]) if b > 0 else None,
        "z_high": float(edges[b]) if b < len(edges) else None,
        "n": n, "windows": n / h,
        "mean_pct": 100.0 * mean, "median_pct": 100.0 * float(np.median(sample)),
        "hit_rate": float(np.mean(sample > 0)),
        "ci_low_pct": 100.0 * (mean - Z95 * se), "ci_high_pct": 100.0 * (mean + Z95 * se),
        "t": t, "p": p_value(t),
        "older_mean_pct": 100.0 * float(older.mean()) if len(older) else None,
        "newer_mean_pct": 100.0 * float(newer.mean()) if len(newer) else None,
        "all_mean_pct": 100.0 * float(ys.mean()), "all_hit_rate": float(np.mean(ys > 0)),
        "all_n": int(len(ys)),
    }


def momentum_signal(after: Optional[Dict[str, Any]], spec: MomentumSpec) -> Tuple[str, str]:
    """(signal, reason) from an event study; reason is one of flat, few, noise, fees, unstable, edge."""
    if after is None:
        return NEUTRAL, "few"
    if after["buckets"] % 2 and after["bucket"] == (after["buckets"] + 1) // 2:
        return NEUTRAL, "flat"
    if after["windows"] < spec.min_windows:
        return NEUTRAL, "few"
    mean = after["mean_pct"]
    side = LONG if mean > 0 else SHORT
    sign = 1.0 if side == LONG else -1.0
    if not (after["ci_low_pct"] > 0 if side == LONG else after["ci_high_pct"] < 0):
        return NEUTRAL, "noise"
    if abs(mean) <= spec.round_trip_pct:
        return NEUTRAL, "fees"
    halves = (after["older_mean_pct"], after["newer_mean_pct"])
    if not all(v is not None and sign * v > 0 for v in halves):
        return NEUTRAL, "unstable"
    return side, "edge"


def character(vr: Dict[str, float]) -> Optional[str]:
    """trending, mean_reverting or random, from a variance ratio test at 5%."""
    if not np.isfinite(vr["vr"]):
        return None
    if vr["p"] < SIGNIFICANT:
        return "trending" if vr["vr"] > 1 else "mean_reverting"
    return "random"


# ───────── one horizon ─────────

def _window_sum(x: np.ndarray, h: int) -> np.ndarray:
    """Sum of the h values ending at each position (NaN for the first h - 1)."""
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    return np.concatenate([np.full(h - 1, np.nan), c[h:] - c[:-h]])


def _flow(candles: pd.DataFrame, symbol: str, h: int) -> Optional[Dict[str, float]]:
    """Share of the last h candles' traded value bought by takers (market buys), against history."""
    buy_col, all_col = f"{symbol}_taker_buy_quote_volume", f"{symbol}_quote_volume"
    if buy_col not in candles or all_col not in candles or len(candles) < h:
        return None
    buys = _window_sum(candles[buy_col].to_numpy(dtype=float), h)
    total = _window_sum(candles[all_col].to_numpy(dtype=float), h)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total > 0, buys / total, np.nan)
    if not np.isfinite(share[-1]):
        return None
    return {"buy_pct": 100.0 * float(share[-1]), "percentile": percentile_of(share[:-1], share[-1])}


def _versus(log_price: np.ndarray, candles: pd.DataFrame, benchmark: Optional[str], h: int,
            window: int) -> Optional[Dict[str, Any]]:
    """The coin's last h candles against its benchmark's: beta from one-candle returns over the
    trailing `window`, and the coin's return beyond beta times the benchmark's."""
    col = f"{benchmark}_close" if benchmark else None
    if col is None or col not in candles or len(log_price) <= h:
        return None
    bench = np.log(candles[col].to_numpy(dtype=float))
    r, rb = np.diff(log_price)[-window:], np.diff(bench)[-window:]
    ok = np.isfinite(r) & np.isfinite(rb)
    if ok.sum() < 20 or np.var(rb[ok]) <= 0:
        return None
    beta = float(np.cov(r[ok], rb[ok])[0, 1] / np.var(rb[ok], ddof=1))
    coin_move, bench_move = log_price[-1] - log_price[-1 - h], bench[-1] - bench[-1 - h]
    return {
        "symbol": benchmark, "beta": beta,
        "return_pct": 100.0 * math.expm1(bench_move),
        "excess_pct": 100.0 * math.expm1(coin_move - beta * bench_move),
    }


def analyze_horizon(candles: pd.DataFrame, symbol: str, horizon: Horizon, spec: MomentumSpec | None = None,
                    benchmark: Optional[str] = None) -> Dict[str, Any]:
    """Every statistic of one horizon at the latest closed candle of `candles` (an aligned frame
    with `{symbol}_close` columns, indexed by candle open time)."""
    spec = spec or MomentumSpec()
    h = horizon.candles
    minutes = interval_minutes(horizon.interval)
    out: Dict[str, Any] = {"interval": horizon.interval, "candles": h, "history_candles": int(len(candles))}
    if len(candles) <= h + 2:
        return {**out, "signal": NEUTRAL, "reason": "few"}

    out["as_of"] = (candles.index[-1] + pd.Timedelta(minutes=minutes)).isoformat()
    out["history_from"] = candles.index[0].isoformat()
    log_price = np.log(candles[f"{symbol}_close"].to_numpy(dtype=float))
    window = spec.vol_window(horizon)
    vol = rolling_vol(log_price, window)
    move = log_price - lagged(log_price, h)
    scale = vol * math.sqrt(h)
    z = move / scale
    now = float(z[-1])
    forward_log = lead(log_price, h) - log_price

    after = event_study(z, np.expm1(forward_log), h, now, spec.quantiles)
    signal, reason = momentum_signal(after, spec)
    vr = variance_ratio(log_price, h)
    ic = information_coefficient(z, forward_log / scale, h - 1)
    out.update({
        "return_pct": 100.0 * math.expm1(move[-1]) if np.isfinite(move[-1]) else None,
        "z": now if np.isfinite(now) else None,
        "percentile": percentile_of(z[:-1], now),
        "typical_move_pct": 100.0 * float(vol[-1]) * math.sqrt(h) if np.isfinite(vol[-1]) else None,
        "flow": _flow(candles, symbol, h),
        "versus": _versus(log_price, candles, benchmark, h, window),
        "after": after,
        "persistence": {"ic": ic["ic"], "ic_t": ic["t"], "ic_p": ic["p"],
                        "vr": vr["vr"], "vr_z": vr["z"], "vr_p": vr["p"], "character": character(vr)},
        "signal": signal, "reason": reason,
        "kind": None if signal == NEUTRAL else ("continuation" if (now > 0) == (signal == LONG) else "reversal"),
    })
    return out


def momentum_view(candles: Dict[str, pd.DataFrame], symbol: str, benchmark: Optional[str] = None,
                  spec: MomentumSpec | None = None) -> Dict[str, Any]:
    """Every horizon of `spec` from the charts in `candles` (interval -> aligned frame)."""
    spec = spec or MomentumSpec()
    horizons = {}
    for hz in spec.horizons:
        frame = candles.get(hz.interval)
        if frame is not None and not frame.empty:
            horizons[hz.name] = analyze_horizon(frame, symbol, hz, spec, benchmark)
    return {
        "horizons": horizons,
        "round_trip_pct": spec.round_trip_pct,
        "quantiles": spec.quantiles,
        "min_windows": spec.min_windows,
        "vol_days": spec.vol_days,
    }
