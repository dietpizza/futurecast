"""Momentum and market state per trading window, from closed candles. Rules, no model: they
describe what has happened, not what comes next.

A window (WINDOWS) is h candles of its chart: 5m to 1h on 5m candles, 2h to 4h on 15m, 8h on 1h.
Each reading looks back over the last `lookback` windows, L = lookback x h candles (at least
`min_candles`). With r the one-candle log returns and σ their standard deviation over the trailing
`vol_days`:

- Momentum, -100..+100: the mean of two z-like readings, divided by `momentum_scale` and clipped:
  the log return over L candles in units of σ√L, and the slope of an EMA spanning L candles over
  the last h candles in units of σ√h. Bullish above +`bullish`, bearish below -`bullish`.
- Efficiency (Kaufman's ratio): |net move over L| / Σ|r| over L, 0 (all noise) to 1 (a straight line).
- Volatility percentile: the standard deviation of r over L candles, ranked against the same
  reading over the trailing `rank_days`. Labelled calm / normal / high / wild.
- State: wild if the volatility percentile is ≥ `wild_pct`; else trending if efficiency ≥ the
  window's threshold and |momentum| ≥ `bullish`; else quiet if the percentile is ≤ `quiet_pct`;
  else choppy. A random walk's efficiency falls like 1/√L, so the threshold is `trend_efficiency`
  at `min_candles` and scales with √(min_candles / L): noise passes equally often in every window.

Every column at a candle uses that candle's close and earlier ones only. Positions count candles,
not clock time; gaps the aligner dropped are stepped over.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from deltarisk.features.setups import interval_minutes


@dataclass(frozen=True)
class Window:
    """A trading window of `candles` candles of the `interval` chart (1h = 12 candles of 5m)."""

    name: str
    interval: str
    candles: int

    @property
    def minutes(self) -> int:
        return interval_minutes(self.interval) * self.candles


WINDOWS: Tuple[Window, ...] = (
    Window("5m", "5m", 1), Window("15m", "5m", 3), Window("30m", "5m", 6), Window("1h", "5m", 12),
    Window("2h", "15m", 8), Window("3h", "15m", 12), Window("4h", "15m", 16), Window("8h", "1h", 8),
)
WINDOW_BY_NAME = {w.name: w for w in WINDOWS}


@dataclass(frozen=True)
class RegimeSpec:
    lookback: int = 3  # windows looked back over
    min_candles: int = 12
    vol_days: float = 14.0  # σ that scales moves
    rank_days: float = 30.0  # volatility percentile against this much history
    momentum_scale: float = 2.0  # mean z at which momentum reaches ±100
    bullish: float = 25.0
    trend_efficiency: float = 0.35  # at min_candles; see trend_threshold
    wild_pct: float = 90.0
    high_pct: float = 75.0
    quiet_pct: float = 25.0

    def span(self, window: Window) -> int:
        return max(self.lookback * window.candles, self.min_candles)

    def trend_threshold(self, window: Window) -> float:
        return self.trend_efficiency * math.sqrt(self.min_candles / self.span(window))

    def candles_in(self, days: float, window: Window) -> int:
        return int(round(days * 1440 / interval_minutes(window.interval)))


def _log_close(frame: pd.DataFrame, symbol: str) -> pd.Series:
    return np.log(frame[f"{symbol}_close"].astype(float))


def regime_frame(frame: pd.DataFrame, symbol: str, window: Window, spec: RegimeSpec = RegimeSpec()) -> pd.DataFrame:
    """Momentum, efficiency, volatility percentile and their labels at every candle of the chart."""
    p = _log_close(frame, symbol)
    r = p.diff()
    h, L = window.candles, spec.span(window)
    vol_n = spec.candles_in(spec.vol_days, window)
    sigma = r.rolling(vol_n, min_periods=max(vol_n // 2, 2 * L)).std()

    z_return = (p - p.shift(L)) / (sigma * math.sqrt(L))
    ema = p.ewm(span=L, adjust=False).mean()
    z_slope = (ema - ema.shift(h)) / (sigma * math.sqrt(h))
    momentum = (100.0 * ((z_return + z_slope) / 2.0 / spec.momentum_scale)).clip(-100.0, 100.0)

    path = r.abs().rolling(L).sum()
    efficiency = ((p - p.shift(L)).abs() / path).where(path > 0, 0.0)

    vol = r.rolling(L).std()
    rank_n = spec.candles_in(spec.rank_days, window)
    vol_pct = 100.0 * vol.rolling(rank_n, min_periods=max(rank_n // 4, 2 * L)).rank(pct=True)

    out = pd.DataFrame({"momentum": momentum, "efficiency": efficiency, "vol_pct": vol_pct}, index=frame.index)
    out["momentum_label"] = momentum_label(out["momentum"], spec)
    out["volatility_label"] = volatility_label(out["vol_pct"], spec)
    out["state"] = market_state(out, spec.trend_threshold(window), spec)
    return out


def momentum_label(momentum: pd.Series, spec: RegimeSpec = RegimeSpec()) -> pd.Series:
    labels = np.where(momentum > spec.bullish, "bullish", np.where(momentum < -spec.bullish, "bearish", "neutral"))
    return pd.Series(labels, index=momentum.index).where(momentum.notna())


def volatility_label(vol_pct: pd.Series, spec: RegimeSpec = RegimeSpec()) -> pd.Series:
    labels = np.select([vol_pct >= spec.wild_pct, vol_pct >= spec.high_pct, vol_pct <= spec.quiet_pct],
                       ["wild", "high", "calm"], "normal")
    return pd.Series(labels, index=vol_pct.index).where(vol_pct.notna())


def market_state(regime: pd.DataFrame, trend_threshold: float, spec: RegimeSpec = RegimeSpec()) -> pd.Series:
    m, eff, pct = regime["momentum"], regime["efficiency"], regime["vol_pct"]
    labels = np.select(
        [pct >= spec.wild_pct, (eff >= trend_threshold) & (m.abs() >= spec.bullish), pct <= spec.quiet_pct],
        ["wild", "trending", "quiet"], "choppy")
    return pd.Series(labels, index=regime.index).where(m.notna() & eff.notna() & pct.notna())


def trend_record(frame: pd.DataFrame, symbol: str, window: Window, regime: pd.DataFrame,
                 round_trip_pct: float) -> Dict[str, Any]:
    """How the next window went after past trending readings, per side, against every window: the
    in-sample record behind the page's Long / Short call (the call adds a risk check this ignores).

    Long: candles labelled trending and bullish; short: trending and bearish. For each, `n`
    readings with a known next window (overlapping: about n / h independent ones), `win_rate`
    (share where the next h candles closed the call's way) and `mean_pct` (mean simple return in
    the call's direction, before fees, against `round_trip_pct`). `base` is the same over every
    reading: the coin's plain drift."""
    close = frame[f"{symbol}_close"].astype(float)
    fwd = (close.shift(-window.candles) / close - 1.0).to_numpy()
    known = np.isfinite(fwd) & regime["state"].notna().to_numpy()
    trending = (regime["state"] == "trending").to_numpy()
    label = regime["momentum_label"].to_numpy()

    def record(mask: np.ndarray, sign: float) -> Dict[str, Any]:
        moves = sign * fwd[mask & known]
        if not len(moves):
            return {"n": 0, "win_rate": None, "mean_pct": None}
        return {"n": int(len(moves)), "win_rate": float(np.mean(moves > 0)), "mean_pct": 100.0 * float(np.mean(moves))}

    return {
        "long": record(trending & (label == "bullish"), 1.0),
        "short": record(trending & (label == "bearish"), -1.0),
        "base": record(np.ones(len(fwd), dtype=bool), 1.0),
        "round_trip_pct": round_trip_pct,
    }


def regime_view(candles: Dict[str, pd.DataFrame], symbol: str, spec: RegimeSpec = RegimeSpec()) -> Dict[str, Any]:
    """The latest reading per window, None where its chart is missing or too short."""
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for w in WINDOWS:
        frame = candles.get(w.interval)
        if frame is None or frame.empty:
            out[w.name] = None
            continue
        last = regime_frame(frame, symbol, w, spec).iloc[-1]
        if pd.isna(last["state"]):
            out[w.name] = None
            continue
        out[w.name] = {
            "momentum": float(last["momentum"]), "momentum_label": last["momentum_label"],
            "efficiency": float(last["efficiency"]), "vol_pct": float(last["vol_pct"]),
            "volatility": last["volatility_label"], "state": last["state"],
        }
    return out
