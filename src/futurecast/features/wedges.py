"""Wedge breakouts with a retest, found by exact rules on one chart (no models).

A long setup (shorts mirror it):

1. Wedge: the latest two swing highs and the latest two swing lows (structure.swing_points,
   `swing_len` candles on each side, so each is known `swing_len` candles after it) give a
   resistance line through the highs and a support line through the lows. They make a wedge
   when the lines converge (the resistance slope is below the support slope) toward an apex
   `apex_min` to `apex_max` candles ahead, the first of the four pivots is at most `window`
   candles back, no candle has closed outside either line since that pivot, and each line has
   been tested at least `min_touches` times (runs of candles whose high, or low, came within
   `touch_tolerance` of it; the pivots count). Both lines rising is a rising wedge, both falling
   a falling wedge, anything else a triangle. Only pivots known by the previous candle's close
   count, so a wedge is complete before the candle that breaks it.
2. Breakout: a candle closes above the resistance line. That uses the wedge up, whatever happens
   next. Grade "A+" when the breakout candle's volume is above `volume_mult` times its
   `volume_window`-candle average and its body (close minus open) is bigger than the ATR, else
   "B". Both grades are kept; backtests take A+ by default.
3. Retest: within `retest_window` candles after the breakout, a low comes back to within
   `retest_tolerance` above the broken line (low <= line * (1 + tolerance)). A close more than
   `retest_tolerance` below the line fails the breakout, and the search starts over.
4. Trigger: the retest candle, or the one right after it, closes up (close > open) in the upper
   half of its range: buyers defended the line. Without one in time the breakout expires.
5. Orders: buy at the trigger candle's close (a market order), stop `stop_atr` ATR below the
   lower of the retest's low and the line, take-profits at the wedge's widest height (at its
   first pivot) above the entry ("measured") and at `targets` multiples of the entry-to-stop
   distance (R). Once price reaches `breakeven_r` R the stop moves to the entry plus fees
   (labels.wedges).

One state machine per chart: while a breakout waits for its retest, no other wedge is looked
at. Everything is known at the trigger candle's close; outcomes are in labels.wedges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from futurecast.features.setups import SetupSpec, interval_minutes, interval_name
from futurecast.features.structure import swing_points

_ON_LINE = 1e-9  # closes this close to a line (relative) count as on it, not through it


@dataclass(frozen=True)
class WedgeSpec:
    """Wedge rules. Counts are in candles of the chart; tolerances are shares of the price."""

    strategy: ClassVar[str] = "wedge"
    market_entry: ClassVar[bool] = True  # entries fill at a candle's close, so trades start on the next one

    minutes: int = 60  # the chart
    swing_len: int = 5  # candles on each side of a pivot
    window: int = 60  # candles from the wedge's first pivot to the breakout, at most
    apex_min: int = 10  # the lines must meet between apex_min and apex_max candles after the breakout
    apex_max: int = 30
    touch_tolerance: float = 0.003  # a high (low) this close to a line tests it
    min_touches: int = 2  # tests of each line, pivots included
    atr_len: int = 14  # Wilder's average true range
    volume_window: int = 20  # candles in the breakout's volume average (the breakout included)
    volume_mult: float = 1.5  # an A+ breakout's volume over that average
    retest_window: int = 5  # candles after the breakout in which price may come back to the line
    retest_tolerance: float = 0.002  # how far short of the line a retest may stop, and how far back inside a close fails it
    stop_atr: float = 0.5  # the stop sits this many ATR beyond the retest
    targets: Tuple[Union[str, float], ...] = field(default=("measured", 2.0, 2.5))  # "measured": the wedge's height
    breakeven_r: Optional[float] = 1.5  # the stop moves to break-even once price gets this many R; None keeps it
    max_hold: int = 240  # candles a trade is held at most
    # Costs per leg, in bps of the entry price, the same CoinDCX fees as the fair-value-gap rules.
    # The entry, a stop and a timeout exit are market orders (taker; a stop also fills past its
    # price), the take-profit a resting limit order (maker).
    maker_bps: float = SetupSpec.maker_bps
    taker_bps: float = SetupSpec.taker_bps
    slippage_bps: float = SetupSpec.slippage_bps

    @classmethod
    def for_chart(cls, interval: str, **overrides) -> "WedgeSpec":
        """The rules on another chart, e.g. WedgeSpec.for_chart("4h")."""
        return cls(minutes=interval_minutes(interval), **overrides)

    @property
    def ltf_minutes(self) -> int:
        """The chart trades are entered on (the backtest's name for it)."""
        return self.minutes

    @property
    def entry_interval(self) -> str:
        return interval_name(self.minutes)

    def round_trip(self, outcome: str) -> float:
        """Fees and slippage of a trade as a share of the entry price, by how it ended: "won"
        (taker in, maker out), "lost" or "breakeven" (taker in, taker plus slippage out) or
        "timeout" (taker in and out)."""
        stop = self.taker_bps + self.slippage_bps
        exit_bps = {"won": self.maker_bps, "lost": stop, "breakeven": stop, "timeout": self.taker_bps}
        return (self.taker_bps + exit_bps[outcome]) / 10_000.0

    def breakeven_stop(self, entry: float, side: int) -> float:
        """Where the stop moves at `breakeven_r`: the entry plus both legs' taker fees, so the
        exit nets nothing but its slippage."""
        return entry * (1.0 + side * 2.0 * self.taker_bps / 10_000.0)

    def costs(self) -> Dict[str, float]:
        """Per-leg fees and each outcome's round trip, in percent, for reports."""
        return {"maker_pct": self.maker_bps / 100.0, "taker_pct": self.taker_bps / 100.0,
                "slippage_pct": self.slippage_bps / 100.0,
                **{f"{o}_pct": self.round_trip(o) * 100.0 for o in ("won", "lost", "timeout")}}


WEDGE_COLUMNS = [
    "idx", "side", "shape", "grade", "entry", "stop", "risk", "measured_r", "breakout_idx", "retest_idx",
    "high_1_idx", "high_2_idx", "low_1_idx", "low_2_idx", "res_slope", "sup_slope", "line",
    "height", "apex_ahead", "res_touches", "sup_touches", "volume_ratio", "body_atr", "breakout_close",
]


def average_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """Wilder's average true range; the first candle's true range is its high minus its low."""
    prev = np.r_[close[0], close[:-1]]
    true_range = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(true_range).ewm(alpha=1.0 / length, adjust=False).mean().to_numpy()


@dataclass(frozen=True)
class _Wedge:
    """Two lines: resistance through swing highs h1 and h2, support through swing lows l1 and l2."""

    h1: int
    h2: int
    l1: int
    l2: int
    res_at_h2: float
    res_slope: float  # price per candle
    sup_at_l2: float
    sup_slope: float

    @classmethod
    def through(cls, h1: int, h2: int, l1: int, l2: int, high: np.ndarray, low: np.ndarray) -> "_Wedge":
        return cls(h1, h2, l1, l2, float(high[h2]), float((high[h2] - high[h1]) / (h2 - h1)),
                   float(low[l2]), float((low[l2] - low[l1]) / (l2 - l1)))

    @property
    def start(self) -> int:
        return min(self.h1, self.l1)

    @property
    def converging(self) -> bool:
        return self.res_slope < self.sup_slope

    @property
    def shape(self) -> str:
        if self.res_slope > 0 and self.sup_slope > 0:
            return "rising wedge"
        if self.res_slope < 0 and self.sup_slope < 0:
            return "falling wedge"
        return "triangle"

    def res(self, x):
        return self.res_at_h2 + self.res_slope * (x - self.h2)

    def sup(self, x):
        return self.sup_at_l2 + self.sup_slope * (x - self.l2)

    def line(self, side: int, x):
        """The line a breakout to `side` crossed: resistance for longs, support for shorts."""
        return self.res(x) if side > 0 else self.sup(x)

    def holds(self, close: np.ndarray, stop: int) -> bool:
        """No close outside the lines from the first pivot through candle `stop` - 1."""
        x = np.arange(self.start, stop)
        c = close[self.start:stop]
        return bool(np.all(c - self.res(x) <= _ON_LINE * c) and np.all(self.sup(x) - c <= _ON_LINE * c))

    def touches(self, high: np.ndarray, low: np.ndarray, stop: int, tolerance: float) -> Tuple[int, int]:
        """Separate tests of the resistance and support lines from the first pivot through candle
        `stop` - 1: runs of candles whose high (low) came within `tolerance` of the line or past it."""
        x = np.arange(self.start, stop)
        near_res = high[self.start:stop] >= self.res(x) * (1.0 - tolerance)
        near_sup = low[self.start:stop] <= self.sup(x) * (1.0 + tolerance)
        return _runs(near_res), _runs(near_sup)


def _runs(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask & ~np.r_[False, mask[:-1]]))


def find_wedge_setups(candles: pd.DataFrame, spec: WedgeSpec | None = None) -> pd.DataFrame:
    """Every triggered wedge setup in `candles` (open/high/low/close/volume, indexed by open time).

    Returns one row per setup, indexed by the time of its trigger candle (the entry is that
    candle's close). `attrs["funnel"]` counts how far the patterns got: wedges, breakouts, and
    breakouts that failed (closed back inside), expired (no trigger in time) or triggered.
    """
    spec = spec or WedgeSpec()
    o, h, l, c, v = (candles[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
    n = len(c)
    atr = average_true_range(h, l, c, spec.atr_len)
    volume_avg = pd.Series(v).rolling(spec.volume_window, min_periods=1).mean().to_numpy()
    is_high, is_low = swing_points(h, l, spec.swing_len)
    highs, lows = np.flatnonzero(is_high), np.flatnonzero(is_low)
    # pivots known by each candle's close
    known_h = np.searchsorted(highs + spec.swing_len, np.arange(n), side="right")
    known_l = np.searchsorted(lows + spec.swing_len, np.arange(n), side="right")
    eps = spec.retest_tolerance

    funnel = {"wedges": 0, "breakouts": 0, "failed": 0, "expired": 0, "triggered": 0}
    rows, times = [], []
    key, wedge, inside, counted = None, None, False, False
    watch: Optional[Dict] = None  # a breakout waiting for its retest
    for t in range(1, n):
        if watch is not None:
            b, d, w = watch["idx"], watch["side"], watch["wedge"]
            line = w.line(d, t)
            if (c[t] - line * (1 - d * eps)) * d < 0:  # closed back inside
                funnel["failed"] += 1
                watch = None
            else:
                reach = l[t] if d > 0 else h[t]
                touched = t - b <= spec.retest_window and (reach - line * (1 + d * eps)) * d <= 0
                rejected = (c[t] - o[t]) * d > 0 and ((c[t] - l[t]) if d > 0 else (h[t] - c[t])) >= 0.5 * (h[t] - l[t])
                if rejected and (touched or watch["touch"] == t - 1):
                    rows.append(_setup_row(t, t if touched else t - 1, watch, h, l, c, atr, spec))
                    times.append(candles.index[t])
                    funnel["triggered"] += 1
                    watch = None
                elif touched:
                    watch["touch"] = t
                elif t - b >= spec.retest_window:  # no candle left that could trigger
                    funnel["expired"] += 1
                    watch = None
            if watch is not None:
                continue

        # Searching: the wedge drawn through the pivots known by the previous candle's close
        kh, kl = known_h[t - 1], known_l[t - 1]
        if kh < 2 or kl < 2:
            continue
        if (kh, kl) != key:
            key = (kh, kl)
            wedge = _Wedge.through(highs[kh - 2], highs[kh - 1], lows[kl - 2], lows[kl - 1], h, l)
            inside, counted = wedge.converging and wedge.holds(c, t), False
        if not inside:
            continue
        res, sup = wedge.res(t), wedge.sup(t)
        up, down = c[t] - res > _ON_LINE * c[t], sup - c[t] > _ON_LINE * c[t]
        gap = res - sup
        apex = gap / (wedge.sup_slope - wedge.res_slope)
        if gap > 0 and spec.apex_min <= apex <= spec.apex_max and t - wedge.start <= spec.window:
            res_touches, sup_touches = wedge.touches(h, l, t, spec.touch_tolerance)
            if min(res_touches, sup_touches) >= spec.min_touches:
                if not counted:
                    funnel["wedges"] += 1
                    counted = True
                if up or down:
                    d = 1 if up else -1
                    funnel["breakouts"] += 1
                    volume_ratio = v[t] / volume_avg[t] if volume_avg[t] > 0 else np.nan
                    body_atr = (c[t] - o[t]) * d / atr[t] if atr[t] > 0 else np.nan
                    watch = {"idx": t, "side": d, "wedge": wedge, "touch": -1, "apex": float(apex),
                             "touches": (res_touches, sup_touches), "volume_ratio": float(volume_ratio),
                             "body_atr": float(body_atr),
                             "grade": "A+" if volume_ratio > spec.volume_mult and body_atr > 1.0 else "B"}
        if up or down:
            inside = False  # a close outside uses the wedge up

    out = pd.DataFrame(rows, columns=WEDGE_COLUMNS, index=pd.DatetimeIndex(times, name="timestamp"))
    out.attrs["funnel"] = funnel
    return out


def _setup_row(t: int, retest: int, watch: Dict, high: np.ndarray, low: np.ndarray, close: np.ndarray,
               atr: np.ndarray, spec: WedgeSpec) -> Dict:
    """The orders of a breakout whose retest (from candle `retest`) triggered on candle `t`."""
    d, w = watch["side"], watch["wedge"]
    line = float(w.line(d, t))
    if d > 0:
        stop = min(float(low[retest:t + 1].min()), line) - spec.stop_atr * atr[t]
    else:
        stop = max(float(high[retest:t + 1].max()), line) + spec.stop_atr * atr[t]
    entry = float(close[t])
    R = abs(entry - stop)
    height = float(w.res(w.start) - w.sup(w.start))
    return {
        "idx": t, "side": d, "shape": w.shape, "grade": watch["grade"], "entry": entry, "stop": float(stop),
        "risk": R / entry, "measured_r": height / R, "breakout_idx": watch["idx"], "retest_idx": retest,
        "high_1_idx": w.h1, "high_2_idx": w.h2, "low_1_idx": w.l1, "low_2_idx": w.l2,
        "res_slope": w.res_slope, "sup_slope": w.sup_slope, "line": line, "height": height,
        "apex_ahead": watch["apex"], "res_touches": watch["touches"][0], "sup_touches": watch["touches"][1],
        "volume_ratio": watch["volume_ratio"], "body_atr": watch["body_atr"],
        "breakout_close": float(close[watch["idx"]]),
    }
