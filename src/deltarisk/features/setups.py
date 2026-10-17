"""Fair-value-gap setups: a higher context chart, a lower entry chart (rules only, no models).

The same rules run on any pair of charts (SetupSpec.ltf_minutes for entries, htf_minutes for
context): 15m entries in 1h context by default, the pair that held up best in backtests of
1m/15m, 5m/1h, 15m/1h and 15m/4h. A long setup (shorts mirror it):

1. Context: the context chart's trend is up (its latest structure break was upward), and the
   entry chart's pullback before the setup reached into a bullish context-chart fair value gap
   that existed before price got there and that no context candle has closed below since
   (price was absorbed, not sliced through).
2. Shift: an entry-chart change of character (CHoCH) up, i.e. a close above the last swing
   high while the entry chart's trend was down.
3. Gap: the move that broke structure left a bullish entry-chart fair value gap behind (the
   gap's first candle at or after the pullback's low, its third candle at most
   `gap_after_break` candles after the CHoCH).
4. Orders: a limit buy at the gap's midpoint, the stop just below the pullback's low, and
   take-profits at `targets` multiples of the entry-to-stop distance (R): 2R, 3R and 4R.

Every entry-chart CHoCH that leaves a gap is a setup. Step 1's two checks are recorded, not
required: a setup passing all four steps is grade "A+", the rest are grade "B". Backtests take
A+ setups by default. Everything is known at the setup candle's close; outcomes are in
labels.setups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, Tuple

import numpy as np
import pandas as pd

from futurecast.features.structure import CHOCH, UP, fair_value_gaps, market_structure


_UNIT_MINUTES = {"m": 1, "h": 60, "d": 1440}


def interval_minutes(interval: str) -> int:
    """Chart interval name to minutes: "15m" -> 15, "4h" -> 240."""
    return int(interval[:-1]) * _UNIT_MINUTES[interval[-1]]


def interval_name(minutes: int) -> str:
    """Minutes to a chart interval name: 240 -> "4h", 15 -> "15m"."""
    for unit, size in (("d", 1440), ("h", 60)):
        if minutes % size == 0:
            return f"{minutes // size}{unit}"
    return f"{minutes}m"


@dataclass(frozen=True)
class SetupSpec:
    """Setup rules. Counts are in candles of each chart (entry chart for `ltf_*`, `fill_window`
    and `max_hold`, context chart for `htf_*`), so on higher charts every window scales with the
    chart: the same rules, zoomed out. Comments give the clock time on 15m entries / 1h context.
    """

    strategy: ClassVar[str] = "fvg"
    market_entry: ClassVar[bool] = False  # entries are limit orders filled inside a candle

    ltf_minutes: int = 15  # entry chart
    htf_minutes: int = 60  # context chart
    ltf_swing_len: int = 3  # entry candles on each side of a swing point (internal structure, 45 minutes)
    htf_swing_len: int = 5  # context candles on each side (5 hours)
    htf_zone_max_age: int = 288  # context candles a gap stays a zone (12 days)
    ltf_gap_max_age: int = 120  # entry candles an entry-chart gap stays a zone (30 hours)
    gap_after_break: int = 2  # entry candles after the CHoCH in which its gap may still complete
    zone_tolerance: float = 0.25  # the pullback may poke past a zone's far side by this share of its height
    stop_buffer_bps: float = 2.0  # the stop sits this far beyond the pullback's extreme
    min_risk_bps: float = 10.0  # tighter entry-to-stop distances are skipped: fees would swallow them
    fill_window: int = 30  # entry candles the limit order waits for a fill (7.5 hours)
    max_hold: int = 240  # entry candles a filled trade is held at most (2.5 days)
    targets: Tuple[float, ...] = field(default=(2.0, 3.0, 4.0))  # take-profits in multiples of R
    cancel_r: float = 3.0  # an unfilled limit is cancelled once price runs this many R away, for every target
    # Costs per leg, in bps of the entry price. The entry and the take-profit are resting limit
    # orders (maker); the stop and a timeout exit are market orders (taker), and a triggered stop
    # also fills past its price. Fees: CoinDCX USDT-M futures, General tier, 0.025% maker and
    # 0.075% taker plus 18% GST on fees (https://coindcx.com/vip, checked October 2026).
    maker_bps: float = 2.95  # 0.025% + GST
    taker_bps: float = 8.85  # 0.075% + GST
    slippage_bps: float = 2.0  # a stop's fill past its price; an assumption, not measured

    @classmethod
    def for_charts(cls, entry: str, context: str, **overrides) -> "SetupSpec":
        """The rules on another pair of charts, e.g. SetupSpec.for_charts("5m", "1h")."""
        return cls(ltf_minutes=interval_minutes(entry), htf_minutes=interval_minutes(context), **overrides)

    @property
    def entry_interval(self) -> str:
        return interval_name(self.ltf_minutes)

    @property
    def context_interval(self) -> str:
        return interval_name(self.htf_minutes)

    def round_trip(self, outcome: str) -> float:
        """Fees and slippage of a filled trade as a share of the entry price, by how it ended:
        "won" (maker in, maker out), "lost" (maker in, taker plus slippage out) or "timeout"
        (maker in, taker out). Both legs are charged on the entry price; the exit price differs
        from it by the trade's move, a rounding error on the fee at these stop sizes."""
        exit_bps = {"won": self.maker_bps, "lost": self.taker_bps + self.slippage_bps, "timeout": self.taker_bps}
        return (self.maker_bps + exit_bps[outcome]) / 10_000.0

    def costs(self) -> Dict[str, float]:
        """Per-leg fees and each outcome's round trip, in percent, for reports and the dashboard."""
        return {"maker_pct": self.maker_bps / 100.0, "taker_pct": self.taker_bps / 100.0,
                "slippage_pct": self.slippage_bps / 100.0,
                **{f"{o}_pct": self.round_trip(o) * 100.0 for o in ("won", "lost", "timeout")}}


SETUP_COLUMNS = [
    "idx", "side", "break_idx", "break_level", "extreme", "gap_top", "gap_bottom", "entry", "stop", "risk",
    "htf_trend", "htf_bias", "htf_zone", "zone_top", "zone_bottom",
]


def close_times(index: pd.DatetimeIndex, minutes: int) -> np.ndarray:
    """Close time of each candle (candles are indexed by open time), as int64 nanoseconds."""
    return (index + pd.Timedelta(minutes=minutes)).as_unit("ns").asi8


def htf_positions(ltf_index: pd.DatetimeIndex, htf_index: pd.DatetimeIndex, spec: SetupSpec) -> np.ndarray:
    """For each entry candle, the position of the latest context candle closed by its close (-1 if none)."""
    return np.searchsorted(close_times(htf_index, spec.htf_minutes),
                           close_times(ltf_index, spec.ltf_minutes), side="right") - 1


def find_setups(ltf: pd.DataFrame, htf: pd.DataFrame, spec: SetupSpec | None = None) -> pd.DataFrame:
    """Every setup in `ltf` (entry-chart candles), with its context from `htf` (context-chart candles).

    Both frames need open/high/low/close columns and are indexed by candle open time. Returns one
    row per setup, indexed by the time of the candle at whose close it is complete.
    """
    spec = spec or SetupSpec()
    high, low, close = (ltf[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    ls = market_structure(high, low, close, spec.ltf_swing_len)
    lg = fair_value_gaps(high, low, close, spec.ltf_gap_max_age)
    hh, hl, hc = (htf[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    hs = market_structure(hh, hl, hc, spec.htf_swing_len)
    hg = fair_value_gaps(hh, hl, hc, spec.htf_zone_max_age)
    pos = htf_positions(ltf.index, htf.index, spec)

    gaps_by_dir = {d: lg[lg["direction"] == d] for d in (1, -1)}
    zones_by_dir = {d: hg[hg["direction"] == d] for d in (1, -1)}
    break_dir = ls["break_dir"].to_numpy()
    h_trend = hs["trend"].to_numpy()
    buffer = spec.stop_buffer_bps / 10_000.0

    rows, times = [], []
    for t in np.flatnonzero(ls["break_kind"].to_numpy() == CHOCH):
        d = int(break_dir[t])
        ext_i = int(ls["leg_extreme_idx"].iat[t])

        # 3. a gap left by the move that broke structure
        gaps = gaps_by_dir[d]
        g_idx = gaps["idx"].to_numpy()
        lo, hi = np.searchsorted(g_idx, ext_i + 2), np.searchsorted(g_idx, t + spec.gap_after_break, side="right")
        if lo >= hi:
            continue
        before = np.flatnonzero(g_idx[lo:hi] <= t)
        g = lo + (before[-1] if len(before) else 0)  # the latest gap by the CHoCH, else the first one after it
        s = max(t, int(g_idx[g]))
        if (break_dir[t + 1:s + 1] == -d).any():  # structure turned back before the gap completed
            continue

        # 4. orders
        top, bottom = float(gaps["top"].iat[g]), float(gaps["bottom"].iat[g])
        if d == UP:
            extreme = float(low[ext_i:s + 1].min())
            stop = extreme * (1.0 - buffer)
        else:
            extreme = float(high[ext_i:s + 1].max())
            stop = extreme * (1.0 + buffer)
        entry = (top + bottom) / 2.0
        R = abs(entry - stop)
        if R / entry < spec.min_risk_bps / 10_000.0:
            continue

        # 1. 15m context as of the setup candle's close
        p, p_ext = int(pos[s]), int(pos[ext_i])
        row = {
            "idx": s, "side": d, "break_idx": int(t), "break_level": float(ls["break_level"].iat[t]),
            "extreme": extreme, "gap_top": top, "gap_bottom": bottom, "entry": entry, "stop": stop,
            "risk": R / entry, "htf_trend": int(h_trend[p]) if p >= 0 else 0,
            "htf_zone": False, "zone_top": np.nan, "zone_bottom": np.nan,
        }
        row["htf_bias"] = row["htf_trend"] == d

        # zones that existed before price reached the extreme and still stand at the setup
        zones = zones_by_dir[d]
        z_idx = zones["idx"].to_numpy()
        if p_ext >= 0 and len(z_idx):
            zlo = np.searchsorted(z_idx, p - spec.htf_zone_max_age - 1)
            zhi = np.searchsorted(z_idx, p_ext, side="right")
            cand = np.arange(zlo, zhi)
            cand = cand[zones["active_until"].to_numpy()[cand] > p]
            if len(cand):
                zt, zb = zones["top"].to_numpy()[cand], zones["bottom"].to_numpy()[cand]
                height = zt - zb
                if d == UP:
                    inside = (extreme <= zt) & (extreme >= zb - spec.zone_tolerance * height)
                else:
                    inside = (extreme >= zb) & (extreme <= zt + spec.zone_tolerance * height)
                if inside.any():
                    k = np.flatnonzero(inside)[-1]  # the most recent zone holding the extreme
                    row.update({"htf_zone": True, "zone_top": float(zt[k]), "zone_bottom": float(zb[k])})
        rows.append(row)
        times.append(ltf.index[s])

    out = pd.DataFrame(rows, columns=SETUP_COLUMNS, index=pd.DatetimeIndex(times, name="timestamp"))
    out["grade"] = np.where(out["htf_bias"].astype(bool) & out["htf_zone"].astype(bool), "A+", "B")
    return out
