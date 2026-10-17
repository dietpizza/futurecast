"""What happened to each setup's orders (see features.setups).

The limit order waits up to `fill_window` candles after the setup candle. It fills on the first
candle that trades through the entry, and is cancelled if price first runs `cancel_r` R away
(3R) without filling. One cancel level for every target keeps a setup's fill the same whichever
target is traded. Once filled the trade is left alone ("set and forget"):

- It wins if price touches the take-profit before the stop. On the fill candle only the stop
  counts, because the candle cannot say whether its high came before or after the fill.
- It loses if the stop is touched first, or both in the same candle (ambiguous counts as a loss).
- Touching neither within `max_hold` candles is a timeout, closed at that candle's close.

Results are in R, the entry-to-stop distance: a win is +target, a loss -1, before costs.
Anything the data does not cover yet is left open (NaN), never guessed.

Costs depend on how a trade ended (SetupSpec.round_trip): a win pays the maker fee in and out,
a stop pays the maker fee in and the taker fee plus slippage out, a timeout maker in and taker
out. `break_even` turns costs in R into the win rate a trade needs.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from futurecast.features.setups import SetupSpec

def _first(mask: np.ndarray) -> int:
    """Position of the first True, or len(mask) when there is none."""
    return int(mask.argmax()) if mask.any() else len(mask)


def setup_outcomes(setups: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series,
                   spec: SetupSpec | None = None) -> pd.DataFrame:
    """Fill and per-target outcome of every setup, on the candles the setups were found on.

    Columns: fill ("pending" while the order can still fill, "filled", "expired", "cancelled"),
    fill_idx, order_end (candle an unfilled order stopped waiting on, -1 otherwise), and per
    target T (e.g. 3.0 -> suffix "3"): outcome_T ("open", "won", "lost", "timeout"; empty when
    never filled), win_T (1 / 0, NaN unless resolved), r_T (gross result in R, NaN unless
    resolved) and end_T (candle the trade resolved on, -1 unless resolved).
    """
    spec = spec or SetupSpec()
    h, l, c = (s.to_numpy(dtype=float) for s in (high, low, close))
    n = len(c)
    out = outcome_table(len(setups), spec.targets)

    for row, (s, d, entry, stop) in enumerate(zip(setups["idx"].to_numpy(), setups["side"].to_numpy(),
                                                  setups["entry"].to_numpy(), setups["stop"].to_numpy())):
        R = abs(entry - stop)
        start, stop_win = s + 1, min(s + 1 + spec.fill_window, n)
        if d > 0:
            fills = l[start:stop_win] <= entry
            runs = h[start:stop_win] >= entry + spec.cancel_r * R
        else:
            fills = h[start:stop_win] >= entry
            runs = l[start:stop_win] <= entry - spec.cancel_r * R
        f, r = _first(fills), _first(runs)
        if f < len(fills) and f <= r:
            fill = start + f
        else:
            if r < len(runs):
                out["fill"][row], out["order_end"][row] = "cancelled", start + r
            elif s + spec.fill_window < n:
                out["fill"][row], out["order_end"][row] = "expired", s + spec.fill_window
            continue
        out["fill"][row], out["fill_idx"][row] = "filled", fill
        for T in spec.targets:
            record_trade(out, row, target_key(T), resolve_trade(h, l, c, fill, entry, stop, d, T, spec.max_hold))

    return pd.DataFrame(out, index=setups.index)


def outcome_table(count: int, targets) -> Dict[str, np.ndarray]:
    """Outcome columns for `count` setups (see setup_outcomes): every order pending, no trade resolved."""
    out = {"fill": np.full(count, "pending", dtype=object), "fill_idx": np.full(count, -1, dtype=np.int64),
           "order_end": np.full(count, -1, dtype=np.int64)}
    for T in targets:
        key = target_key(T)
        out[f"outcome_{key}"] = np.full(count, "", dtype=object)
        out[f"win_{key}"] = np.full(count, np.nan)
        out[f"r_{key}"] = np.full(count, np.nan)
        out[f"end_{key}"] = np.full(count, -1, dtype=np.int64)
    return out


def record_trade(out: Dict[str, np.ndarray], row: int, key: str, result: Tuple[str, float, int]) -> None:
    """Store resolve_trade's result for one setup and target in an outcome_table."""
    outcome, gross, end = result
    out[f"outcome_{key}"][row] = outcome
    if outcome != "open":
        out[f"win_{key}"][row] = 1.0 if outcome == "won" else 0.0
        out[f"r_{key}"][row] = gross
        out[f"end_{key}"][row] = end


def resolve_trade(high: np.ndarray, low: np.ndarray, close: np.ndarray, fill: int, entry: float, stop: float,
                  side: int, target_r: float, max_hold: int, at_close: bool = False,
                  breakeven: Optional[Tuple[float, float]] = None) -> Tuple[str, float, int]:
    """How a trade filled on candle `fill` ended: (outcome, gross result in R, candle it ended on).

    Outcomes: "won" (take-profit `target_r` R away touched before the stop), "lost" (the stop
    first, or both in one candle), "breakeven", "timeout" (neither through candle fill + max_hold,
    closed at that candle's close) or "open" (the data ends first; NaN and -1).

    A limit fill (at_close False) happens inside the fill candle, which can stop the trade out
    but not win it, since its high may have come before the fill. A fill at the candle's close
    (at_close True) starts the trade on the next candle. `breakeven` is (R multiple, new stop):
    from the candle after the one that first reaches that many R, the stop sits at the new
    level, and touching it is a "breakeven" exit (with the take-profit in the same candle too).
    """
    n = len(close)
    R = abs(entry - stop)
    first = fill + 1 if at_close else fill
    last = min(fill + max_hold, n - 1)
    hs, ls = high[first:last + 1], low[first:last + 1]
    up = side > 0
    stopped = (ls <= stop) if up else (hs >= stop)
    hit = (hs >= entry + target_r * R) if up else (ls <= entry - target_r * R)
    if not at_close and len(hit):
        hit[0] = False
    sl, tp = _first(stopped), _first(hit)
    exit_kind, exit_r = "lost", -1.0
    if breakeven is not None:
        be_r, be_stop = breakeven
        reached = (hs >= entry + be_r * R) if up else (ls <= entry - be_r * R)
        if not at_close and len(reached):
            reached[0] = False
        a = _first(reached)
        if a < sl and a < tp:
            later = (ls[a + 1:] <= be_stop) if up else (hs[a + 1:] >= be_stop)
            sl, exit_kind, exit_r = a + 1 + _first(later), "breakeven", (be_stop - entry) * side / R
    if tp < sl and tp < len(hit):
        return "won", float(target_r), first + tp
    if sl < len(stopped):
        return exit_kind, float(exit_r), first + sl
    if fill + max_hold < n:
        return "timeout", float((close[fill + max_hold] - entry) * side / R), fill + max_hold
    return "open", np.nan, -1


def break_even(target: float, loss_ratio: float, win_cost_r, loss_cost_r):
    """Win rate at which a filled trade makes nothing after costs: a win earns target - win_cost_r R,
    a non-win loses loss_ratio + loss_cost_r R."""
    lose = loss_ratio + np.asarray(loss_cost_r, dtype=float)
    return lose / (target - np.asarray(win_cost_r, dtype=float) + lose)


def target_key(target) -> str:
    """Column suffix for a take-profit: 3.0 -> "3", 2.5 -> "2.5", "measured" -> "measured"."""
    return target if isinstance(target, str) else f"{target:g}"


def target_multiples(setups: pd.DataFrame, target) -> np.ndarray:
    """Each setup's take-profit in R: a fixed multiple, or a per-setup one (e.g. "measured" from
    the setups' measured_r column)."""
    if isinstance(target, str):
        return setups[f"{target}_r"].to_numpy(dtype=float)
    return np.full(len(setups), float(target))
