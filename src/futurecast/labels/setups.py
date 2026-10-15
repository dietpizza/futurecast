"""What happened to each setup's orders (see features.setups).

The limit order waits up to `fill_window` candles after the setup candle. It fills on the first
candle that trades through the entry, and is cancelled if price first reaches the smallest
take-profit without filling. Once filled the trade is left alone ("set and forget"):

- It wins if price touches the take-profit before the stop. On the fill candle only the stop
  counts, because the candle cannot say whether its high came before or after the fill.
- It loses if the stop is touched first, or both in the same candle (ambiguous counts as a loss).
- Touching neither within `max_hold` candles is a timeout, closed at that candle's close.

Results are in R, the entry-to-stop distance: a win is +target, a loss -1, before costs.
Anything the data does not cover yet is left open (NaN), never guessed.

Costs depend on how a trade ended (SetupSpec.round_trip): a win pays the maker fee in and out,
a stop pays the maker fee in and the taker fee plus slippage out, a timeout maker in and taker
out. `trade_cost_r` puts them in R; `break_even` and `expected_r` use them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from futurecast.features.setups import SetupSpec

FILL_STATES = ("pending", "filled", "expired", "cancelled")


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
    first_target = min(spec.targets)
    out = {"fill": np.full(len(setups), "pending", dtype=object), "fill_idx": np.full(len(setups), -1, dtype=np.int64),
           "order_end": np.full(len(setups), -1, dtype=np.int64)}
    for T in spec.targets:
        key = target_key(T)
        out[f"outcome_{key}"] = np.full(len(setups), "", dtype=object)
        out[f"win_{key}"] = np.full(len(setups), np.nan)
        out[f"r_{key}"] = np.full(len(setups), np.nan)
        out[f"end_{key}"] = np.full(len(setups), -1, dtype=np.int64)

    for row, (s, d, entry, stop) in enumerate(zip(setups["idx"].to_numpy(), setups["side"].to_numpy(),
                                                  setups["entry"].to_numpy(), setups["stop"].to_numpy())):
        R = abs(entry - stop)
        start, stop_win = s + 1, min(s + 1 + spec.fill_window, n)
        if d > 0:
            fills = l[start:stop_win] <= entry
            runs = h[start:stop_win] >= entry + first_target * R
        else:
            fills = h[start:stop_win] >= entry
            runs = l[start:stop_win] <= entry - first_target * R
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

        last = min(fill + spec.max_hold, n - 1)
        stopped = (l[fill:last + 1] <= stop) if d > 0 else (h[fill:last + 1] >= stop)
        sl = _first(stopped)
        for T in spec.targets:
            key = target_key(T)
            tp_price = entry + d * T * R
            hit = (h[fill:last + 1] >= tp_price) if d > 0 else (l[fill:last + 1] <= tp_price)
            hit[0] = False  # the fill candle can stop a trade out but not win it
            tp = _first(hit)
            if tp < sl and tp < len(hit):
                outcome, gross, end = "won", T, fill + tp
            elif sl < len(stopped):
                outcome, gross, end = "lost", -1.0, fill + sl
            elif fill + spec.max_hold < n:
                outcome, gross, end = "timeout", (c[fill + spec.max_hold] - entry) * d / R, fill + spec.max_hold
            else:
                out[f"outcome_{key}"][row] = "open"
                continue
            out[f"outcome_{key}"][row] = outcome
            out[f"win_{key}"][row] = 1.0 if outcome == "won" else 0.0
            out[f"r_{key}"][row] = gross
            out[f"end_{key}"][row] = end

    return pd.DataFrame(out, index=setups.index)


def trade_cost_r(outcome, risk, spec: SetupSpec) -> np.ndarray:
    """Fees and slippage of each trade in R, by its outcome ("won", "lost", "timeout"; NaN for
    anything else). `risk` is the entry-to-stop distance as a share of the entry price."""
    share = {o: spec.round_trip(o) for o in ("won", "lost", "timeout")}
    return np.array([share.get(o, np.nan) for o in np.atleast_1d(outcome)]) / np.asarray(risk, dtype=float)


def expected_r(p, target: float, loss_ratio: float, win_cost_r, loss_cost_r):
    """Expected net result per filled trade in R: a win earns target - win_cost_r, a non-win
    loses loss_ratio + loss_cost_r."""
    p = np.asarray(p, dtype=float)
    return p * (target - np.asarray(win_cost_r, dtype=float)) - (1.0 - p) * (loss_ratio + np.asarray(loss_cost_r, dtype=float))


def break_even(target: float, loss_ratio: float, win_cost_r, loss_cost_r):
    """Win rate at which a filled trade makes nothing after costs (where expected_r is 0)."""
    lose = loss_ratio + np.asarray(loss_cost_r, dtype=float)
    return lose / (target - np.asarray(win_cost_r, dtype=float) + lose)


def target_key(target: float) -> str:
    """Column suffix for a take-profit multiple: 3.0 -> "3", 2.5 -> "2.5"."""
    return f"{target:g}"
