"""What happened to each wedge trade (see features.wedges).

The entry is a market order at the trigger candle's close, so it always fills, and the next
candle is the first that can reach the stop or a take-profit. From there a trade follows the
same rules as a fair-value-gap trade (labels.setups.resolve_trade): a take-profit before the
stop wins, the stop first or both in one candle loses, and `max_hold` candles without either
is a timeout at that close. Once a candle reaches `breakeven_r` R, the stop moves from the next
candle on to the entry plus fees (WedgeSpec.breakeven_stop), and touching it is a "breakeven"
exit, also when the take-profit is in the same candle.

Targets are fixed R multiples or "measured": the wedge's height from the entry, a different
multiple of R for every trade (the setup's measured_r). Results are in R before costs;
anything the data does not cover yet is left open (NaN).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from deltarisk.features.wedges import WedgeSpec
from deltarisk.labels.setups import outcome_table, record_trade, resolve_trade, target_key, target_multiples


def wedge_outcomes(setups: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series,
                   spec: WedgeSpec | None = None) -> pd.DataFrame:
    """Per-target outcome of every wedge setup, in the columns of labels.setups.setup_outcomes
    (fill is always "filled", on the trigger candle; outcomes can also be "breakeven")."""
    spec = spec or WedgeSpec()
    h, l, c = (s.to_numpy(dtype=float) for s in (high, low, close))
    out = outcome_table(len(setups), spec.targets)
    out["fill"][:] = "filled"
    out["fill_idx"][:] = setups["idx"].to_numpy(dtype=np.int64)
    multiples = {T: target_multiples(setups, T) for T in spec.targets}

    for row, (s, d, entry, stop) in enumerate(zip(setups["idx"].to_numpy(), setups["side"].to_numpy(),
                                                  setups["entry"].to_numpy(), setups["stop"].to_numpy())):
        breakeven = None if spec.breakeven_r is None else (spec.breakeven_r, spec.breakeven_stop(entry, d))
        for T in spec.targets:
            record_trade(out, row, target_key(T), resolve_trade(h, l, c, s, entry, stop, d, multiples[T][row],
                                                                spec.max_hold, at_close=True, breakeven=breakeven))
    return pd.DataFrame(out, index=setups.index)
