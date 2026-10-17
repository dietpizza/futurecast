"""Market structure (swings, BOS, CHoCH) and fair value gaps: exact rules, never looking ahead."""

import numpy as np
import pandas as pd

from deltarisk.features.setups import SetupSpec, find_setups
from deltarisk.features.structure import (
    BOS, CHOCH, DOWN, UP, fair_value_gaps, market_structure, swing_points,
)


def _walk(n, seed, minutes=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.001, n))
    idx = pd.date_range("2026-01-01", periods=n, freq=f"{minutes}min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + wick),
                         "low": np.minimum(open_, close) * (1 - wick), "close": close}, index=idx)


def test_swing_points_need_both_sides():
    high = np.array([1, 2, 3, 2, 1, 2, 5, 2, 1], dtype=float)
    is_high, _ = swing_points(high, high - 1, 2)
    assert list(np.flatnonzero(is_high)) == [2, 6]
    # The last two candles cannot be swings yet: their right side has not closed
    assert not swing_points(np.array([1, 2, 3, 4, 9.0]), np.zeros(5), 2)[0][-2:].any()


def test_bos_then_choch_with_their_legs():
    #               0     1     2     3     4     5     6     7     8     9
    high = np.array([10, 11, 12, 11.5, 10.5, 11, 12.5, 12.4, 12, 11])
    low = np.array([9, 10, 11, 10, 9, 9.5, 10.8, 11.5, 10.5, 8.5])
    close = np.array([9.5, 10.5, 11.5, 10.2, 9.2, 10.8, 12.3, 11.8, 10.8, 8.8])
    s = market_structure(high, low, close, swing_len=1)

    # Close 12.3 clears the swing high at 12 (candle 2): first break, so a BOS that sets the trend up
    assert s["break_dir"].iat[6] == UP and s["break_kind"].iat[6] == BOS
    assert s["break_level"].iat[6] == 12 and s["leg_extreme_idx"].iat[6] == 4 and low[4] == 9
    # Close 8.8 loses the swing low at 9 (candle 4) while the trend is up: a change of character
    assert s["break_dir"].iat[9] == DOWN and s["break_kind"].iat[9] == CHOCH
    assert s["break_level"].iat[9] == 9 and s["leg_extreme_idx"].iat[9] == 6 and high[6] == 12.5
    assert list(s["trend"]) == [0, 0, 0, 0, 0, 0, 1, 1, 1, -1]
    assert (s["break_dir"].to_numpy()[[0, 1, 2, 3, 4, 5, 7, 8]] == 0).all()
    assert s["last_break"].iat[8] == 6


def test_fair_value_gaps_and_their_zones():
    high = np.array([10, 12, 13, 12.5, 12, 11, 11])
    low = np.array([9, 10, 10.5, 11, 11.5, 10.6, 9.5])
    close = np.array([9.8, 11.8, 12.8, 12, 11.8, 10.8, 9.8])
    gaps = fair_value_gaps(high, low, close, max_age=10)
    bull = gaps[gaps["direction"] == UP].iloc[0]
    assert bull["idx"] == 2 and bull["bottom"] == 10 and bull["top"] == 10.5  # candle 0 high to candle 2 low
    # Candle 6 closes below the gap's bottom: price sliced through, so it stops being a zone there
    assert bull["active_until"] == 6

    # A gap also ages out
    young = fair_value_gaps(high[:5], low[:5], close[:5], max_age=1)
    assert young.loc[young["idx"] == 2, "active_until"].iat[0] == 4


def test_structure_and_zones_never_look_ahead():
    candles = _walk(3000, seed=3)
    h, l, c = (candles[k].to_numpy() for k in ("high", "low", "close"))
    full, part = market_structure(h, l, c, 3), market_structure(h[:2000], l[:2000], c[:2000], 3)
    pd.testing.assert_frame_equal(full.iloc[:2000], part)

    # Gaps known by candle 1999 are the same, and so is whether each is still a zone up to there
    def known(gaps):
        g = gaps[gaps["idx"] < 2000].reset_index(drop=True)
        return g.assign(active_until=g["active_until"].clip(upper=2000))

    gaps_full, gaps_part = fair_value_gaps(h, l, c, 100), fair_value_gaps(h[:2000], l[:2000], c[:2000], 100)
    pd.testing.assert_frame_equal(known(gaps_full), known(gaps_part))


def test_live_tail_finds_the_same_setups_as_the_full_history():
    ltf = _walk(12_000, seed=5)
    htf = ltf.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    spec = SetupSpec(ltf_minutes=1, htf_minutes=15)
    full = find_setups(ltf, htf, spec)
    tail = find_setups(ltf.iloc[-6000:], htf.iloc[-400:], spec)  # what the live buffers hold
    recent = full.index[-1] - pd.Timedelta(hours=40)
    a, b = full[full.index > recent], tail[tail.index > recent]
    assert len(a) > 10
    cols = ["side", "entry", "stop", "grade", "htf_trend", "htf_zone", "zone_top"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
