"""Market structure and fair value gaps, found by exact rules (no models).

Swing points: a swing high is a candle whose high is above the `swing_len` candles before it
and not below the `swing_len` candles after it (swing lows mirror this). It is only known
`swing_len` candles later, once the right side has closed, so every value below is stored on
the candle where it first becomes known, never earlier.

Structure: walking forward, a close above the latest unbroken swing high breaks structure up
and a close below the latest unbroken swing low breaks it down. A break in the direction of
the current trend is a break of structure (BOS: the trend continues). A break against it is a
change of character (CHoCH: the trend flips). Closes decide breaks, not wicks. The first break
of a series sets the trend and counts as a BOS.

Fair value gap (FVG): three candles where the first candle's high is below the third candle's
low (bullish: price skipped that range on the way up) or its low is above the third candle's
high (bearish). Known at the third candle's close. A gap stays a zone until a candle closes
beyond its far side (price sliced through it) or it is older than `max_age` candles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

UP, DOWN = 1, -1
BOS, CHOCH = 1, 2
KIND_NAMES = {BOS: "BOS", CHOCH: "CHoCH"}


def swing_points(high: np.ndarray, low: np.ndarray, swing_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Masks of swing highs and lows, set on the pivot candle (known `swing_len` candles later)."""
    high, low = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    n, k = len(high), swing_len
    is_high, is_low = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    if n < 2 * k + 1:
        return is_high, is_low
    win_h = sliding_window_view(high, 2 * k + 1)  # row r covers candles r .. r + 2k; the pivot is r + k
    win_l = sliding_window_view(low, 2 * k + 1)
    pivot_h, pivot_l = win_h[:, k], win_l[:, k]
    is_high[k:n - k] = (pivot_h > win_h[:, :k].max(axis=1)) & (pivot_h >= win_h[:, k + 1:].max(axis=1))
    is_low[k:n - k] = (pivot_l < win_l[:, :k].min(axis=1)) & (pivot_l <= win_l[:, k + 1:].min(axis=1))
    return is_high, is_low


def market_structure(high, low, close, swing_len: int) -> pd.DataFrame:
    """Trend, breaks and swing levels per candle, each known at that candle's close.

    Columns:
    - trend: 1 up, -1 down, 0 before the first break.
    - break_dir / break_kind: direction (1 / -1) and kind (BOS / CHOCH) of a break on this candle, else 0.
    - break_level: the swing level the close broke; break_origin: that swing's pivot candle.
    - leg_extreme / leg_extreme_idx: the lowest low (up break) or highest high (down break) between
      the broken swing and the break, i.e. where the move that broke structure started.
    - swing_high / swing_low: the latest confirmed swing levels (broken or not).
    - last_break: position of the most recent break up to this candle (-1 before the first).
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    is_high, is_low = swing_points(high, low, swing_len)

    trend = np.zeros(n, dtype=np.int8)
    break_dir = np.zeros(n, dtype=np.int8)
    break_kind = np.zeros(n, dtype=np.int8)
    break_level = np.full(n, np.nan)
    break_origin = np.full(n, -1, dtype=np.int64)
    leg_extreme = np.full(n, np.nan)
    leg_extreme_idx = np.full(n, -1, dtype=np.int64)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    last_break = np.full(n, -1, dtype=np.int64)

    state, last = 0, -1
    sh, sh_i, sh_live = np.nan, -1, False
    sl, sl_i, sl_live = np.nan, -1, False
    confirmed_high = np.flatnonzero(is_high) + swing_len  # candle on which each swing becomes known
    confirmed_low = np.flatnonzero(is_low) + swing_len
    new_high = np.zeros(n + swing_len + 1, dtype=bool)
    new_low = np.zeros(n + swing_len + 1, dtype=bool)
    new_high[confirmed_high] = True
    new_low[confirmed_low] = True
    closes = close.tolist()

    for t in range(n):
        if new_high[t]:
            sh_i = t - swing_len
            sh, sh_live = high[sh_i], True
        if new_low[t]:
            sl_i = t - swing_len
            sl, sl_live = low[sl_i], True
        c = closes[t]
        if sh_live and c > sh:
            seg = low[sh_i + 1:t + 1]
            m = int(np.argmin(seg))
            break_dir[t], break_kind[t] = UP, CHOCH if state == DOWN else BOS
            break_level[t], break_origin[t] = sh, sh_i
            leg_extreme[t], leg_extreme_idx[t] = seg[m], sh_i + 1 + m
            state, sh_live, last = UP, False, t
        elif sl_live and c < sl:
            seg = high[sl_i + 1:t + 1]
            m = int(np.argmax(seg))
            break_dir[t], break_kind[t] = DOWN, CHOCH if state == UP else BOS
            break_level[t], break_origin[t] = sl, sl_i
            leg_extreme[t], leg_extreme_idx[t] = seg[m], sl_i + 1 + m
            state, sl_live, last = DOWN, False, t
        trend[t] = state
        swing_high[t], swing_low[t] = sh, sl
        last_break[t] = last

    return pd.DataFrame({
        "trend": trend, "break_dir": break_dir, "break_kind": break_kind, "break_level": break_level,
        "break_origin": break_origin, "leg_extreme": leg_extreme, "leg_extreme_idx": leg_extreme_idx,
        "swing_high": swing_high, "swing_low": swing_low, "last_break": last_break,
    })


def fair_value_gaps(high, low, close, max_age: int) -> pd.DataFrame:
    """Every fair value gap, one row each, with how long it stays a zone.

    Columns: idx (the third candle, where the gap becomes known), direction (1 bullish, -1 bearish),
    top, bottom, touched (first later candle that traded into the gap, -1 if none within max_age)
    and active_until (first candle at whose close the gap is no longer a zone: a close beyond its
    far side, or max_age candles after it formed). A gap is a zone on candles idx .. active_until - 1.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    t = np.arange(2, n)
    bull = low[2:] > high[:-2]
    bear = high[2:] < low[:-2]
    idx = np.concatenate([t[bull], t[bear]])
    direction = np.concatenate([np.full(bull.sum(), UP), np.full(bear.sum(), DOWN)]).astype(np.int8)
    top = np.concatenate([low[2:][bull], low[:-2][bear]])
    bottom = np.concatenate([high[:-2][bull], high[2:][bear]])
    order = np.argsort(idx, kind="stable")
    idx, direction, top, bottom = idx[order], direction[order], top[order], bottom[order]

    touched = np.full(len(idx), -1, dtype=np.int64)
    active_until = idx + max_age + 1
    for g in range(len(idx)):
        start, stop = idx[g] + 1, min(idx[g] + max_age + 1, n)
        if start >= stop:
            continue
        if direction[g] == UP:
            dead = close[start:stop] < bottom[g]
            hit = low[start:stop] <= top[g]
        else:
            dead = close[start:stop] > top[g]
            hit = high[start:stop] >= bottom[g]
        if dead.any():
            active_until[g] = start + int(dead.argmax())
        if hit.any():
            touched[g] = start + int(hit.argmax())

    return pd.DataFrame({"idx": idx, "direction": direction, "top": top, "bottom": bottom,
                         "touched": touched, "active_until": active_until})


def active_gaps(gaps: pd.DataFrame, at: int) -> pd.DataFrame:
    """Gaps that are zones at the close of candle `at`."""
    return gaps[(gaps["idx"] <= at) & (gaps["active_until"] > at)]
