"""Supporting signals at the latest closed candle, each read against the coin's own history:
RSI, the 1h moving-average stack, where price sits in its recent ranges, realized volatility,
traded value and the funding rate. Plain statistics; no fitted models.

`signals_view` takes aligned frames (interval -> frame with `{symbol}_close` etc. columns,
indexed by candle open time) and returns None for anything the candles on hand can't support.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from futurecast.features.setups import interval_minutes
from futurecast.stats.momentum import percentile_of

RSI_PERIOD = 14
RSI_CHARTS = ("15m", "1h")
TREND_CHART = "1h"
EMA_SPANS = (20, 50, 200)
RANGES = (("24h", "15m", 1), ("7d", "1h", 7), ("30d", "1h", 30))  # name, chart, days
VOL_CHART = "15m"  # realized volatility over the last day of these candles
VOLUME_CHART = "1h"
VOLUME_DAYS = 30  # the last 24 hours' traded value against the median day before them
FUNDING_CHART = "1h"
FUNDING_DAYS = 90


def _per_day(interval: str) -> int:
    return 1440 // interval_minutes(interval)


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> Optional[float]:
    """Wilder's relative strength index of the latest close (0-100)."""
    if len(close) <= 2 * period:
        return None
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    loss = (-delta).clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    return float(100.0 - 100.0 / (1.0 + gain / loss))


def ema_stack(close: pd.Series, spans=EMA_SPANS) -> Optional[Dict[str, Any]]:
    """Price against its exponential moving averages, fastest first: "up" when price > EMA20 >
    EMA50 > EMA200, "down" when the order is reversed, else "mixed"."""
    if len(close) < 3 * max(spans):
        return None
    close = close.astype(float)
    emas = [float(close.ewm(span=s, adjust=False).mean().iloc[-1]) for s in spans]
    line = [float(close.iloc[-1]), *emas]
    pairs = list(zip(line, line[1:]))
    stack = "up" if all(a > b for a, b in pairs) else "down" if all(a < b for a, b in pairs) else "mixed"
    return {"ema": {str(s): v for s, v in zip(spans, emas)}, "stack": stack,
            "vs_slowest_pct": 100.0 * (line[0] / emas[-1] - 1.0)}


def range_position(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> Optional[Dict[str, float]]:
    """Where the latest close sits between the lowest low and highest high of the last n candles
    (0 at the low, 1 at the high)."""
    if len(close) < n:
        return None
    hi, lo = float(high.iloc[-n:].max()), float(low.iloc[-n:].min())
    last = float(close.iloc[-1])
    return {"high": hi, "low": lo, "position": (last - lo) / (hi - lo) if hi > lo else 0.5}


def realized_vol(close: pd.Series, per_day: int) -> Optional[Dict[str, float]]:
    """The last day's realized volatility (standard deviation of one-candle log returns, scaled to
    a day) and its percentile among every earlier day-long window on the chart."""
    if len(close) < 2 * per_day:
        return None
    returns = np.log(close.astype(float)).diff()
    vol = (returns.rolling(per_day).std() * math.sqrt(per_day)).to_numpy()
    if not np.isfinite(vol[-1]):
        return None
    return {"daily_pct": 100.0 * float(vol[-1]), "percentile": percentile_of(vol[:-1], vol[-1])}


def relative_volume(quote_volume: pd.Series, per_day: int, days: int = VOLUME_DAYS) -> Optional[Dict[str, float]]:
    """Traded value of the last 24 hours against the median of the `days` days before it."""
    v = quote_volume.astype(float).to_numpy()
    n_days = min(len(v) // per_day, days + 1)
    if n_days < 3:
        return None
    day_sums = v[::-1][: n_days * per_day].reshape(n_days, per_day).sum(axis=1)  # newest day first
    last, before = float(day_sums[0]), day_sums[1:]
    before = before[before > 0]
    if not len(before):
        return None
    typical = float(np.median(before))
    return {"last_24h_usd": last, "typical_24h_usd": typical, "ratio": last / typical}


def funding(rates: pd.Series, per_day: int, days: int = FUNDING_DAYS) -> Optional[Dict[str, float]]:
    """The latest settled funding rate and its percentile among the rates in force over `days`."""
    r = rates.astype(float).to_numpy()[-days * per_day:]
    r = r[np.isfinite(r)]
    if not len(r):
        return None
    return {"rate_pct": 100.0 * float(r[-1]), "mean_pct": 100.0 * float(r.mean()),
            "percentile": percentile_of(r[:-1], r[-1]) if len(r) > 1 else None, "days": days}


def signals_view(candles: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
    """Every supporting signal the charts in `candles` allow."""
    def chart(interval: str) -> Optional[pd.DataFrame]:
        frame = candles.get(interval)
        return frame if frame is not None and not frame.empty else None

    def col(frame: pd.DataFrame, name: str) -> pd.Series:
        return frame[f"{symbol}_{name}"]

    out: Dict[str, Any] = {"rsi": {}, "ranges": {}}
    for iv in RSI_CHARTS:
        frame = chart(iv)
        out["rsi"][iv] = rsi(col(frame, "close")) if frame is not None else None

    trend = chart(TREND_CHART)
    stack = ema_stack(col(trend, "close")) if trend is not None else None
    out["ema"] = {"interval": TREND_CHART, **stack} if stack else None

    for name, iv, days in RANGES:
        frame = chart(iv)
        out["ranges"][name] = (range_position(col(frame, "high"), col(frame, "low"), col(frame, "close"),
                                              days * _per_day(iv)) if frame is not None else None)

    vol = chart(VOL_CHART)
    out["volatility"] = realized_vol(col(vol, "close"), _per_day(VOL_CHART)) if vol is not None else None

    volume = chart(VOLUME_CHART)
    has_volume = volume is not None and f"{symbol}_quote_volume" in volume
    out["volume"] = relative_volume(col(volume, "quote_volume"), _per_day(VOLUME_CHART)) if has_volume else None

    fund = chart(FUNDING_CHART)
    has_funding = fund is not None and f"{symbol}_funding_rate" in fund
    out["funding"] = funding(col(fund, "funding_rate"), _per_day(FUNDING_CHART)) if has_funding else None
    return out
