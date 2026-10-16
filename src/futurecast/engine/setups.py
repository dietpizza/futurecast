"""Setups on aligned candle frames: the training dataset, and the live view for the dashboard.

The live view needs no trained model: market structure, fair value gaps and setups come from
exact rules (features.structure, features.setups). A trained SetupModel adds win chances,
break-even and the evidence behind them; without one those fields stay empty.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from futurecast.features.setups import SetupSpec, close_times, find_setups, setup_features
from futurecast.features.structure import KIND_NAMES, UP, active_gaps, fair_value_gaps, market_structure
from futurecast.labels.setups import setup_outcomes, target_key
from futurecast.models.setup_model import SetupModel
from futurecast.models.trade_risk import TradeRiskConfig, max_safe_leverage

TREND_NAMES = {1: "up", -1: "down", 0: "none"}
STATE_BY_FILL = {"pending": "waiting", "expired": "expired", "cancelled": "cancelled"}
STATE_BY_OUTCOME = {"open": "in_trade", "won": "won", "lost": "lost", "timeout": "timeout"}
MAX_ZONES = 3  # gaps listed per direction, nearest to price first


def ohlc(aligned: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """The coin's own open/high/low/close from an aligned cross-asset frame."""
    cols = ("open", "high", "low", "close")
    return aligned[[f"{symbol}_{c}" for c in cols]].set_axis(list(cols), axis=1)


def setup_dataset(aligned_ltf: pd.DataFrame, aligned_htf: pd.DataFrame, market_features: pd.DataFrame,
                  symbol: str, spec: SetupSpec) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every setup in the history, its outcome, and its model inputs (setup + market features)."""
    ltf, htf = ohlc(aligned_ltf, symbol), ohlc(aligned_htf, symbol)
    setups = find_setups(ltf, htf, spec)
    outcomes = setup_outcomes(setups, ltf["high"], ltf["low"], ltf["close"], spec)
    X = setup_features(setups, ltf["close"], spec).join(market_features)
    return setups, outcomes, X


def _iso(ts_ns: int) -> str:
    return pd.Timestamp(int(ts_ns), tz="UTC").isoformat()


def _structure_view(candles: pd.DataFrame, structure: pd.DataFrame, gaps: pd.DataFrame,
                    interval: str, minutes: int) -> Dict[str, Any]:
    """Trend, latest break, swing levels and the active gaps nearest to price on one chart."""
    last = len(candles) - 1
    ends = close_times(candles.index, minutes)
    price = float(candles["close"].iat[last])
    lb = int(structure["last_break"].iat[last])
    last_break = None
    if lb >= 0:
        last_break = {
            "kind": KIND_NAMES[int(structure["break_kind"].iat[lb])],
            "direction": "up" if structure["break_dir"].iat[lb] == UP else "down",
            "level": float(structure["break_level"].iat[lb]),
            "time": _iso(ends[lb]),
            "candles_ago": last - lb,
        }
    zones = []
    live = active_gaps(gaps, last)
    for direction in (1, -1):
        g = live[live["direction"] == direction]
        mid = (g["top"] + g["bottom"]) / 2.0
        for _, z in g.loc[(mid - price).abs().sort_values().index[:MAX_ZONES]].iterrows():
            zones.append({
                "direction": "bullish" if direction == UP else "bearish",
                "top": float(z["top"]), "bottom": float(z["bottom"]), "mid": float((z["top"] + z["bottom"]) / 2.0),
                "formed": _iso(ends[int(z["idx"])]), "age_candles": last - int(z["idx"]),
                "touched": 0 <= int(z["touched"]) <= last,
                "contains_price": bool(z["bottom"] <= price <= z["top"]),
            })
    def finite(v: float) -> Optional[float]:
        return float(v) if np.isfinite(v) else None

    return {
        "interval": interval,
        "trend": TREND_NAMES[int(structure["trend"].iat[last])],
        "last_break": last_break,
        "swing_high": finite(structure["swing_high"].iat[last]),
        "swing_low": finite(structure["swing_low"].iat[last]),
        "gaps": zones,
    }


def _setup_view(setup: pd.Series, outcome: pd.Series, score: Optional[Dict[str, Any]], spec: SetupSpec,
                ends: np.ndarray, last: int) -> Dict[str, Any]:
    """One setup as the dashboard shows it, with its orders' state on the candles so far."""
    d = int(setup["side"])
    entry, stop = float(setup["entry"]), float(setup["stop"])
    R = abs(entry - stop)
    s = int(setup["idx"])
    pick = score["targets"][score["pick"]] if score and score["pick"] is not None else None
    target = pick["target_r"] if pick else spec.targets[0]
    fill = outcome["fill"]
    state = STATE_BY_FILL.get(fill) or STATE_BY_OUTCOME.get(outcome[f"outcome_{target_key(target)}"], "in_trade")
    fill_idx = int(outcome["fill_idx"])

    targets = []
    for T in spec.targets:
        row = {"target_r": T, "price": entry + d * T * R}
        if score:
            row.update(next(t for t in score["targets"] if t["target_r"] == T))
        targets.append(row)
    view = {
        "side": "long" if d == UP else "short",
        "time": _iso(ends[s]),
        "candles_ago": last - s,
        "grade": setup["grade"],
        "checklist": {"htf_trend": bool(setup["htf_bias"]), "htf_zone": bool(setup["htf_zone"]),
                      "ltf_choch": True, "ltf_gap": True},
        "state": state,
        "entry": entry,
        "stop": stop,
        "risk_pct": float(setup["risk"]) * 100,
        "choch_level": float(setup["break_level"]),
        "gap": {"top": float(setup["gap_top"]), "bottom": float(setup["gap_bottom"])},
        "zone": ({"top": float(setup["zone_top"]), "bottom": float(setup["zone_bottom"])}
                 if setup["htf_zone"] else None),
        "order_expires": _iso(ends[s] + spec.fill_window * 60_000_000_000 * spec.ltf_minutes),
        "filled_at": _iso(ends[fill_idx]) if fill_idx >= 0 else None,
        "target_r": target,
        "take_profit": entry + d * target * R,
        "targets": targets,
        # Display-only sizing: position value per account for a 1% account risk, and the highest
        # leverage whose liquidation stays beyond the stop
        "size_for_1pct_risk": 0.01 / float(setup["risk"]),
        "max_leverage": max_safe_leverage(float(setup["risk"]), TradeRiskConfig()),
    }
    for field in ("probability", "probability_source", "model_probability", "rules_probability", "per_model",
                  "break_even", "expected_r", "status", "validated"):
        view[field] = pick.get(field) if pick else None
    return view


def live_view(aligned_ltf: pd.DataFrame, aligned_htf: pd.DataFrame, symbol: str,
              market_features: Optional[pd.DataFrame], model: Optional[SetupModel],
              spec: Optional[SetupSpec] = None) -> Dict[str, Any]:
    """Structure on both charts, the latest setup per side and the current signal.

    `market_features` are the model features of the entry-chart candles in `aligned_ltf` (needed only with a model).
    """
    spec = model.spec if model is not None else (spec or SetupSpec())
    ltf, htf = ohlc(aligned_ltf, symbol), ohlc(aligned_htf, symbol)
    lh, ll, lc = (ltf[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    hh, hl, hc = (htf[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    ls, lg = market_structure(lh, ll, lc, spec.ltf_swing_len), fair_value_gaps(lh, ll, lc, spec.ltf_gap_max_age)
    hs, hg = market_structure(hh, hl, hc, spec.htf_swing_len), fair_value_gaps(hh, hl, hc, spec.htf_zone_max_age)

    structure = {
        "htf": _structure_view(htf, hs, hg, spec.context_interval, spec.htf_minutes),
        "ltf": _structure_view(ltf, ls, lg, spec.entry_interval, spec.ltf_minutes),
    }
    setups = find_setups(ltf, htf, spec, ltf_structure=ls, htf_structure=hs, htf_gaps=hg)
    latest = {}
    for d in (1, -1):
        side = setups[setups["side"] == d]
        if len(side):
            latest[d] = side.iloc[-1:]
    out: Dict[str, Any] = {"as_of": _iso(close_times(ltf.index, spec.ltf_minutes)[-1]), "signal": "NEUTRAL",
                           "entry_interval": spec.entry_interval, "context_interval": spec.context_interval,
                           "cancel_r": spec.cancel_r,
                           "costs": spec.costs(), "long": None, "short": None}
    if not latest:
        return {"structure": structure, "setups": out}

    picked = pd.concat(list(latest.values()))
    outcomes = setup_outcomes(picked, ltf["high"], ltf["low"], ltf["close"], spec)
    scores = None
    if model is not None and market_features is not None:
        X = setup_features(picked, ltf["close"], spec).join(market_features)
        scores = model.score(X, picked)
    ends = close_times(ltf.index, spec.ltf_minutes)
    for i in range(len(picked)):
        view = _setup_view(picked.iloc[i], outcomes.iloc[i], scores[i] if scores else None, spec, ends, len(ltf) - 1)
        out[view["side"]] = view

    # A signal needs a waiting A+ order whose served chance clears break-even and whose replay made money
    ready = [v for v in (out["long"], out["short"])
             if v and v["state"] == "waiting" and v["grade"] == "A+" and v["validated"]
             and v["probability"] is not None and v["probability"] >= v["break_even"]]
    if ready:
        out["signal"] = max(ready, key=lambda v: v["expected_r"])["side"].upper()
    return {"structure": structure, "setups": out}
