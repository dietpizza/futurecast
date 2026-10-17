"""Backtest of the setup rules on historical candles (rules only, no models).

Every setup the rules find (fair-value-gap setups, features.setups, or wedge breakouts,
features.wedges) is replayed with what its orders really did (labels.setups, labels.wedges),
like a trader running one account:

- Only setups of the chosen grades are taken (A+ by default), one at a time: a setup is
  skipped while an earlier order is still waiting or its trade is still open.
- Each filled trade risks `risk_pct` of the account, so a result of r R (after its fees and
  slippage, also counted in R) moves equity by r * risk_pct percent, compounding. With
  `max_leverage` the position is at most that multiple of the account, so a trade whose stop is
  too tight for it risks less. Costs follow how the orders are placed (the spec's round_trip):
  for fair-value-gap setups maker fees in and out on a win, maker in and taker plus slippage
  out on a stop, maker in and taker out on a timeout; wedge entries are market orders (taker).
- Orders that never fill cost nothing. Setups whose outcome the data does not cover yet are
  left out.

Reports win rate, average and total R, profit factor, drawdown, losing streaks, monthly,
per-side and (for wedges) per-shape results, and the trade list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from deltarisk.features.setups import SetupSpec, close_times, find_setups
from deltarisk.features.wedges import WedgeSpec, find_wedge_setups
from deltarisk.labels.setups import break_even, setup_outcomes, target_key, target_multiples
from deltarisk.labels.wedges import wedge_outcomes


@dataclass(frozen=True)
class BacktestConfig:
    target: Union[float, str] = 3.0  # take-profit in R, or "measured" (wedges); one of the spec's targets
    grades: Tuple[str, ...] = ("A+",)
    risk_pct: float = 1.0  # account % risked per trade
    max_leverage: Optional[float] = None  # position value per account at most this; tighter stops then risk less


def _iso(ns: int) -> str:
    return pd.Timestamp(int(ns), tz="UTC").isoformat()


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for hit in mask:
        run = run + 1 if hit else 0
        best = max(best, run)
    return best


def take_trades(setups: pd.DataFrame, outcomes: pd.DataFrame, index: pd.DatetimeIndex,
                spec: Union[SetupSpec, WedgeSpec], cfg: BacktestConfig) -> Tuple[pd.DataFrame, int]:
    """The trades the rules would have taken, oldest first, and how many orders were placed.

    `index` is the entry-chart candle index the setups and outcomes were computed on.
    """
    key = target_key(cfg.target)
    multiples = target_multiples(setups, cfg.target)
    fill = outcomes["fill"].to_numpy()
    filled = fill == "filled"
    outcome = outcomes[f"outcome_{key}"].to_numpy()
    resolved = (fill != "pending") & ~(filled & (outcome == "open"))
    take = setups["grade"].isin(cfg.grades).to_numpy() & resolved
    idx = setups["idx"].to_numpy()
    end = outcomes[f"end_{key}"].to_numpy()
    busy_until = np.where(filled, end, outcomes["order_end"].to_numpy())
    ends = close_times(index, spec.ltf_minutes)

    rows: List[Dict[str, Any]] = []
    free_from, orders = -1, 0
    for i in np.flatnonzero(take):
        if idx[i] <= free_from:
            continue
        orders += 1
        free_from = busy_until[i]
        if not filled[i]:
            continue
        s, o = setups.iloc[i], outcomes.iloc[i]
        d = int(s["side"])
        R = abs(s["entry"] - s["stop"])
        T = float(multiples[i])
        gross = float(o[f"r_{key}"])
        cost_r = spec.round_trip(o[f"outcome_{key}"]) / float(s["risk"])
        leverage, account_risk = cfg.risk_pct / 100.0 / float(s["risk"]), cfg.risk_pct
        if cfg.max_leverage is not None and leverage > cfg.max_leverage:
            leverage, account_risk = cfg.max_leverage, cfg.max_leverage * float(s["risk"]) * 100.0
        held = int(o[f"end_{key}"]) - int(o["fill_idx"]) + (0 if spec.market_entry else 1)
        rows.append({
            "setup_at": _iso(ends[idx[i]]),
            "side": "long" if d > 0 else "short",
            **({"shape": s["shape"]} if "shape" in s.index else {}),
            "grade": s["grade"],
            "entry": float(s["entry"]),
            "stop": float(s["stop"]),
            "take_profit": float(s["entry"] + d * T * R),
            "target_r": T,
            "stop_pct": float(s["risk"]) * 100,
            "leverage": leverage,
            "account_risk_pct": account_risk,
            "filled_at": _iso(ends[int(o["fill_idx"])]),
            "closed_at": _iso(ends[int(o[f"end_{key}"])]),
            "outcome": o[f"outcome_{key}"],
            "gross_r": gross,
            "cost_r": cost_r,
            "net_r": gross - cost_r,
            "hold_hours": held * spec.ltf_minutes / 60.0,
        })
    return pd.DataFrame(rows), orders


def _breakdown(trades: pd.DataFrame, column: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for name, g in trades.groupby(column):
        out[name] = {"trades": int(len(g)), "win_rate": round(float((g["outcome"] == "won").mean()), 4),
                     "avg_net_r": round(float(g["net_r"].mean()), 4), "total_net_r": round(float(g["net_r"].sum()), 2)}
    return out


def run_backtest(setups: pd.DataFrame, outcomes: pd.DataFrame, index: pd.DatetimeIndex,
                 spec: Union[SetupSpec, WedgeSpec], cfg: BacktestConfig | None = None) -> Dict[str, Any]:
    """Summary, trade list (with the account after each trade), monthly, per-side and per-shape results
    of one backtest."""
    cfg = cfg or BacktestConfig()
    trades, orders = take_trades(setups, outcomes, index, spec, cfg)
    ends = close_times(index, spec.ltf_minutes)
    days = (ends[-1] - ends[0]) / 86_400e9 if len(ends) else 0.0
    summary: Dict[str, Any] = {
        "strategy": spec.strategy,
        "entry_interval": spec.entry_interval,
        "context_interval": getattr(spec, "context_interval", None),
        "target_r": cfg.target,
        "breakeven_r": getattr(spec, "breakeven_r", None),
        "grades": list(cfg.grades),
        "risk_pct": cfg.risk_pct,
        "max_leverage": cfg.max_leverage,
        "costs": spec.costs(),
        "start": _iso(ends[0]) if len(ends) else None,
        "end": _iso(ends[-1]) if len(ends) else None,
        "days": round(float(days), 1),
        "setups": int(len(setups)),
        "setups_in_grades": int(setups["grade"].isin(cfg.grades).sum()) if len(setups) else 0,
        "orders": orders,
        "trades": int(len(trades)),
    }
    if trades.empty:
        return {"summary": summary, "trades": trades, "monthly": [], "sides": {}, "shapes": {}}

    net = trades["net_r"].to_numpy()
    won = (trades["outcome"] == "won").to_numpy()
    risk = trades["stop_pct"].to_numpy() / 100.0
    account_risk = trades["account_risk_pct"].to_numpy() / 100.0
    equity = 100.0 * np.cumprod(1.0 + account_risk * net)
    cum_r = np.cumsum(net)
    peaks = np.maximum.accumulate(np.r_[100.0, equity])
    peaks_r = np.maximum.accumulate(np.r_[0.0, cum_r])
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    trades["equity_after"] = equity
    summary.update({
        "fill_rate": round(len(trades) / orders, 4) if orders else None,
        "wins": int(won.sum()),
        "losses": int((trades["outcome"] == "lost").sum()),
        "breakevens": int((trades["outcome"] == "breakeven").sum()),
        "timeouts": int((trades["outcome"] == "timeout").sum()),
        "win_rate": round(float(won.mean()), 4),
        # a win nets target minus its costs, a non-win counts as a stop: -1 minus the stop's costs
        "break_even_win_rate": round(float(np.mean(break_even(
            trades["target_r"].to_numpy(), 1.0, spec.round_trip("won") / risk, spec.round_trip("lost") / risk))), 4),
        "median_target_r": round(float(trades["target_r"].median()), 2),
        "avg_gross_r": round(float(trades["gross_r"].mean()), 4),
        "avg_cost_r": round(float(trades["cost_r"].mean()), 4),
        "avg_net_r": round(float(net.mean()), 4),
        "total_net_r": round(float(net.sum()), 2),
        "profit_factor": round(float(gains / losses), 3) if losses > 0 else None,
        "total_return_pct": round(float(equity[-1] - 100.0), 2),
        "max_drawdown_pct": round(float(np.max(1.0 - np.r_[100.0, equity] / peaks) * 100), 2),
        "max_drawdown_r": round(float(np.max(peaks_r - np.r_[0.0, cum_r])), 2),
        "longest_losing_streak": _longest_run(net < 0),
        "avg_hold_hours": round(float(trades["hold_hours"].mean()), 2),
        "trades_per_month": round(len(trades) / max(days / 30.44, 1e-9), 2),
        "median_stop_pct": round(float(trades["stop_pct"].median()), 4),
        "capped_trades": int((trades["account_risk_pct"] < cfg.risk_pct * (1 - 1e-9)).sum()),
    })

    month = pd.to_datetime(trades["closed_at"]).dt.strftime("%Y-%m")
    monthly = []
    for m, g in trades.groupby(month, sort=True):
        r = g["net_r"].to_numpy()
        monthly.append({"month": m, "trades": int(len(g)), "win_rate": round(float((g["outcome"] == "won").mean()), 4),
                        "net_r": round(float(r.sum()), 2),
                        "return_pct": round(float((np.prod(1.0 + g["account_risk_pct"].to_numpy() / 100.0 * r) - 1.0)
                                                  * 100), 2)})
    shapes = _breakdown(trades, "shape") if "shape" in trades else {}
    return {"summary": summary, "trades": trades, "monthly": monthly,
            "sides": _breakdown(trades, "side"), "shapes": shapes}


def backtest_candles(entry: pd.DataFrame, context: pd.DataFrame, spec: SetupSpec,
                     cfg: BacktestConfig | None = None) -> Dict[str, Any]:
    """Find the setups in raw candles (open/high/low/close, indexed by open time) and backtest them."""
    setups = find_setups(entry, context, spec)
    outcomes = setup_outcomes(setups, entry["high"], entry["low"], entry["close"], spec)
    return run_backtest(setups, outcomes, entry.index, spec, cfg)


def backtest_wedge_candles(candles: pd.DataFrame, spec: WedgeSpec, cfg: BacktestConfig | None = None) -> Dict[str, Any]:
    """Find the wedge setups in raw candles (open/high/low/close/volume, indexed by open time) and
    backtest them. The summary also carries how far the patterns got (`funnel`)."""
    setups = find_wedge_setups(candles, spec)
    outcomes = wedge_outcomes(setups, candles["high"], candles["low"], candles["close"], spec)
    result = run_backtest(setups, outcomes, candles.index, spec, cfg)
    result["summary"]["funnel"] = dict(setups.attrs["funnel"])
    return result


def _fmt(value: Any, suffix: str = "", digits: int = 2, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:+.{digits}f}{suffix}" if signed else f"{value:.{digits}f}{suffix}"


def fees_text(costs: Dict[str, float]) -> str:
    """SetupSpec.costs() in words: the per-leg rates and what a win and a stop cost round trip."""
    return (f"fees {costs['maker_pct']:g}% maker / {costs['taker_pct']:g}% taker, {costs['slippage_pct']:g}% stop "
            f"slippage ({costs['won_pct']:g}% round trip on a win, {costs['lost_pct']:g}% on a stop)")


def _target_text(target: Union[float, str]) -> str:
    return "measured-move" if target == "measured" else f"{target:g}R"


def format_report(symbol: str, result: Dict[str, Any]) -> str:
    """Plain-text report of one backtest."""
    s = result["summary"]
    if s["strategy"] == "wedge":
        charts = f"wedge breakouts on {s['entry_interval']}"
    else:
        charts = f"{s['entry_interval']} entries in {s['context_interval']} context"
    head = (f"{symbol} · {charts} · {'/'.join(s['grades'])} setups · {_target_text(s['target_r'])} target · "
            f"{s['risk_pct']:g}% risk per trade")
    if s["max_leverage"]:
        head += f", at most {s['max_leverage']:g}x leverage"
    lines = [head]
    if s["strategy"] == "wedge":
        lines.append("market entries, " + (f"stop to break-even at {s['breakeven_r']:g}R"
                                           if s["breakeven_r"] else "stop never moved"))
    taken = (f"{s['trades']} taken one at a time" if s["strategy"] == "wedge"
             else f"{s['orders']} orders placed one at a time, {s['trades']} filled")
    lines += [
        fees_text(s["costs"]),
        f"{s['start'][:10] if s['start'] else '?'} to {s['end'][:10] if s['end'] else '?'} ({s['days']:.0f} days): "
        f"{s['setups']} setups found, {s['setups_in_grades']} in grade, {taken}",
    ]
    if "funnel" in s:
        f = s["funnel"]
        lines.append(f"{f['wedges']} wedges, {f['breakouts']} broke out: {f['failed']} closed back inside, "
                     f"{f['expired']} found no retest in time, {f['triggered']} triggered")
    if not s["trades"]:
        lines.append("No filled trades.")
        return "\n".join(lines)
    ended = f"{s['wins']} won, {s['losses']} stopped"
    if s["breakevens"]:
        ended += f", {s['breakevens']} out at break-even"
    ended += f", {s['timeouts']} timed out"
    lines += [
        "",
        f"  Win rate          {s['win_rate'] * 100:.1f}%  ({ended})"
        f"   needs {s['break_even_win_rate'] * 100:.1f}% to break even after fees"
        + (" if every non-win were a full stop" if s["breakevens"] else ""),
        f"  Average trade     {_fmt(s['avg_net_r'], 'R', signed=True)} after fees  "
        f"({_fmt(s['avg_gross_r'], 'R', signed=True)} before; fees cost {_fmt(s['avg_cost_r'], 'R')} a trade, "
        f"median stop {_fmt(s['median_stop_pct'], '%')})",
        f"  Total             {_fmt(s['total_net_r'], 'R', signed=True)}  =  {_fmt(s['total_return_pct'], '%', signed=True)} "
        f"on the account, compounding",
        f"  Profit factor     {_fmt(s['profit_factor'])}",
        f"  Max drawdown      {_fmt(s['max_drawdown_pct'], '%')}  ({_fmt(s['max_drawdown_r'], 'R')}), "
        f"longest losing streak {s['longest_losing_streak']}",
        f"  Pace              {_fmt(s['trades_per_month'], ' trades a month', 1)}, "
        f"held {_fmt(s['avg_hold_hours'], ' h', 1)} on average"
        + ("" if s["strategy"] == "wedge" else f", {s['fill_rate'] * 100:.0f}% of orders filled"),
    ]
    if s["target_r"] == "measured":
        lines.append(f"  Target            the wedge's height, {_fmt(s['median_target_r'], 'R')} for the median trade")
    if s["capped_trades"]:
        lines.append(f"  Leverage          {s['capped_trades']} of {s['trades']} trades risked less than "
                     f"{s['risk_pct']:g}%: their stop was too tight for {s['max_leverage']:g}x")
    if s["trades"] < 100:
        lines.append(f"  Only {s['trades']} trades: too few to trust the rates above (aim for 100+).")
    for title, groups in (("Side", result["sides"]), ("Shape", result.get("shapes") or {})):
        if not groups:
            continue
        width = max(6, *(len(name) for name in groups))
        lines += ["", f"  {title:<{width}}  trades  win rate  avg R    total R"]
        for name, v in groups.items():
            lines.append(f"  {name:<{width}}  {v['trades']:>6}  {v['win_rate'] * 100:>7.1f}%  "
                         f"{v['avg_net_r']:>+6.2f}  {v['total_net_r']:>+8.2f}")
    if result["monthly"]:
        lines += ["", "  Month    trades  win rate   net R   account"]
        for m in result["monthly"]:
            lines.append(f"  {m['month']}  {m['trades']:>6}  {m['win_rate'] * 100:>7.1f}%  "
                         f"{m['net_r']:>+6.2f}  {m['return_pct']:>+7.2f}%")
    return "\n".join(lines)
