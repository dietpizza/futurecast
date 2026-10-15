"""Backtest of the fair-value-gap setup rules on historical candles (rules only, no models).

Every setup the rules find (features.setups) is replayed with what its orders really did
(labels.setups), like a trader running one account:

- Only setups of the chosen grades are taken (A+ by default), one at a time: a setup is
  skipped while an earlier order is still waiting or its trade is still open.
- Each filled trade risks `risk_pct` of the account, so a result of r R (after its fees and
  slippage, also counted in R) moves equity by r * risk_pct percent, compounding. Costs follow
  how the orders are placed (SetupSpec.round_trip): maker fees in and out on a win, maker in
  and taker plus slippage out on a stop, maker in and taker out on a timeout.
- Orders that never fill cost nothing. Setups whose outcome the data does not cover yet are
  left out.

Reports win rate, average and total R, profit factor, drawdown, losing streaks, monthly and
per-side results, and the trade list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from futurecast.features.setups import SetupSpec, close_times, find_setups
from futurecast.labels.setups import break_even, setup_outcomes, target_key


@dataclass(frozen=True)
class BacktestConfig:
    target: float = 3.0  # take-profit in R (one of SetupSpec.targets)
    grades: Tuple[str, ...] = ("A+",)
    risk_pct: float = 1.0  # account % risked per trade
    curve_points: int = 300  # equity-curve points kept for the dashboard


def _iso(ns: int) -> str:
    return pd.Timestamp(int(ns), tz="UTC").isoformat()


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for hit in mask:
        run = run + 1 if hit else 0
        best = max(best, run)
    return best


def take_trades(setups: pd.DataFrame, outcomes: pd.DataFrame, index: pd.DatetimeIndex, spec: SetupSpec,
                cfg: BacktestConfig) -> Tuple[pd.DataFrame, int]:
    """The trades the rules would have taken, oldest first, and how many orders were placed.

    `index` is the entry-chart candle index the setups and outcomes were computed on.
    """
    key = target_key(cfg.target)
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
        gross = float(o[f"r_{key}"])
        cost_r = spec.round_trip(o[f"outcome_{key}"]) / float(s["risk"])
        rows.append({
            "setup_at": _iso(ends[idx[i]]),
            "side": "long" if d > 0 else "short",
            "grade": s["grade"],
            "entry": float(s["entry"]),
            "stop": float(s["stop"]),
            "take_profit": float(s["entry"] + d * cfg.target * R),
            "stop_pct": float(s["risk"]) * 100,
            "filled_at": _iso(ends[int(o["fill_idx"])]),
            "closed_at": _iso(ends[int(o[f"end_{key}"])]),
            "outcome": o[f"outcome_{key}"],
            "gross_r": gross,
            "cost_r": cost_r,
            "net_r": gross - cost_r,
            "hold_hours": (int(o[f"end_{key}"]) - int(o["fill_idx"]) + 1) * spec.ltf_minutes / 60.0,
        })
    return pd.DataFrame(rows), orders


def run_backtest(setups: pd.DataFrame, outcomes: pd.DataFrame, index: pd.DatetimeIndex, spec: SetupSpec,
                 cfg: BacktestConfig | None = None) -> Dict[str, Any]:
    """Summary, trade list, equity curve, monthly and per-side results of one backtest."""
    cfg = cfg or BacktestConfig()
    trades, orders = take_trades(setups, outcomes, index, spec, cfg)
    ends = close_times(index, spec.ltf_minutes)
    days = (ends[-1] - ends[0]) / 86_400e9 if len(ends) else 0.0
    summary: Dict[str, Any] = {
        "entry_interval": spec.entry_interval,
        "context_interval": spec.context_interval,
        "target_r": cfg.target,
        "grades": list(cfg.grades),
        "risk_pct": cfg.risk_pct,
        "costs": spec.costs(),
        "start": _iso(ends[0]) if len(ends) else None,
        "end": _iso(ends[-1]) if len(ends) else None,
        "days": round(float(days), 1),
        "setups": int(len(setups)),
        "setups_in_grades": int(setups["grade"].isin(cfg.grades).sum()) if len(setups) else 0,
        "orders": orders,
        "trades": int(len(trades)),
    }
    curve = [{"time": summary["start"], "equity": 100.0, "net_r": 0.0}]
    if trades.empty:
        return {"summary": summary, "trades": trades, "equity": curve, "monthly": [], "sides": {}}

    net = trades["net_r"].to_numpy()
    won = (trades["outcome"] == "won").to_numpy()
    risk = trades["stop_pct"].to_numpy() / 100.0
    equity = 100.0 * np.cumprod(1.0 + cfg.risk_pct / 100.0 * net)
    cum_r = np.cumsum(net)
    peaks = np.maximum.accumulate(np.r_[100.0, equity])
    peaks_r = np.maximum.accumulate(np.r_[0.0, cum_r])
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    trades["equity_after"] = equity
    summary.update({
        "fill_rate": round(len(trades) / orders, 4) if orders else None,
        "wins": int(won.sum()),
        "losses": int((trades["outcome"] == "lost").sum()),
        "timeouts": int((trades["outcome"] == "timeout").sum()),
        "win_rate": round(float(won.mean()), 4),
        # a win nets target minus its costs, a non-win counts as a stop: -1 minus the stop's costs
        "break_even_win_rate": round(float(np.mean(break_even(
            cfg.target, 1.0, spec.round_trip("won") / risk, spec.round_trip("lost") / risk))), 4),
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
    })

    keep = np.unique(np.linspace(0, len(trades) - 1, min(cfg.curve_points, len(trades))).round().astype(int))
    curve += [{"time": trades["closed_at"].iat[k], "equity": round(float(equity[k]), 3),
               "net_r": round(float(cum_r[k]), 3)} for k in keep]

    month = pd.to_datetime(trades["closed_at"]).dt.strftime("%Y-%m")
    monthly = []
    for m, g in trades.groupby(month, sort=True):
        r = g["net_r"].to_numpy()
        monthly.append({"month": m, "trades": int(len(g)), "win_rate": round(float((g["outcome"] == "won").mean()), 4),
                        "net_r": round(float(r.sum()), 2),
                        "return_pct": round(float((np.prod(1.0 + cfg.risk_pct / 100.0 * r) - 1.0) * 100), 2)})
    sides = {}
    for side, g in trades.groupby("side"):
        sides[side] = {"trades": int(len(g)), "win_rate": round(float((g["outcome"] == "won").mean()), 4),
                       "avg_net_r": round(float(g["net_r"].mean()), 4), "total_net_r": round(float(g["net_r"].sum()), 2)}
    return {"summary": summary, "trades": trades, "equity": curve, "monthly": monthly, "sides": sides}


def backtest_candles(entry: pd.DataFrame, context: pd.DataFrame, spec: SetupSpec,
                     cfg: BacktestConfig | None = None) -> Dict[str, Any]:
    """Find the setups in raw candles (open/high/low/close, indexed by open time) and backtest them."""
    setups = find_setups(entry, context, spec)
    outcomes = setup_outcomes(setups, entry["high"], entry["low"], entry["close"], spec)
    return run_backtest(setups, outcomes, entry.index, spec, cfg)


def dashboard_view(result: Dict[str, Any]) -> Dict[str, Any]:
    """A backtest without its trade list, for the snapshot sent to browsers."""
    return {k: v for k, v in result.items() if k != "trades"}


def _fmt(value: Any, suffix: str = "", digits: int = 2, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:+.{digits}f}{suffix}" if signed else f"{value:.{digits}f}{suffix}"


def fees_text(costs: Dict[str, float]) -> str:
    """SetupSpec.costs() in words: the per-leg rates and what a win and a stop cost round trip."""
    return (f"fees {costs['maker_pct']:g}% maker / {costs['taker_pct']:g}% taker, {costs['slippage_pct']:g}% stop "
            f"slippage ({costs['won_pct']:g}% round trip on a win, {costs['lost_pct']:g}% on a stop)")


def format_report(symbol: str, result: Dict[str, Any]) -> str:
    """Plain-text report of one backtest."""
    s = result["summary"]
    lines = [
        f"{symbol} · {s['entry_interval']} entries in {s['context_interval']} context · "
        f"{'/'.join(s['grades'])} setups · {s['target_r']:g}R target · {s['risk_pct']:g}% risk per trade",
        fees_text(s["costs"]),
        f"{s['start'][:10] if s['start'] else '?'} to {s['end'][:10] if s['end'] else '?'} ({s['days']:.0f} days): "
        f"{s['setups']} setups found, {s['setups_in_grades']} in grade, {s['orders']} orders placed one at a time, "
        f"{s['trades']} filled",
    ]
    if not s["trades"]:
        lines.append("No filled trades.")
        return "\n".join(lines)
    lines += [
        "",
        f"  Win rate          {s['win_rate'] * 100:.1f}%  ({s['wins']} won, {s['losses']} stopped, {s['timeouts']} timed out)"
        f"   needs {s['break_even_win_rate'] * 100:.1f}% to break even after fees",
        f"  Average trade     {_fmt(s['avg_net_r'], 'R', signed=True)} after fees  "
        f"({_fmt(s['avg_gross_r'], 'R', signed=True)} before; fees cost {_fmt(s['avg_cost_r'], 'R')} a trade, "
        f"median stop {_fmt(s['median_stop_pct'], '%')})",
        f"  Total             {_fmt(s['total_net_r'], 'R', signed=True)}  =  {_fmt(s['total_return_pct'], '%', signed=True)} "
        f"on the account, compounding",
        f"  Profit factor     {_fmt(s['profit_factor'])}",
        f"  Max drawdown      {_fmt(s['max_drawdown_pct'], '%')}  ({_fmt(s['max_drawdown_r'], 'R')}), "
        f"longest losing streak {s['longest_losing_streak']}",
        f"  Pace              {_fmt(s['trades_per_month'], ' trades a month', 1)}, "
        f"held {_fmt(s['avg_hold_hours'], ' h', 1)} on average, {s['fill_rate'] * 100:.0f}% of orders filled",
    ]
    if s["trades"] < 100:
        lines.append(f"  Only {s['trades']} trades: too few to trust the rates above (aim for 100+).")
    if result["sides"]:
        lines += ["", "  Side    trades  win rate  avg R    total R"]
        for side, v in result["sides"].items():
            lines.append(f"  {side:<6}  {v['trades']:>6}  {v['win_rate'] * 100:>7.1f}%  "
                         f"{v['avg_net_r']:>+6.2f}  {v['total_net_r']:>+8.2f}")
    if result["monthly"]:
        lines += ["", "  Month    trades  win rate   net R   account"]
        for m in result["monthly"]:
            lines.append(f"  {m['month']}  {m['trades']:>6}  {m['win_rate'] * 100:>7.1f}%  "
                         f"{m['net_r']:>+6.2f}  {m['return_pct']:>+7.2f}%")
    return "\n".join(lines)
