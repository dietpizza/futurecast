"""Command-line entry point: the futurecast web dashboard, a backtest of the setup rules
(fair-value-gap setups or wedge breakouts), and the risk models' walk-forward check."""

from __future__ import annotations

import sys
from typing import Optional

import typer


def dashboard(
    host: str = typer.Option("0.0.0.0", "--host", help="Interface to bind the dashboard to (default: every interface)"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to serve the dashboard on"),
    offline: bool = typer.Option(False, "--offline", help="Use simulated candles instead of live Binance data"),
) -> None:
    """Start the futurecast web dashboard (`futurecast web` works too)."""
    from futurecast.publisher.web_server import run_web_server

    run_web_server(host=host, port=port, live=not offline)


def _target(value, default: float):
    """--target: a multiple of R ("3", "2.5") or "measured"."""
    if value is None:
        return default
    if str(value).lower() == "measured":
        return "measured"
    try:
        return float(value)
    except ValueError:
        raise typer.BadParameter(f"--target must be a number of R or 'measured', not {value!r}")


def backtest(
    symbol: str = typer.Argument(..., help="CoinDCX pair, e.g. B-BTC_USDT"),
    strategy: str = typer.Option("fvg", "--strategy", help="fvg (fair-value-gap setups on an entry chart in a "
                                 "context chart) or wedge (wedge breakouts with a retest, on one chart)"),
    entry: Optional[str] = typer.Option(None, "--entry", help="Entry chart interval (default 15m; 1h for wedges)"),
    context: str = typer.Option("1h", "--context", help="Context chart interval (structure and fair value gaps; fvg only)"),
    days: int = typer.Option(730, "--days", help="Days of history to replay"),
    target: Optional[str] = typer.Option(None, "--target", help="Take-profit in multiples of the risk (R), or 'measured' "
                                         "for a wedge's height (default 3; 2 for wedges)"),
    grade: str = typer.Option("A+", "--grade", help="Setups to take: A+ or all"),
    risk: float = typer.Option(1.0, "--risk", help="Account % risked per trade"),
    max_leverage: Optional[float] = typer.Option(None, "--max-leverage", help="Largest position as a multiple of the "
                                                 "account; tighter stops then risk less (default 5 for wedges, "
                                                 "none for fvg; 0 for none)"),
    breakeven: bool = typer.Option(True, "--breakeven/--no-breakeven", help="Wedges: move the stop to break-even plus "
                                   "fees once price gets 1.5R"),
    maker: Optional[float] = typer.Option(None, "--maker", help="Maker fee in basis points, paid by limit orders "
                                          "(default: CoinDCX USDT-M futures, GST included)"),
    taker: Optional[float] = typer.Option(None, "--taker", help="Taker fee in basis points, paid by market orders and "
                                          "stops (default: CoinDCX USDT-M futures, GST included)"),
    slippage: Optional[float] = typer.Option(None, "--slippage", help="How far past its price a stop fills, in basis points"),
    csv: str = typer.Option("", "--csv", help="Write the trade list to this CSV file"),
) -> None:
    """Backtest the setup rules on Binance history (downloads go to the candle cache)."""
    import pandas as pd

    from futurecast.backtest import BacktestConfig, backtest_candles, backtest_wedge_candles, format_report
    from futurecast.config import load_config
    from futurecast.features.setups import SetupSpec
    from futurecast.features.wedges import WedgeSpec
    from futurecast.publisher.multi_manager import _binance_client

    strategy = strategy.lower()
    if strategy not in ("fvg", "wedge"):
        raise typer.BadParameter(f"--strategy must be fvg or wedge, not {strategy!r}")
    wedge = strategy == "wedge"
    entry = entry or ("1h" if wedge else "15m")
    target = _target(target, 2.0 if wedge else 3.0)
    fees = {k: v for k, v in (("maker_bps", maker), ("taker_bps", taker), ("slippage_bps", slippage)) if v is not None}
    if wedge:
        fixed = {2.0, 2.5} | ({target} if isinstance(target, float) else set())
        spec = WedgeSpec.for_chart(entry, targets=("measured", *sorted(fixed)),
                                   breakeven_r=WedgeSpec.breakeven_r if breakeven else None, **fees)
        charts = ((entry, spec.minutes, 0),)
        max_leverage = 5.0 if max_leverage is None else max_leverage
    else:
        if target == "measured":
            raise typer.BadParameter("a measured-move target needs --strategy wedge")
        spec = SetupSpec.for_charts(entry, context, targets=tuple(sorted({2.0, 3.0, 4.0, target})), **fees)
        charts = ((entry, spec.ltf_minutes, 0), (context, spec.htf_minutes, spec.htf_zone_max_age))
    grades = ("A+", "B") if grade.lower() == "all" else (grade,)
    cfg = BacktestConfig(target=target, grades=grades, risk_pct=risk, max_leverage=max_leverage or None)
    client = _binance_client(load_config().assets)
    candles = {}
    try:
        for interval, minutes, warmup in charts:
            bars = days * 1440 // minutes + warmup
            typer.echo(f"Loading {bars:,} {interval} candles of {symbol}…", err=True)
            df, source = client.fetch_candle_history(symbol.upper(), interval, bars)
            if df.empty:
                raise typer.BadParameter(f"no Binance perpetual found for {symbol}")
            candles[interval] = (df.set_index(pd.DatetimeIndex(df["timestamp"]))
                                 [["open", "high", "low", "close", "volume"]].astype(float))
    finally:
        client.close()

    if wedge:
        result = backtest_wedge_candles(candles[entry], spec, cfg)
    else:
        result = backtest_candles(candles[entry], candles[context], spec, cfg)
    typer.echo(format_report(symbol.upper(), result))
    if csv:
        result["trades"].to_csv(csv, index=False)
        typer.echo(f"\nTrades written to {csv}")


def _pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}%"


def risk(
    symbol: str = typer.Argument(..., help="CoinDCX pair, e.g. B-SOL_USDT"),
    window: Optional[str] = typer.Option(None, "--window", help="One window (5m, 15m, 30m, 1h, 2h, 3h, 4h, 8h); "
                                         "default all"),
    stop: float = typer.Option(5.0, "--stop", help="Stop loss in % of margin (the PnL % CoinDCX shows)"),
    leverage: Optional[float] = typer.Option(None, "--leverage", help="Leverage (default by tier: megacap 20, "
                                             "altcoin 10, shitcoin 3)"),
    tier: Optional[str] = typer.Option(None, "--tier", help="megacap, altcoin or shitcoin (default from the coin)"),
    calibration: bool = typer.Option(False, "--calibration", help="Also print the stop-hit calibration buckets"),
) -> None:
    """Walk-forward check of the risk models on Binance history, and today's readings."""
    import pandas as pd

    from futurecast.config import load_config
    from futurecast.ingestion.factory import get_symbol_sector, resolve_benchmark
    from futurecast.publisher.multi_manager import HISTORY_BARS, SymbolWorker, _binance_client
    from futurecast.stats.regime import WINDOW_BY_NAME, WINDOWS, RegimeSpec, regime_frame
    from futurecast.stats.risk import (
        SIDES, TIERS, RiskSpec, WindowRiskModel, coin_tier, latest_reading, liquidation_move, stop_move,
        walk_forward, window_data,
    )

    symbol = symbol.upper()
    if window is not None and window not in WINDOW_BY_NAME:
        raise typer.BadParameter(f"--window must be one of {', '.join(WINDOW_BY_NAME)}")
    windows = [WINDOW_BY_NAME[window]] if window else list(WINDOWS)
    charts = [iv for iv in HISTORY_BARS if iv == "1h" or any(w.interval == iv for w in windows)]  # 1h: funding
    worker = SymbolWorker(symbol, resolve_benchmark(symbol), get_symbol_sector(symbol), load_config())
    client = _binance_client(worker.config.assets)
    candles = {}
    try:
        if client.resolve(symbol) is None:
            raise typer.BadParameter(f"no Binance perpetual found for {symbol}")
        for iv in charts:
            typer.echo(f"Loading {HISTORY_BARS[iv]:,} {iv} candles of {symbol} and {worker.benchmark}…", err=True)
            candles[iv], _ = worker._fetch_aligned(client, iv, HISTORY_BARS[iv])
            if candles[iv].empty:
                raise typer.BadParameter(f"no closed {iv} candles for {symbol}")
    finally:
        client.close()

    hourly = candles["1h"]
    volume_24h = float(hourly[f"{symbol}_quote_volume"].tail(24).sum())
    tier_name = tier.lower() if tier else coin_tier(symbol, volume_24h)
    if tier_name not in TIERS:
        raise typer.BadParameter(f"--tier must be one of {', '.join(TIERS)}")
    lev = leverage or TIERS[tier_name].leverage
    stop_px, liq_px = stop_move(stop, lev), liquidation_move(lev, TIERS[tier_name].mmr)
    spec, regime_spec = RiskSpec(), RegimeSpec()

    typer.echo(f"\n{symbol} · {tier_name} · {lev:g}x · 24h volume ${volume_24h / 1e6:,.0f}M · benchmark "
               f"{worker.benchmark}")
    typer.echo(f"Stop −{stop:g}% of margin = {_pct(stop_px, 2)} price · liquidation ≈ {_pct(liq_px, 2)} price")
    for iv, df in candles.items():
        typer.echo(f"  {iv}: {len(df):,} candles, {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d %H:%M} UTC")

    checks, today, labels = {}, {}, {}
    for w in windows:
        typer.echo(f"Checking {w.name}…", err=True)
        data = window_data(candles, symbol, w, worker.benchmark, regime_spec)
        checks[w.name] = walk_forward(data, stop_px, liq_px, spec)
        model = WindowRiskModel(spec).fit(data, data.trainable)
        choice = {s: checks[w.name].get(s, {}).get("model", "har") for s in SIDES}
        today[w.name] = latest_reading(model, data, choice, stop_px, liq_px, lev)
        reg = regime_frame(candles[w.interval], symbol, w, regime_spec)
        labels[w.name] = (reg, None if pd.isna(reg["state"].iloc[-1]) else reg.iloc[-1])

    typer.echo(f"\nWalk-forward check: train on the first {spec.train_share:.0%} (expanding), "
               f"{spec.folds} test blocks after it, {spec.gap_hours:g}h gap. Coverage = share of test "
               f"windows worse than the predicted 1-in-10 bad case (want {spec.coverage_band[0]:.0%}–"
               f"{spec.coverage_band[1]:.0%}). Pinball loss in price %, LightGBM ships if ≥ "
               f"{spec.min_gain:.0%} lower.")
    head = (f"{'window':<7}{'side':<6}{'tests':>7}  {'cover HAR':>9} {'LGBM':>6}  {'pinball HAR':>11} {'LGBM':>7} "
            f"{'gain':>6}  {'model':<5}  {'stop-hit pred':>13} {'actual':>7}  {'brier':>5}  check")
    typer.echo(head)
    typer.echo("─" * len(head))
    for name, c in checks.items():
        for side in SIDES:
            if side not in c:
                typer.echo(f"{name:<7}{side:<6}  not enough history")
                continue
            s = c[side]
            m = s["model"]
            typer.echo(f"{name:<7}{side:<6}{c['tests']:>7,}  {_pct(s['har']['coverage']):>9} "
                       f"{_pct(s['lgbm']['coverage']):>6}  {s['har']['pinball_pct']:>11.3f} "
                       f"{s['lgbm']['pinball_pct']:>7.3f} {s['gain'] * 100:>+5.1f}%  {m:<5}  "
                       f"{_pct(s[m]['stop_hit_predicted']):>13} {_pct(s['stop_hit_rate']):>7}  "
                       f"{s[m]['brier']:>5.3f}  {'✓' if s['checked'] else '✗ —'}")
    if calibration:
        typer.echo("\nStop-hit calibration (shipped model): bucket of predicted chance → mean predicted / "
                   "actual (test windows)")
        for name, c in checks.items():
            for side in SIDES:
                if side not in c:
                    continue
                cells = " · ".join(f"{b['from']:.0%}–{b['to']:.0%} {b['predicted']:.0%}/{b['actual']:.0%} "
                                   f"({b['n']:,})" for b in c[side]["calibration"])
                typer.echo(f"{name:<7}{side:<6}{cells}")

    typer.echo("\nLabels over the history (share of candles)")
    typer.echo(f"{'window':<7}{'bullish':>8}{'neutral':>8}{'bearish':>8}   {'trending':>8}{'choppy':>8}"
               f"{'quiet':>8}{'wild':>8}")
    for name, (reg, _) in labels.items():
        mom = reg["momentum_label"].value_counts(normalize=True)
        st = reg["state"].value_counts(normalize=True)
        typer.echo(f"{name:<7}" + "".join(f"{mom.get(k, 0):>8.0%}" for k in ("bullish", "neutral", "bearish"))
                   + "   " + "".join(f"{st.get(k, 0):>8.0%}" for k in ("trending", "choppy", "quiet", "wild")))

    last = max(t["at"] for t in today.values())
    typer.echo(f"\nToday (models trained on all history; each chart's last closed candle, the newest opened {last:%Y-%m-%d %H:%M} UTC)")
    head = (f"{'window':<7}{'momentum':>13} {'state':<9}{'vol':<7}  {'side':<6}{'model':<6}{'bad case':>9} "
            f"{'margin':>8} {'liq':>6} {'score':>6}")
    typer.echo(head)
    typer.echo("─" * len(head))
    for name, t in today.items():
        _, reg = labels[name]
        regime = (f"{reg['momentum']:>+5.0f} {reg['momentum_label']:<7} {reg['state']:<9}{reg['volatility_label']:<7}"
                  if reg is not None else f"{'—':>13} {'—':<9}{'—':<7}")
        for i, side in enumerate(SIDES):
            r = t[side]
            checked = checks[name].get(side, {}).get("checked", False)
            score = f"{r['score']}" if checked else f"({r['score']})"
            typer.echo(f"{name if i == 0 else '':<7}{regime if i == 0 else ' ' * 31}  {side:<6}{r['model']:<6}"
                       f"{r['bad_case_pct']:>8.2f}% {r['bad_case_margin_pct']:>7.1f}% "
                       f"{_pct(r['liquidation_chance'], 2):>6} {score:>6}")
    typer.echo("Score = the chance in % that the stop is hit within the window; in brackets where the check "
               "failed (the page would show —).")


def main() -> None:
    # `futurecast web ...` predates the dashboard being the only command; accept it as-is.
    if sys.argv[1:2] == ["web"]:
        del sys.argv[1]
    if sys.argv[1:2] == ["backtest"]:
        del sys.argv[1]
        typer.run(backtest)
        return
    if sys.argv[1:2] == ["risk"]:
        del sys.argv[1]
        typer.run(risk)
        return
    typer.run(dashboard)


if __name__ == "__main__":
    main()
