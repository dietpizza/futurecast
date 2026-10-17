"""Command-line entry point: the futurecast web dashboard, and a backtest of the setup rules
(fair-value-gap setups or wedge breakouts)."""

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


def main() -> None:
    # `futurecast web ...` predates the dashboard being the only command; accept it as-is.
    if sys.argv[1:2] == ["web"]:
        del sys.argv[1]
    if sys.argv[1:2] == ["backtest"]:
        del sys.argv[1]
        typer.run(backtest)
        return
    typer.run(dashboard)


if __name__ == "__main__":
    main()
