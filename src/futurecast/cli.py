"""Command-line entry point: the futurecast web dashboard, and a backtest of the setup rules."""

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


def backtest(
    symbol: str = typer.Argument(..., help="CoinDCX pair, e.g. B-BTC_USDT"),
    entry: str = typer.Option("15m", "--entry", help="Entry chart interval"),
    context: str = typer.Option("1h", "--context", help="Context chart interval (structure and fair value gaps)"),
    days: int = typer.Option(730, "--days", help="Days of history to replay"),
    target: float = typer.Option(3.0, "--target", help="Take-profit in multiples of the risk (R)"),
    grade: str = typer.Option("A+", "--grade", help="Setups to take: A+ or all"),
    risk: float = typer.Option(1.0, "--risk", help="Account % risked per trade"),
    maker: Optional[float] = typer.Option(None, "--maker", help="Maker fee in basis points, paid by the limit entry and "
                                          "take-profit (default: CoinDCX USDT-M futures, GST included)"),
    taker: Optional[float] = typer.Option(None, "--taker", help="Taker fee in basis points, paid by stop and timeout exits "
                                          "(default: CoinDCX USDT-M futures, GST included)"),
    slippage: Optional[float] = typer.Option(None, "--slippage", help="How far past its price a stop fills, in basis points"),
    csv: str = typer.Option("", "--csv", help="Write the trade list to this CSV file"),
) -> None:
    """Backtest the fair-value-gap setup rules on Binance history (downloads go to the candle cache)."""
    import pandas as pd

    from futurecast.backtest import BacktestConfig, backtest_candles, format_report
    from futurecast.config import load_config
    from futurecast.features.setups import SetupSpec
    from futurecast.publisher.multi_manager import _binance_client

    fees = {k: v for k, v in (("maker_bps", maker), ("taker_bps", taker), ("slippage_bps", slippage)) if v is not None}
    spec = SetupSpec.for_charts(entry, context, targets=tuple(sorted({2.0, 3.0, 4.0, target})), **fees)
    grades = ("A+", "B") if grade.lower() == "all" else (grade,)
    client = _binance_client(load_config().assets)
    candles = {}
    try:
        for interval, minutes, warmup in ((entry, spec.ltf_minutes, 0), (context, spec.htf_minutes, spec.htf_zone_max_age)):
            bars = days * 1440 // minutes + warmup
            typer.echo(f"Loading {bars:,} {interval} candles of {symbol}…", err=True)
            df, source = client.fetch_candle_history(symbol.upper(), interval, bars)
            if df.empty:
                raise typer.BadParameter(f"no Binance perpetual found for {symbol}")
            candles[interval] = df.set_index(pd.DatetimeIndex(df["timestamp"]))[["open", "high", "low", "close"]].astype(float)
    finally:
        client.close()

    result = backtest_candles(candles[entry], candles[context], spec, BacktestConfig(target=target, grades=grades, risk_pct=risk))
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
