"""Command-line entry point: starts the futurecast web dashboard."""

from __future__ import annotations

import sys

import typer


def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind the dashboard to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to serve the dashboard on"),
    offline: bool = typer.Option(False, "--offline", help="Use simulated candles instead of live CoinDCX data"),
) -> None:
    """Start the futurecast web dashboard (`futurecast web` works too)."""
    from futurecast.publisher.web_server import run_web_server

    run_web_server(host=host, port=port, live=not offline)


def main() -> None:
    # `futurecast web ...` predates the dashboard being the only command; accept it as-is.
    if sys.argv[1:2] == ["web"]:
        del sys.argv[1]
    typer.run(dashboard)


if __name__ == "__main__":
    main()
