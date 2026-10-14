"""Air-gapped signal publisher and telemetry broadcast engine.

Strictly zero trade execution logic. Outputs to JSONL logs, terminal, WebSocket, and webhooks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional
import httpx
from rich.console import Console
from rich.table import Table

from futurecast.schema import ForecastEvent, SignalState
from futurecast.publisher.server import LocalSignalStreamServer

logger = logging.getLogger(__name__)


class SignalBroadcaster:
    """Dispatches real-time forecasts to audit logs, terminal, local WS, and monitoring webhooks."""

    def __init__(
        self,
        output_dir: str | Path = "logs/forecasts",
        webhook_url: Optional[str] = None,
        console: Optional[Console] = None,
        ws_server: Optional[LocalSignalStreamServer] = None,
        render_terminal: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.webhook_url = webhook_url
        self.console = console or Console()
        self.ws_server = ws_server
        self.render_terminal = render_terminal
        self._http_client: Optional[httpx.Client] = None

    def _get_log_file(self, date_str: str) -> Path:
        return self.output_dir / f"forecasts_{date_str}.jsonl"

    def broadcast(self, event: ForecastEvent) -> None:
        """Broadcast forecast event across all air-gapped channels."""
        # 1. Enforce security assertion
        assert event.execution_disabled is True, "Security violation: execution must be disabled"

        # 2. Append to local audit JSONL
        date_str = event.timestamp.strftime("%Y-%m-%d")
        log_file = self._get_log_file(date_str)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

        # 3. Print to Terminal using Rich (if enabled)
        if self.render_terminal:
            self._render_terminal(event)

        # 4. Local WebSocket Server broadcast
        if self.ws_server is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.ws_server.broadcast_event(event))
                else:
                    loop.run_until_complete(self.ws_server.broadcast_event(event))
            except Exception as e:
                logger.debug(f"WS broadcast warning: {e}")

        # 5. Optional Webhook dispatch
        if self.webhook_url:
            self._send_webhook(event)

    def _render_terminal(self, event: ForecastEvent) -> None:
        table = Table(title=f"Forecast Event @ {event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Target Symbol", event.symbol)
        table.add_row("Benchmark Symbol", event.benchmark_symbol)
        table.add_row("Current Target Price", f"${event.current_price:.6f}")
        table.add_row("Benchmark Price", f"${event.benchmark_price:.2f}")
        table.add_row("15m Expected Residual Alpha", f"{event.expected_residual_alpha_15m_bps:+.2f} bps")
        table.add_row("Normalized Z-Score", f"{event.z_score:+.2f}")
        table.add_row("Beta Hedge Ratio", f"{event.beta_hedge_ratio:.3f}")

        # Signal state styling
        if event.signal == SignalState.BULLISH_RESIDUAL:
            sig_styled = "[bold green]BULLISH RESIDUAL (+)[/bold green]"
        elif event.signal == SignalState.BEARISH_RESIDUAL:
            sig_styled = "[bold red]BEARISH RESIDUAL (-)[/bold red]"
        else:
            sig_styled = "[yellow]NEUTRAL (CHOP)[/yellow]"

        table.add_row("Signal State", sig_styled)
        table.add_row("Confidence Percentile", f"{event.confidence_percentile:.1f}%")
        table.add_row("Execution Safety Guard", "[bold green]AIR-GAPPED (NO TRADES)[/bold green]")

        self.console.print(table)

    def _send_webhook(self, event: ForecastEvent) -> None:
        if not self.webhook_url:
            return
        try:
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = httpx.Client(timeout=5.0)
            payload = event.model_dump(mode="json")
            self._http_client.post(self.webhook_url, json=payload)
        except Exception as e:
            logger.warning(f"Failed to post forecast to webhook: {e}")
