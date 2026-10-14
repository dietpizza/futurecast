"""Rich terminal live dashboard for continuous signal state visualization."""

from __future__ import annotations

from datetime import datetime
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text

from futurecast.schema import ForecastEvent, SignalState


class LiveSignalDashboard:
    """Renders a real-time reactive terminal dashboard for streaming signal states."""

    def __init__(self, target_symbol: str = "DOGEUSDT", benchmark_symbol: str = "BTCUSDT") -> None:
        self.target_symbol = target_symbol
        self.benchmark_symbol = benchmark_symbol
        self.last_event: ForecastEvent | None = None
        self.rolling_metrics: dict[str, float] = {}
        self.event_history: list[ForecastEvent] = []

    def update(self, event: ForecastEvent, rolling_metrics: dict[str, float] | None = None) -> None:
        self.last_event = event
        self.event_history.append(event)
        if len(self.event_history) > 10:
            self.event_history.pop(0)
        if rolling_metrics:
            self.rolling_metrics = rolling_metrics

    def render(self) -> RenderableType:
        """Create renderable layout for rich.live.Live."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", size=10),
            Layout(name="history", size=8),
        )

        # Header
        header_text = Text()
        header_text.append("⚡ futurecast ", style="bold cyan")
        header_text.append("│ AIR-GAPPED SIGNAL STREAMER │ ", style="bold white")
        header_text.append("EXECUTION: DISABLED (READ-ONLY) ", style="bold green on black")
        header_panel = Panel(header_text, style="blue")
        layout["header"].update(header_panel)

        if not self.last_event:
            wait_panel = Panel("Waiting for first closed bar from market data feed...", style="dim")
            layout["main"].update(wait_panel)
            layout["history"].update(Panel("", style="dim"))
            return layout

        ev = self.last_event

        # Main Table
        table = Table(show_header=True, expand=True)
        table.add_column("Property", style="cyan", width=26)
        table.add_column("Current State", style="bold white")
        table.add_column("Diagnostics / Microstructure", style="magenta")

        # Style signal state
        if ev.signal == SignalState.BULLISH_RESIDUAL:
            sig_badge = "[bold white on green]  ▲ BULLISH RESIDUAL  [/bold white on green]"
        elif ev.signal == SignalState.BEARISH_RESIDUAL:
            sig_badge = "[bold white on red]  ▼ BEARISH RESIDUAL  [/bold white on red]"
        else:
            sig_badge = "[bold black on yellow]  ■ NEUTRAL (CHOP)    [/bold black on yellow]"

        table.add_row("Live Signal State", sig_badge, f"Bar Timestamp: {ev.timestamp.strftime('%H:%M:%S UTC')}")
        table.add_row(
            "15m Expected Alpha",
            f"{ev.expected_residual_alpha_15m_bps:+.2f} bps  (Z={ev.z_score:+.2f})",
            f"Confidence: {ev.confidence_percentile:.1f}%",
        )
        table.add_row(
            "Target Price (DOGE)",
            f"${ev.current_price:.6f}",
            f"Benchmark (BTC): ${ev.benchmark_price:.2f}",
        )
        table.add_row(
            "Point-in-Time Beta",
            f"{ev.beta_hedge_ratio:.3f}x BTC",
            f"Vol Z60: {ev.microstructure_context.get('volume_zscore_60m', 0.0):+.2f}",
        )
        
        # Drift metrics
        ic_str = f"Rolling IC: {self.rolling_metrics.get('rolling_ic', 0.0):+.3f}"
        hit_str = f"Hit Rate: {self.rolling_metrics.get('hit_rate', 0.5)*100:.1f}%"
        table.add_row("Model Health (Drift)", f"{ic_str} │ {hit_str}", f"Completed Audits: {self.rolling_metrics.get('sample_count', 0)}")

        layout["main"].update(Panel(table, title=f"Real-Time Telemetry: {ev.symbol} vs {ev.benchmark_symbol}", border_style="cyan"))

        # History Table
        hist_table = Table(show_header=True, expand=True)
        hist_table.add_column("Time", style="dim", width=12)
        hist_table.add_column("Signal", width=22)
        hist_table.add_column("Alpha (bps)", justify="right")
        hist_table.add_column("Z-Score", justify="right")
        hist_table.add_column("DOGE Price", justify="right")
        hist_table.add_column("Beta Hedge", justify="right")

        for h in reversed(self.event_history):
            s_color = "green" if h.signal == SignalState.BULLISH_RESIDUAL else "red" if h.signal == SignalState.BEARISH_RESIDUAL else "yellow"
            hist_table.add_row(
                h.timestamp.strftime("%H:%M:%S"),
                f"[{s_color}]{h.signal.value}[/{s_color}]",
                f"{h.expected_residual_alpha_15m_bps:+.2f}",
                f"{h.z_score:+.2f}",
                f"${h.current_price:.5f}",
                f"{h.beta_hedge_ratio:.3f}",
            )

        layout["history"].update(Panel(hist_table, title="Recent Signal Transitions", border_style="dim"))
        return layout
