"""Tests for continuous streaming, local WebSocket broadcast server, and live dashboard."""

import asyncio
from datetime import datetime, timezone
import json
import pytest
import websockets
from futurecast.publisher.server import LocalSignalStreamServer
from futurecast.publisher.live_dashboard import LiveSignalDashboard
from futurecast.schema import ForecastEvent, SignalState


@pytest.mark.anyio
async def test_local_signal_stream_server():
    server = LocalSignalStreamServer(host="127.0.0.1", port=8768)
    await server.start()

    received_messages = []

    async def client_worker():
        async with websockets.connect("ws://127.0.0.1:8768") as ws:
            # 1. Handshake greeting
            init_msg = await ws.recv()
            received_messages.append(json.loads(init_msg))

            # 2. Wait for broadcast event
            event_msg = await ws.recv()
            received_messages.append(json.loads(event_msg))

    client_task = asyncio.create_task(client_worker())
    await asyncio.sleep(0.1)

    # Broadcast test event
    ev = ForecastEvent(
        timestamp=datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
        symbol="DOGEUSDT",
        benchmark_symbol="BTCUSDT",
        current_price=0.15,
        benchmark_price=65000.0,
        expected_residual_alpha_15m_bps=12.5,
        z_score=1.85,
        beta_hedge_ratio=1.45,
        signal=SignalState.BULLISH_RESIDUAL,
        confidence_percentile=88.0,
        microstructure_context={"volume_zscore_60m": 1.2},
        execution_disabled=True,
    )

    await server.broadcast_event(ev)
    await client_task

    await server.stop()

    assert len(received_messages) == 2
    assert received_messages[0]["type"] == "CONNECTION_ESTABLISHED"
    assert received_messages[0]["execution_disabled"] is True
    assert received_messages[1]["symbol"] == "DOGEUSDT"
    assert received_messages[1]["signal"] == "BULLISH_RESIDUAL"
    assert received_messages[1]["expected_residual_alpha_15m_bps"] == 12.5
    assert received_messages[1]["execution_disabled"] is True


def test_live_signal_dashboard_render():
    dashboard = LiveSignalDashboard(target_symbol="DOGEUSDT", benchmark_symbol="BTCUSDT")

    # Initial render before events
    initial_render = dashboard.render()
    assert initial_render is not None

    ev = ForecastEvent(
        timestamp=datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
        symbol="DOGEUSDT",
        benchmark_symbol="BTCUSDT",
        current_price=0.15,
        benchmark_price=65000.0,
        expected_residual_alpha_15m_bps=22.5,
        z_score=2.1,
        beta_hedge_ratio=1.52,
        signal=SignalState.BULLISH_RESIDUAL,
        confidence_percentile=92.0,
        microstructure_context={"volume_zscore_60m": 2.1},
        execution_disabled=True,
    )

    dashboard.update(ev, rolling_metrics={"rolling_ic": 0.045, "hit_rate": 0.58, "sample_count": 35})
    rendered = dashboard.render()
    assert rendered is not None
    assert len(dashboard.event_history) == 1
