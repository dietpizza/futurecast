"""Local WebSocket server for broadcasting real-time signal states to external clients."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Set
import websockets
try:
    from websockets.asyncio.server import ServerConnection as ClientSocket
except ImportError:
    from websockets.server import WebSocketServerProtocol as ClientSocket

from futurecast.schema import ForecastEvent

logger = logging.getLogger(__name__)


class LocalSignalStreamServer:
    """Lightweight broadcast WebSocket server for continuous signal streaming.
    
    Permits external frontends, dashboards, or alerting services to subscribe to
    the live signal state stream locally.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self.clients: Set[ClientSocket] = set()
        self._server = None
        self._running = False

    async def register(self, ws: ClientSocket) -> None:
        self.clients.add(ws)
        logger.info(f"Client connected to signal stream. Total clients: {len(self.clients)}")
        try:
            # Send initial greeting with safety guarantee
            await ws.send(json.dumps({
                "type": "CONNECTION_ESTABLISHED",
                "message": "futurecast real-time signal stream",
                "execution_disabled": True,
            }))
            await ws.wait_closed()
        finally:
            self.clients.discard(ws)
            logger.info(f"Client disconnected. Remaining clients: {len(self.clients)}")

    async def start(self) -> None:
        """Start the WebSocket broadcast server."""
        self._running = True
        self._server = await websockets.serve(self.register, self.host, self.port)
        logger.info(f"Signal broadcast server listening at ws://{self.host}:{self.port}")

    async def broadcast_event(self, event: ForecastEvent) -> None:
        """Push a new forecast event to all connected clients."""
        if not self.clients:
            return

        payload = event.model_dump_json()
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(payload)
            except Exception:
                disconnected.add(client)

        for client in disconnected:
            self.clients.discard(client)

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Signal broadcast server stopped.")
