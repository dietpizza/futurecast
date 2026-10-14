"""FastAPI Web Server for real-time multi-coin residual alpha forecasting and cross-verification."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from futurecast.config import load_config
from futurecast.publisher.multi_manager import MultiCoinStreamManager

logger = logging.getLogger(__name__)

HTML_PATH = Path(__file__).parent / "web" / "index.html"


class AddCoinRequest(BaseModel):
    symbol: str
    exchange: Optional[str] = None
    auto_train: bool = True


def create_app(
    initial_symbols: Optional[List[str]] = None,
    live_mode: bool = True,
    poll_interval: float = 3.0,
) -> FastAPI:
    """Create and configure the FastAPI web application."""
    config = load_config()
    config.enforce_zero_execution_policy()
    manager = MultiCoinStreamManager(config=config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: populate initial symbols
        symbols = initial_symbols or ["B-DOGE_USDT", "B-SOL_USDT", "B-BLUR_USDT", "B-AVA_USDT", "DOGEUSDT"]
        for s in symbols:
            try:
                manager.add_symbol(s)
            except Exception as e:
                logger.warning(f"Could not initialize symbol {s}: {e}")

        # Start background streaming task
        streaming_task = asyncio.create_task(
            manager.start_streaming(poll_interval_sec=poll_interval, live=live_mode)
        )
        yield
        # Shutdown
        manager.stop()
        streaming_task.cancel()
        try:
            await streaming_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(
        title="futurecast Real-Time Forecaster",
        description="Air-Gapped Multi-Model Residual Alpha Forecasting Engine",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    async def get_index() -> HTMLResponse:
        if not HTML_PATH.exists():
            return HTMLResponse("<h1>Web UI template missing</h1>", status_code=500)
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)

    @app.get("/api/status")
    async def get_status() -> Dict[str, Any]:
        mgr: MultiCoinStreamManager = app.state.manager
        return {
            "status": "online",
            "execution_disabled": True,
            "air_gapped": True,
            "active_coins": mgr.get_tracked_symbols(),
            "count": len(mgr.get_tracked_symbols()),
        }

    @app.get("/api/coins")
    async def get_coins() -> List[Dict[str, Any]]:
        mgr: MultiCoinStreamManager = app.state.manager
        return mgr.get_snapshot()

    @app.post("/api/coins")
    async def add_coin(req: AddCoinRequest) -> Dict[str, Any]:
        mgr: MultiCoinStreamManager = app.state.manager
        sym = req.symbol.strip().upper()
        if not sym:
            raise HTTPException(status_code=400, detail="Invalid symbol")
        worker = mgr.add_symbol(sym, exchange=req.exchange)
        snapshot = mgr.get_snapshot()

        # Trigger automatic background training if custom model not yet trained
        if req.auto_train and not worker.is_custom_trained:
            asyncio.create_task(mgr.train_symbol_models(worker.symbol, days=7))

        # Find the snapshot for this symbol
        for item in snapshot:
            if item["symbol"] == worker.symbol:
                await mgr.broadcast({
                    "type": "FORECAST_UPDATE",
                    **item,
                })
                return item
        return {"symbol": worker.symbol, "exchange": worker.exchange, "sector": worker.sector}

    @app.post("/api/coins/{symbol}/train")
    async def train_coin_endpoint(symbol: str) -> Dict[str, Any]:
        mgr: MultiCoinStreamManager = app.state.manager
        sym = symbol.strip().upper()
        if sym not in mgr.workers:
            raise HTTPException(status_code=404, detail="Coin not tracked")
        asyncio.create_task(mgr.train_symbol_models(sym, days=7))
        return {"status": "training_started", "symbol": sym}

    @app.delete("/api/coins/{symbol}")
    async def delete_coin(symbol: str) -> Dict[str, Any]:
        mgr: MultiCoinStreamManager = app.state.manager
        removed = mgr.remove_symbol(symbol)
        if not removed:
            raise HTTPException(status_code=404, detail="Coin not tracked")
        await mgr.broadcast({"type": "COIN_REMOVED", "symbol": symbol.upper()})
        return {"status": "removed", "symbol": symbol.upper()}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        mgr: MultiCoinStreamManager = app.state.manager
        await websocket.accept()
        mgr.ws_clients.add(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(mgr.ws_clients)}")
        try:
            # Send initial state snapshot
            await websocket.send_text(json.dumps({
                "type": "INIT_STATE",
                "coins": mgr.get_snapshot(),
                "execution_disabled": True,
            }))
            while True:
                data = await websocket.receive_text()
                try:
                    payload = json.loads(data)
                    action = payload.get("action")
                    if action == "add_coin":
                        sym = payload.get("symbol", "").strip().upper()
                        if sym:
                            mgr.add_symbol(sym)
                    elif action == "remove_coin":
                        sym = payload.get("symbol", "").strip().upper()
                        if sym:
                            mgr.remove_symbol(sym)
                except Exception:
                    pass
        except WebSocketDisconnect:
            pass
        finally:
            mgr.ws_clients.discard(websocket)
            logger.info(f"WebSocket client disconnected. Total clients: {len(mgr.ws_clients)}")

    return app


def run_web_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    symbols: Optional[List[str]] = None,
    live: bool = True,
    poll_interval: float = 3.0,
) -> None:
    """Launch the Web UI server."""
    app = create_app(initial_symbols=symbols, live_mode=live, poll_interval=poll_interval)
    print(f"\n⚡ futurecast Web UI running at: http://{host}:{port}")
    print(f"🛡️  Air-Gapped Mode: STRICTLY ZERO ORDER EXECUTION\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
