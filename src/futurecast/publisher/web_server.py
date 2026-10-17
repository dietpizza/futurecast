"""FastAPI web server for the momentum dashboard: REST for the watchlist, /ws for live updates."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from fastapi.staticfiles import StaticFiles

from futurecast.config import load_config
from futurecast.ingestion.coindcx_catalog import get_all_coindcx_instruments
from futurecast.ingestion.factory import detect_exchange
from futurecast.publisher.multi_manager import MultiCoinStreamManager

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
HTML_PATH = WEB_DIR / "index.html"
STATIC_DIR = WEB_DIR / "static"


class RevalidatingStaticFiles(StaticFiles):
    """Static files the browser must revalidate on every load (cheap 304 via ETag).

    Without an explicit Cache-Control, browsers heuristically cache assets based on
    their Last-Modified age, so an older stylesheet can be served stale alongside
    freshly edited ones and break the UI.
    """

    async def get_response(self, path: str, scope: Any) -> Any:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class AddCoinRequest(BaseModel):
    symbol: str


def create_app(
    initial_symbols: Optional[List[str]] = None,
    live_mode: bool = True,
    persistence_path: Optional[str | Path] = "tracked_symbols.json",
) -> FastAPI:
    """Create and configure the FastAPI web application (load_config refuses any config that could trade)."""
    manager = MultiCoinStreamManager(config=load_config(), persistence_path=persistence_path, live=live_mode)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: restore the coins the user added (explicit initial_symbols win, e.g. from tests).
        # There are no default coins: with nothing saved, the watchlist starts empty.
        symbols = initial_symbols if initial_symbols is not None else manager.load_persisted_symbols()
        for s in symbols:
            try:
                manager.add_symbol(s, persist=False)
            except Exception as e:
                logger.warning(f"Could not initialize symbol {s}: {e}")
        # Persist loaded active coins
        manager.persist_symbols()

        # Start background streaming task
        streaming_task = asyncio.create_task(manager.start_streaming())
        yield
        # Shutdown
        manager.stop()
        streaming_task.cancel()
        try:
            await streaming_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(
        title="FutureCast",
        description="Momentum statistics for CoinDCX futures (read-only)",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.manager = manager

    app.mount("/static", RevalidatingStaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def get_index() -> HTMLResponse:
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"), headers={"Cache-Control": "no-cache"})

    @app.get("/api/coins")
    async def get_coins() -> List[Dict[str, Any]]:
        mgr: MultiCoinStreamManager = app.state.manager
        return mgr.get_snapshot()

    @app.get("/api/coindcx/futures")
    async def get_coindcx_futures() -> List[Dict[str, Any]]:
        """Every CoinDCX futures pair the add-coin search offers; the page searches and filters it."""
        return get_all_coindcx_instruments()

    @app.post("/api/coins")
    async def add_coin(req: AddCoinRequest) -> Dict[str, Any]:
        mgr: MultiCoinStreamManager = app.state.manager
        sym = req.symbol.strip().upper()
        if not sym:
            raise HTTPException(status_code=400, detail="Invalid symbol")
        if detect_exchange(sym) != "coindcx":  # CoinDCX symbols only (B- or I- or containing _)
            raise HTTPException(
                status_code=400,
                detail=f"Only CoinDCX symbols are supported (e.g. B-DOGE_USDT, B-SOL_USDT, B-PEPE_USDT, I-BTC_INR). Received '{sym}'."
            )

        worker = mgr.add_symbol(sym)
        # Download its history in the background (one coin at a time)
        if mgr.live and worker.status in ("pending", "failed"):
            mgr.schedule_load(worker.symbol)
        item = worker.snapshot()
        await mgr.broadcast({"type": "FORECAST_UPDATE", **item})
        return item

    @app.post("/api/coins/{symbol}/reload")
    async def reload_coin(symbol: str) -> Dict[str, Any]:
        mgr: MultiCoinStreamManager = app.state.manager
        sym = symbol.strip().upper()
        if sym not in mgr.workers:
            raise HTTPException(status_code=404, detail="Coin not tracked")
        if mgr.live:
            mgr.schedule_load(sym)
        return {"status": "load_queued" if mgr.live else "offline", "symbol": sym}

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
            await websocket.send_text(json.dumps({"type": "INIT_STATE", "coins": mgr.get_snapshot()}))
            while True:
                await websocket.receive_text()  # the page only listens; coins change through the REST API
        except WebSocketDisconnect:
            pass
        finally:
            mgr.ws_clients.discard(websocket)
            logger.info(f"WebSocket client disconnected. Total clients: {len(mgr.ws_clients)}")

    return app


def run_web_server(host: str = "0.0.0.0", port: int = 8000, live: bool = True) -> None:
    """Launch the Web UI server."""
    app = create_app(live_mode=live)
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host  # a wildcard bind is not a browsable address
    print(f"\n⚡ FutureCast Web UI running at: http://{shown}:{port} (listening on {host})")
    print("🛡️  Safety: STRICTLY ZERO ORDER EXECUTION\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
