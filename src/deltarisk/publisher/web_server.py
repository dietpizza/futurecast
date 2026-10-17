"""FastAPI web server for the dashboard: REST for the watchlist and coin settings, /ws for live updates."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

from fastapi.staticfiles import StaticFiles

from deltarisk.config import load_config
from deltarisk.ingestion.coindcx_catalog import get_all_coindcx_instruments
from deltarisk.ingestion.factory import detect_exchange
from deltarisk.publisher.market_board import WINDOW_NAMES, MarketBoard
from deltarisk.publisher.multi_manager import MultiCoinStreamManager, _binance_client

logger = logging.getLogger(__name__)

# The Svelte page, built by `mise run build` (vite) into web/dist at the repository root
WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"

NOT_BUILT = """<!DOCTYPE html><meta charset="utf-8"><title>DeltaRisk</title>
<body style="font-family: sans-serif; max-width: 40em; margin: 4em auto; line-height: 1.5">
<h1>DeltaRisk</h1>
<p>The dashboard hasn't been built. Run <code>mise run build</code> and reload, or use <code>mise dev</code>
for the development server (it serves the page on port 5000).</p>
<p>The API is up: <a href="/api/coins">/api/coins</a>.</p>"""


class BuiltAssets(StaticFiles):
    """Vite's hashed assets: a changed file gets a new name, so the browser may keep each one for good."""

    async def get_response(self, path: str, scope: Any) -> Any:
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


class AddCoinRequest(BaseModel):
    symbol: str


class CoinSettingsRequest(BaseModel):
    """Fields left out stay as they are; null goes back to the automatic tier or leverage."""

    tier: Optional[str] = None
    leverage: Optional[float] = None


def create_app(
    initial_symbols: Optional[List[str]] = None,
    live_mode: bool = True,
    persistence_path: Optional[str | Path] = "tracked_symbols.json",
    web_dist: Path = WEB_DIST,
) -> FastAPI:
    """Create and configure the FastAPI web application.

    The page is served from `web_dist` (the frontend's build output); without a build, / says how to make one.
    """
    manager = MultiCoinStreamManager(config=load_config(), persistence_path=persistence_path, live=live_mode)
    board = MarketBoard(lambda: _binance_client(manager.config.assets))

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
        board.stop()
        streaming_task.cancel()
        try:
            await streaming_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(
        title="DeltaRisk",
        description="Intraday risk readings for CoinDCX futures (read-only)",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.manager = manager

    # index.html and favicon.svg are revalidated on every load, so a new build shows on reload
    app.mount("/assets", BuiltAssets(directory=web_dist / "assets", check_dir=False), name="assets")

    @app.get("/", response_class=HTMLResponse)
    async def get_index() -> Any:
        index = web_dist / "index.html"
        if not index.is_file():
            return HTMLResponse(NOT_BUILT, status_code=503)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    @app.get("/favicon.svg", include_in_schema=False)
    async def get_favicon() -> FileResponse:
        icon = web_dist / "favicon.svg"
        if not icon.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(icon, headers={"Cache-Control": "no-cache"})

    @app.get("/api/coins")
    async def get_coins() -> List[Dict[str, Any]]:
        mgr: MultiCoinStreamManager = app.state.manager
        return mgr.get_snapshot()

    @app.get("/api/coindcx/futures")
    async def get_coindcx_futures() -> List[Dict[str, Any]]:
        """Every CoinDCX futures pair the add-coin search offers; the page searches and filters it."""
        return get_all_coindcx_instruments()

    @app.get("/api/market")
    async def get_market() -> Dict[str, Any]:
        """Price, 24h volume and the change over 5m … 24h for every catalog pair (the add-pair drawer).
        The board starts on the first request and stops when the drawer has been closed for a while;
        offline there is no market data."""
        if not manager.live:
            return {"windows": list(WINDOW_NAMES), "updated_at": None, "markets": 0, "backfilled": 0, "coins": {}, "offline": True}
        board.touch()
        await asyncio.to_thread(board.wait_ready, 5.0)
        symbols = [item["symbol"] for item in get_all_coindcx_instruments()]
        return await asyncio.to_thread(board.view, symbols)

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

    @app.patch("/api/coins/{symbol}")
    async def update_coin(symbol: str, req: CoinSettingsRequest) -> Dict[str, Any]:
        mgr: MultiCoinStreamManager = app.state.manager
        sym = symbol.strip().upper()
        worker = mgr.workers.get(sym)
        if worker is None:
            raise HTTPException(status_code=404, detail="Coin not tracked")
        given = req.model_fields_set
        tier = (req.tier.lower() if req.tier else None) if "tier" in given else worker.tier_override
        leverage = req.leverage if "leverage" in given else worker.leverage_override
        try:
            mgr.set_overrides(sym, tier, leverage)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))
        item = worker.snapshot()
        await mgr.broadcast({"type": "FORECAST_UPDATE", **item})
        return item

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
    print(f"\nΔ DeltaRisk running at: http://{shown}:{port} (listening on {host})\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
