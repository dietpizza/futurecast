"""Multi-coin stream manager: closed candles in, momentum statistics and signals out.

Every number comes from fully closed 15m and 1h candles of the Binance perpetual behind the
CoinDCX pair, aligned with the coin's benchmark (BinancePublicClient.fetch_candle_history).
A coin's history (HISTORY_BARS per chart) is downloaded in the background when it is added,
one coin at a time; downloads are kept on disk (<home>/cache/candles), so a restart only fetches
the candles that closed since. Afterwards the stream loop appends each candle as it closes and
reruns the statistics (stats.momentum, stats.signals); between candles only the displayed price
and the order book refresh. Nothing is trained.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from futurecast.config import AppConfig, AssetsConfig, load_config
from futurecast.features.orderbook import compute_orderbook_metrics
from futurecast.features.setups import interval_minutes
from futurecast.ingestion.binance import BinancePublicClient
from futurecast.ingestion.cache import CandleCache
from futurecast.ingestion.factory import extract_base_and_quote, get_symbol_sector, resolve_benchmark
from futurecast.ingestion.storage import align_cross_asset_data
from futurecast.paths import resolve
from futurecast.stats import MomentumSpec, momentum_view, signals_view

logger = logging.getLogger(__name__)

MOMENTUM_SPEC = MomentumSpec()
CHART_MINUTES: Dict[str, int] = {iv: interval_minutes(iv) for iv in ("15m", "1h")}
# History kept per chart: 180 days of 15m, 2 years of 1h. The event studies need it; the cache
# makes it a one-off download.
HISTORY_BARS: Dict[str, int] = {"15m": 17_280, "1h": 17_520}
FUNDING_CHARTS = ("1h",)  # the coin's settled funding rate rides on these candles
OFFLINE_BARS = 3_000
CACHE_BARS = 120_000  # candles kept on disk per market and interval (backtests read the cache too)


def _synthetic_candles(symbol: str, minutes: int, bars: int, rng: np.random.Generator) -> pd.DataFrame:
    """Random-walk closed candles, used only in offline mode."""
    end = pd.Timestamp.now(tz="UTC").floor(f"{minutes}min") - pd.Timedelta(minutes=minutes)
    ts = pd.date_range(end=end, periods=bars, freq=f"{minutes}min")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002 * np.sqrt(minutes), bars)))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.001 * np.sqrt(minutes), bars))
    volume = rng.uniform(1e5, 5e5, bars)
    taker = rng.uniform(0.4, 0.6, bars)
    return pd.DataFrame({
        "timestamp": ts, "symbol": symbol, "open": open_,
        "high": np.maximum(open_, close) * (1 + wick), "low": np.minimum(open_, close) * (1 - wick),
        "close": close, "volume": volume, "quote_volume": volume * close, "count": 100,
        "taker_buy_volume": volume * taker, "taker_buy_quote_volume": volume * close * taker,
    })


def _binance_client(assets: AssetsConfig) -> BinancePublicClient:
    source, storage = assets.data_source, assets.storage
    cache = CandleCache(Path(storage.cache_dir) / "candles", max_rows=CACHE_BARS) if storage.use_cache else None
    return BinancePublicClient(base_url=source.base_url, timeout=source.request_timeout_sec,
                               max_retries=source.max_retries, candle_cache=cache)


def _base(symbol: str) -> str:
    return extract_base_and_quote(symbol)[0]


def _json_safe(obj: Any) -> Any:
    """Plain JSON types only: numpy scalars unwrapped, NaN/inf become None."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class SymbolWorker:
    """Candle history and latest statistics for one coin.

    `status`: "pending" (nothing loaded), "queued", "downloading", "ready" or "failed".
    """

    def __init__(self, symbol: str, benchmark: str, sector: str, config: AppConfig) -> None:
        self.symbol = symbol
        self.benchmark = benchmark
        self.sector = sector
        self.config = config
        self._lock = threading.RLock()

        self.status = "pending"
        self.error: Optional[str] = None
        self.progress: Optional[Dict[str, Any]] = None  # what the download is doing right now
        self.candles: Dict[str, pd.DataFrame] = {iv: pd.DataFrame() for iv in CHART_MINUTES}
        self.data_source: Optional[str] = None  # "binance:<market>"; a 1000x market's candles aren't in pair units
        self.momentum: Dict[str, Any] = {}
        self.signals: Dict[str, Any] = {}
        self.live_price: Optional[float] = None
        self.orderbook: Dict[str, Any] = {}

    # ───────── candles ─────────

    def _fetch_aligned(
        self,
        client: BinancePublicClient,
        interval: str,
        bars: int,
        progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Closed candles of the coin (with its settled funding rate on FUNDING_CHARTS) and its
        benchmark on one time grid. `progress(symbol, done, total)` reports candles downloaded
        per symbol (0, 0 as each starts)."""
        def fetch(sym: str, funding: bool) -> Tuple[pd.DataFrame, Optional[str]]:
            if progress is None:
                return client.fetch_candle_history(sym, interval, bars, funding=funding)
            progress(sym, 0, 0)
            return client.fetch_candle_history(sym, interval, bars, funding=funding,
                                               progress=lambda done, total: progress(sym, done, total))

        target, source = fetch(self.symbol, interval in FUNDING_CHARTS)
        if target.empty:
            return pd.DataFrame(), None
        bench, _ = fetch(self.benchmark, False)
        if bench.empty:
            return pd.DataFrame(), None
        aligned = align_cross_asset_data(target, {self.benchmark: bench}, freq=f"{CHART_MINUTES[interval]}min")
        return aligned, source

    def _set_progress(self, label: str, step: int, steps: int, done: int = 0, total: int = 0) -> None:
        within = done / total if total else 0.0
        self.progress = {"label": label, "step": step, "steps": steps, "done": done, "total": total,
                         "fraction": round((step - 1 + within) / steps, 4)}

    def load_history(self, client: Optional[BinancePublicClient] = None) -> None:
        """Download HISTORY_BARS closed candles per chart (status "downloading"; candles already
        in the disk cache are not fetched again), then compute the statistics ("ready").
        Failures leave status "failed" and the error, and re-raise. Without a `client` one is
        made from the config and closed afterwards."""
        self.status = "downloading"
        self.error = None
        own_client = client is None
        client = client or _binance_client(self.config.assets)
        try:
            if client.resolve(self.symbol) is None:
                raise RuntimeError(f"no trading Binance perpetual found for {self.symbol}")
            downloads = [(iv, sym) for iv in CHART_MINUTES for sym in (self.symbol, self.benchmark)]
            loaded: Dict[str, pd.DataFrame] = {}
            source = None
            for iv in CHART_MINUTES:
                def downloaded(sym: str, done: int, total: int, iv: str = iv) -> None:
                    self._set_progress(f"{iv} candles · {_base(sym)}", downloads.index((iv, sym)) + 1,
                                       len(downloads), done, total)

                aligned, source = self._fetch_aligned(client, iv, HISTORY_BARS[iv], progress=downloaded)
                if aligned.empty:
                    raise RuntimeError(f"no closed {iv} candles available")
                loaded[iv] = aligned
        except Exception as err:
            self.status = "failed"
            self.error = str(err)
            self.progress = None
            logger.error(f"Loading history failed for {self.symbol}: {err}")
            raise
        finally:
            if own_client:
                client.close()

        with self._lock:
            self.candles.update(loaded)
            self.data_source = source
            self._update_stats()
            self.status = "ready"
            self.progress = None
        counts = ", ".join(f"{len(df):,} {iv}" for iv, df in loaded.items())
        logger.info(f"Loaded {counts} candles for {self.symbol}")

    def refresh_candles(self, client: BinancePublicClient, now: Optional[pd.Timestamp] = None) -> List[str]:
        """Append newly closed candles per chart and rerun the statistics; returns charts updated.
        Does nothing until the history has loaded."""
        if self.status != "ready":
            return []
        now = now or pd.Timestamp.now(tz="UTC")
        updated = []
        for iv, minutes in CHART_MINUTES.items():
            have, keep = self.candles[iv], HISTORY_BARS[iv]
            last_closed = now.floor(f"{minutes}min") - pd.Timedelta(minutes=minutes)
            if not have.empty and have.index[-1] >= last_closed:
                continue
            missing = keep if have.empty else int((last_closed - have.index[-1]) / pd.Timedelta(minutes=minutes))
            fresh, source = self._fetch_aligned(client, iv, min(keep, missing + 3))
            if fresh.empty:
                continue
            with self._lock:
                self.data_source = source
                if have.empty or missing >= keep - 3:  # a gap too long to bridge: start over
                    self.candles[iv] = fresh.tail(keep)
                else:
                    new_rows = fresh[fresh.index > have.index[-1]]
                    if new_rows.empty:
                        continue
                    self.candles[iv] = pd.concat([have, new_rows]).tail(keep)
            updated.append(iv)
        if updated:
            with self._lock:
                self._update_stats()
        return updated

    def load_synthetic(self, seed: int = 7) -> None:
        """Offline mode only: random-walk candles so the page has something to render."""
        rng = np.random.default_rng(seed)
        with self._lock:
            for iv, minutes in CHART_MINUTES.items():
                target = _synthetic_candles(self.symbol, minutes, OFFLINE_BARS, rng)
                bench = _synthetic_candles(self.benchmark, minutes, OFFLINE_BARS, rng)
                self.candles[iv] = align_cross_asset_data(target, {self.benchmark: bench}, freq=f"{minutes}min")
            self.data_source = "synthetic (offline mode)"
            self._update_stats()
            self.status = "ready"

    # ───────── statistics ─────────

    def _update_stats(self) -> None:
        try:
            self.momentum = momentum_view(self.candles, self.symbol, self.benchmark, MOMENTUM_SPEC)
            self.signals = signals_view(self.candles, self.symbol)
        except Exception as err:
            logger.warning(f"Statistics failed for {self.symbol}: {err}")
            self.momentum, self.signals = {}, {}

    def refresh_orderbook(self, client: BinancePublicClient) -> None:
        metrics = compute_orderbook_metrics(client.fetch_orderbook_depth(self.symbol, 20))
        with self._lock:
            self.orderbook = metrics

    # ───────── output ─────────

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            candles = self.candles["15m"]
            price = self.live_price
            if price is None and not candles.empty and ":1000" not in (self.data_source or ""):
                price = float(candles[f"{self.symbol}_close"].iloc[-1])  # skipped for 1000x-quoted contracts
            history = {iv: {"candles": len(df), "from": str(df.index[0]) if len(df) else None}
                       for iv, df in self.candles.items()}
            return _json_safe({
                "symbol": self.symbol,
                "sector": self.sector,
                "has_data": not candles.empty,
                "status": self.status,
                "progress": self.progress,
                "error": self.error,
                "history": history,
                "current_price": price,
                "momentum": self.momentum,
                "signals": self.signals,
                "orderbook_depth": self.orderbook,
            })


class MultiCoinStreamManager:
    """Central supervisor coordinating candle history, live updates and WebSocket clients."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        persistence_path: Optional[str | Path] = "tracked_symbols.json",
        live: bool = True,
    ) -> None:
        self.config = config or load_config()
        self.persistence_path = resolve(persistence_path) if persistence_path is not None else None
        self.live = live
        self.workers: Dict[str, SymbolWorker] = {}
        self.ws_clients: Set[Any] = set()
        self._running = False
        self._load_lock: Optional[asyncio.Lock] = None
        self._load_tasks: Dict[str, asyncio.Task] = {}

    def load_persisted_symbols(self) -> List[str]:
        """Load list of tracked symbols from disk if previously saved."""
        if self.persistence_path is None:
            return []
        try:
            if self.persistence_path.exists():
                with open(self.persistence_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    symbols = data.get("symbols", [])
                    if isinstance(symbols, list) and symbols:
                        cleaned = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
                        logger.info(f"Loaded {len(cleaned)} persisted symbols from {self.persistence_path}: {cleaned}")
                        return cleaned
        except Exception as e:
            logger.warning(f"Could not load tracked symbols from {self.persistence_path}: {e}")
        return []

    def persist_symbols(self) -> None:
        """Save current list of tracked symbols to disk."""
        if self.persistence_path is None:
            return
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"symbols": list(self.workers.keys()), "updated_at": datetime.now(timezone.utc).isoformat()}
            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not persist tracked symbols to {self.persistence_path}: {e}")

    def add_symbol(self, symbol: str, persist: bool = True) -> SymbolWorker:
        """Start tracking a coin. In live mode its history loads in the background (schedule_load)."""
        sym = symbol.strip().upper()
        if sym in self.workers:
            return self.workers[sym]

        benchmark, sector = resolve_benchmark(sym), get_symbol_sector(sym)
        logger.info(f"Adding coin: {sym} (Sector: {sector}, Benchmark: {benchmark})")
        worker = SymbolWorker(sym, benchmark, sector, self.config)
        if not self.live:
            worker.load_synthetic(seed=sum(map(ord, sym)))
        self.workers[sym] = worker
        if persist:
            self.persist_symbols()
        return worker

    def remove_symbol(self, symbol: str, persist: bool = True) -> bool:
        sym = symbol.strip().upper()
        if sym not in self.workers:
            return False
        del self.workers[sym]
        if persist:
            self.persist_symbols()
        logger.info(f"Removed coin: {sym}")
        return True

    # ───────── history ─────────

    def schedule_load(self, symbol: str) -> None:
        """Queue a background history download (one coin at a time) unless one is queued or running."""
        sym = symbol.strip().upper()
        task = self._load_tasks.get(sym)
        if task is not None and not task.done():
            return
        self._load_tasks[sym] = asyncio.create_task(self.load_symbol_history(sym))

    def ensure_loaded(self) -> None:
        for sym, worker in self.workers.items():
            if worker.status in ("pending", "failed"):
                self.schedule_load(sym)

    async def load_symbol_history(self, symbol: str) -> None:
        sym = symbol.strip().upper()
        if sym not in self.workers:
            raise KeyError(f"Symbol {sym} is not currently tracked")
        worker = self.workers[sym]
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()

        worker.status = "queued"
        await self.broadcast({"type": "FORECAST_UPDATE", **worker.snapshot()})
        async with self._load_lock:
            worker.status = "downloading"
            await self.broadcast({"type": "FORECAST_UPDATE", **worker.snapshot()})
            try:
                await asyncio.get_running_loop().run_in_executor(None, worker.load_history)
            except Exception:
                pass  # load_history logged it and left status "failed" with the error
        if sym in self.workers:
            await self.broadcast({"type": "FORECAST_UPDATE", **worker.snapshot()})

    # ───────── output ─────────

    def get_snapshot(self) -> List[Dict[str, Any]]:
        """Current state of every tracked coin."""
        return [w.snapshot() for w in list(self.workers.values())]

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Push JSON message to all connected WebSocket clients."""
        if not self.ws_clients:
            return
        payload = json.dumps(message, default=str)
        disconnected = set()
        for ws in self.ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.add(ws)
        for ws in disconnected:
            self.ws_clients.discard(ws)

    # ───────── streaming ─────────

    def _refresh_worker(self, worker: SymbolWorker, client: BinancePublicClient) -> None:
        try:
            worker.refresh_candles(client)
        except Exception as err:
            logger.warning(f"Candle refresh failed for {worker.symbol}: {err}")
        try:
            worker.refresh_orderbook(client)
        except Exception as err:
            logger.debug(f"Order book refresh failed for {worker.symbol}: {err}")

    async def start_streaming(self, poll_interval_sec: float = 3.0) -> None:
        """Refresh prices and order books every poll; candles and statistics when a candle closes.
        Offline (live=False) it only re-sends the synthetic snapshots."""
        self._running = True
        live = self.live
        logger.info(f"Starting stream loop (interval: {poll_interval_sec}s, live: {live})...")
        client = _binance_client(self.config.assets)
        if live:
            self.ensure_loaded()

        while self._running:
            try:
                workers = list(self.workers.values())
                if not workers:
                    await asyncio.sleep(1.0)
                    continue
                if live:
                    prices = await asyncio.to_thread(client.fetch_live_prices, [w.symbol for w in workers])
                    for w in workers:
                        if w.symbol in prices:
                            w.live_price = prices[w.symbol]
                    await asyncio.gather(*(asyncio.to_thread(self._refresh_worker, w, client) for w in workers))
                for w in workers:
                    if w.symbol in self.workers:
                        await self.broadcast({"type": "FORECAST_UPDATE", **w.snapshot()})
            except Exception as e:
                logger.error(f"Error in stream loop: {e}", exc_info=True)
            await asyncio.sleep(poll_interval_sec)

    def stop(self) -> None:
        self._running = False
