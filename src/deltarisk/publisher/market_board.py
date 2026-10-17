"""Market board for the add-pair drawer: price, 24h volume and price change over several windows for
every Binance USD-M perpetual, keyed by CoinDCX pair.

- 24h change, 24h quote volume and the price come from one all-symbol ticker call (/fapi/v1/ticker/24hr,
  weight 40), repeated every `poll_sec` while the board is in use.
- Shorter changes (CHANGE_WINDOWS) are read off a price history kept per market: on start, each
  market's last `HISTORY_MINUTES` of 1-minute candle closes (one klines call each, weight 2), then the
  ticker price at every poll; markets are filled in a few at a time once the page has asked for them.
  Change over w = price now / the latest price at or before (now - w) - 1,
  None until the history reaches back that far (within `tolerance`).
- The board starts on the first request and stops after `idle_sec` without one, so it costs nothing
  while the drawer stays closed. A restart after a stop backfills again.

Everything here is public market data; it shares Binance's per-IP weight budget with the dashboard,
which BinancePublicClient paces from the weight headers.
"""

from __future__ import annotations

import bisect
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from deltarisk.ingestion.binance import BinancePublicClient

logger = logging.getLogger(__name__)

CHANGE_WINDOWS: Dict[str, int] = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}  # minutes; 24h comes from the ticker
WINDOW_NAMES = (*CHANGE_WINDOWS, "24h")
HISTORY_MINUTES = 245  # the longest window plus a few candles
BACKFILL_WORKERS = 3  # klines calls in flight while filling in (weight 2 each; ~1,050 weight for ~525 markets)


class MarketBoard:
    def __init__(
        self,
        client_factory: Callable[[], BinancePublicClient],
        poll_sec: float = 30.0,
        idle_sec: float = 600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.client_factory = client_factory
        self.poll_sec = poll_sec
        self.idle_sec = idle_sec
        self.clock = clock
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._ready = threading.Event()  # set once the first ticker poll is in
        self._wake = threading.Event()  # the page asked about markets not filled in yet
        self._last_request = 0.0
        self._tickers: Dict[str, Dict[str, float]] = {}  # Binance market -> price, change_24h, volume_24h
        self._history: Dict[str, Tuple[List[int], List[float]]] = {}  # market -> (times ms, prices), oldest first
        self._backfilled: set = set()
        self._wanted: set = set()  # markets behind the pairs the page asks about: only these are backfilled
        self._updated_at: Optional[int] = None
        self.client: Optional[BinancePublicClient] = None

    # ───────── lifecycle ─────────

    def touch(self) -> None:
        """Mark the board as in use, starting it if it isn't running."""
        with self._lock:
            self._last_request = self.clock()
            running = self._thread is not None and self._thread.is_alive()
            if running:
                return
            self._stop.clear()
            self._ready.clear()
            self._tickers, self._history, self._backfilled, self._updated_at = {}, {}, set(), None
            self._wanted = set()
            self._thread = threading.Thread(target=self._run, name="market-board", daemon=True)
            self._thread.start()

    def wait_ready(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds for the first poll after a start (so the first request has prices)."""
        return self._ready.wait(timeout)

    def stop(self) -> None:
        self._stop.set()

    def _idle(self) -> bool:
        with self._lock:
            return self.clock() - self._last_request > self.idle_sec

    def _run(self) -> None:
        self.client = self.client_factory()
        try:
            while not self._stop.is_set() and not self._idle():
                started = self.clock()
                try:
                    self.poll()
                    self.backfill(deadline=started + self.poll_sec)
                except Exception as err:  # a failed poll is retried at the next one
                    logger.warning(f"Market board refresh failed: {err}")
                self._wake.wait(max(1.0, self.poll_sec - (self.clock() - started)))
                self._wake.clear()
        finally:
            self.client.close()
            logger.info("Market board stopped (idle)")

    # ───────── data ─────────

    def poll(self) -> None:
        """One all-symbol 24h ticker: prices, 24h change and volume, and a point in each history."""
        rows = self.client.fetch_tickers_24h()
        perps = self.client.perpetuals()
        now_ms = int(self.clock() * 1000)
        with self._lock:
            for t in rows:
                market = t.get("symbol")
                if market not in perps:
                    continue
                try:
                    price = float(t["lastPrice"])
                    self._tickers[market] = {
                        "price": price,
                        "change_24h": float(t["priceChangePercent"]),
                        "volume_24h": float(t["quoteVolume"]),
                    }
                except (KeyError, TypeError, ValueError):
                    continue
                times, prices = self._history.setdefault(market, ([], []))
                times.append(now_ms)
                prices.append(price)
                self._trim(times, prices, now_ms)
            self._updated_at = now_ms
        self._ready.set()

    def backfill(self, deadline: Optional[float] = None, workers: int = BACKFILL_WORKERS) -> None:
        """Seed the history of the markets the page shows with their last 1-minute candle closes, a few
        at a time, until `deadline` (so the ticker keeps polling while hundreds of markets fill in)."""
        with self._lock:
            todo = [m for m in self._tickers if m in self._wanted and m not in self._backfilled]
        if todo:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="board-fill") as pool:
                list(pool.map(lambda m: self._fill(m, deadline), todo))

    def _fill(self, market: str, deadline: Optional[float]) -> None:
        if self._stop.is_set() or (deadline is not None and self.clock() >= deadline):
            return
        try:
            closes = self.client.fetch_recent_closes(market, HISTORY_MINUTES)
        except Exception as err:
            logger.debug(f"Market board backfill failed for {market}: {err}")
            closes = []
        with self._lock:
            self._backfilled.add(market)
            times, prices = self._history.setdefault(market, ([], []))
            merged = sorted({**dict(closes), **dict(zip(times, prices))}.items())
            times[:], prices[:] = [t for t, _ in merged], [p for _, p in merged]

    @staticmethod
    def _trim(times: List[int], prices: List[float], now_ms: int) -> None:
        cut = bisect.bisect_left(times, now_ms - (HISTORY_MINUTES + 5) * 60_000)
        if cut:
            del times[:cut], prices[:cut]

    @staticmethod
    def change(times: List[int], prices: List[float], now_ms: int, minutes: int, price: float) -> Optional[float]:
        """% change from the latest price at or before (now - minutes) to `price`, None when the history
        doesn't reach back that far (within max(90 s, a tenth of the window))."""
        target = now_ms - minutes * 60_000
        i = bisect.bisect_right(times, target) - 1
        if i < 0 or target - times[i] > max(90_000, minutes * 6_000) or prices[i] <= 0:
            return None
        return 100.0 * (price / prices[i] - 1.0)

    # ───────── output ─────────

    def view(self, symbols: List[str]) -> Dict[str, Any]:
        """Per CoinDCX pair (the catalog's symbols): price in the pair's units, 24h quote volume in USDT
        and the change over every window in %."""
        client = self.client
        with self._lock:
            tickers = dict(self._tickers)
            history = {m: (list(t), list(p)) for m, (t, p) in self._history.items()}
            updated_at = self._updated_at
            backfilled = set(self._backfilled)
        coins: Dict[str, Any] = {}
        wanted: set = set()
        if client is not None and tickers:
            for sym in symbols:
                market = client.resolve(sym)
                if market is None or market.symbol not in tickers:
                    continue
                wanted.add(market.symbol)
                t = tickers[market.symbol]
                times, prices = history.get(market.symbol, ([], []))
                now_ms = times[-1] if times else 0
                change = {name: self.change(times, prices, now_ms, minutes, t["price"]) for name, minutes in CHANGE_WINDOWS.items()}
                change["24h"] = t["change_24h"]
                coins[sym] = {"price": t["price"] / market.multiplier, "volume_24h": t["volume_24h"], "change": change}
        with self._lock:
            new = wanted - self._wanted
            self._wanted |= wanted
        if new:
            self._wake.set()
        return {
            "windows": list(WINDOW_NAMES),
            "updated_at": updated_at,
            "markets": len(wanted),
            "backfilled": len(wanted & backfilled),
            "coins": coins,
        }
