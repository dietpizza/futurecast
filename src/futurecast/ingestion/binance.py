"""Binance USD-M futures public market data (Strictly Read-Only / Public Endpoints).

Every candle, live price and order book for a CoinDCX pair comes from the Binance
perpetual behind it: B-SOL_USDT -> SOLUSDT, B-1000PEPE_USDT -> 1000PEPEUSDT.
Zero API keys or trading credentials required.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import pandas as pd

from futurecast.ingestion.factory import extract_base_and_quote

logger = logging.getLogger(__name__)

BINANCE_FAPI_URL = "https://fapi.binance.com"

_INTERVAL_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
_PERPETUAL_TYPES = ("PERPETUAL", "TRADIFI_PERPETUAL")  # TRADIFI: gold, oil, ... (B-XAU_USDT)
# Binance lists some low-priced coins per 1000 / 1M units (1000PEPEUSDT, 1MBABYDOGEUSDT)
_UNIT_PREFIXES: Tuple[Tuple[str, int], ...] = (("", 1), ("1000", 1_000), ("1000000", 1_000_000), ("1M", 1_000_000))
_KLINE_PAGE = 1000  # klines weight is 5 up to 1000 rows, 10 above: 1000-row pages are cheapest per candle
_DEPTH_LIMITS = (5, 10, 20, 50, 100, 500, 1000)
_WEIGHT_LIMIT_1M = 2400


def interval_to_ms(interval: str) -> int:
    """Candle interval string ("1m", "5m", "15m", "1h", "1d") to milliseconds."""
    return int(interval[:-1]) * _INTERVAL_UNIT_MS[interval[-1]]


@dataclass(frozen=True)
class BinanceMarket:
    """Binance perpetual behind a pair; `multiplier` coins per Binance price unit (1000 for 1000PEPEUSDT
    standing in for B-PEPE_USDT, 1 when the names match)."""

    symbol: str
    multiplier: int = 1


def binance_candidates(symbol: str) -> List[BinanceMarket]:
    """Binance perpetual names a pair could trade under, exact name first.

    'B-SOL_USDT' -> SOLUSDT, then 1000SOLUSDT, 1000000SOLUSDT, 1MSOLUSDT. INR pairs map to the
    USDT perpetual (Binance has no INR markets): 'I-BTC_INR' -> BTCUSDT.
    """
    base, quote = extract_base_and_quote(symbol)
    if quote == "INR":
        quote = "USDT"
    return [BinanceMarket(f"{prefix}{base}{quote}", mult) for prefix, mult in _UNIT_PREFIXES]


class BinancePublicClient:
    """Public read-only client for Binance USD-M perpetual market data, keyed by CoinDCX pair."""

    _HISTORY_COLS = [
        "timestamp", "symbol", "open", "high", "low", "close", "volume",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
    ]

    def __init__(
        self,
        base_url: str = BINANCE_FAPI_URL,
        timeout: float = 15,
        max_retries: int = 3,
        markets_ttl_sec: float = 3600.0,
        weight_budget: int = int(_WEIGHT_LIMIT_1M * 0.8),
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.markets_ttl_sec = markets_ttl_sec
        self.weight_budget = weight_budget
        self._client: Optional[httpx.Client] = None
        self._markets_lock = threading.Lock()
        self._perpetuals: Dict[str, Dict[str, Any]] = {}
        self._perpetuals_at = 0.0

    @property
    def client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"User-Agent": "futurecast-research/1.0"},
            )
        return self._client

    def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            self._client.close()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET with retries; honours 429/418 Retry-After and pauses before the per-minute weight cap."""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.get(path, params=params)
            except httpx.TransportError as err:
                if attempt == self.max_retries:
                    logger.error(f"Binance request failed {path} after {attempt} attempts: {err}")
                    raise
                time.sleep(attempt * 1.5)
                continue

            if resp.status_code in (418, 429):  # 418: IP ban for ignoring 429s
                wait = float(resp.headers.get("Retry-After", 5 * attempt))
                logger.warning(f"Binance rate limited ({resp.status_code}) on {path}; retry after {wait:.0f}s")
                if attempt == self.max_retries or wait > 120:
                    resp.raise_for_status()
                time.sleep(wait)
                continue
            if resp.status_code >= 500 and attempt < self.max_retries:
                time.sleep(attempt * 1.5)
                continue
            resp.raise_for_status()

            used = int(resp.headers.get("X-MBX-USED-WEIGHT-1M", 0) or 0)
            if used >= self.weight_budget:
                pause = 61 - time.time() % 60  # weight resets each clock minute
                logger.warning(f"Binance weight {used}/{_WEIGHT_LIMIT_1M} used this minute; pausing {pause:.0f}s")
                time.sleep(pause)
            return resp.json()
        raise RuntimeError("Unreachable retry loop exit")

    # ───────── symbol mapping ─────────

    def perpetuals(self) -> Dict[str, Dict[str, Any]]:
        """Currently trading Binance perpetuals by symbol (exchangeInfo, refreshed hourly)."""
        with self._markets_lock:
            if self._perpetuals and time.time() - self._perpetuals_at < self.markets_ttl_sec:
                return self._perpetuals
            try:
                info = self._get("/fapi/v1/exchangeInfo")
            except httpx.HTTPError:
                if self._perpetuals:  # keep serving the last good list through an outage
                    return self._perpetuals
                raise
            self._perpetuals = {
                s["symbol"]: s for s in info.get("symbols", [])
                if s.get("status") == "TRADING" and s.get("contractType") in _PERPETUAL_TYPES
            }
            self._perpetuals_at = time.time()
            return self._perpetuals

    def resolve(self, symbol: str) -> Optional[BinanceMarket]:
        """Binance perpetual for a CoinDCX pair, or None when Binance has no such market."""
        perps = self.perpetuals()
        return next((m for m in binance_candidates(symbol) if m.symbol in perps), None)

    def _require(self, symbol: str) -> Optional[BinanceMarket]:
        market = self.resolve(symbol)
        if market is None:
            logger.warning(f"No trading Binance perpetual found for {symbol}")
        return market

    # ───────── live price & order book ─────────

    def fetch_live_prices(self, symbols: Iterable[str]) -> Dict[str, float]:
        """Last traded price per pair from one all-symbol ticker call, in the pair's own units."""
        try:
            tickers = self._get("/fapi/v2/ticker/price")
            last = {t["symbol"]: float(t["price"]) for t in tickers}
            prices: Dict[str, float] = {}
            for sym in symbols:
                market = self.resolve(sym)
                if market is not None and market.symbol in last:
                    prices[sym] = last[market.symbol] / market.multiplier
            return prices
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as err:
            logger.debug(f"Binance live prices failed: {err}")
            return {}

    def fetch_orderbook_depth(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Level-2 order book of the pair's Binance perpetual, best levels first, in the pair's units."""
        book: Dict[str, Any] = {"symbol": symbol, "source": "empty", "timestamp_ms": int(time.time() * 1000),
                                "bids": [], "asks": []}
        market = self._require(symbol)
        if market is None:
            return book
        depth_limit = next((n for n in _DEPTH_LIMITS if n >= limit), _DEPTH_LIMITS[-1])
        data = self._get("/fapi/v1/depth", {"symbol": market.symbol, "limit": depth_limit})
        m = market.multiplier
        levels = {
            side: [(float(p) / m, float(q) * m) for p, q in data.get(side, [])]
            for side in ("bids", "asks")
        }
        levels["bids"].sort(key=lambda x: x[0], reverse=True)
        levels["asks"].sort(key=lambda x: x[0])
        book.update(
            source=f"binance:{market.symbol}",
            timestamp_ms=data.get("E") or book["timestamp_ms"],
            bids=levels["bids"][:limit],
            asks=levels["asks"][:limit],
        )
        return book

    # ───────── model candle history (closed candles only) ─────────

    def fetch_candle_history(
        self,
        symbol: str,
        interval: str = "5m",
        bars: int = 1500,
        end_time_ms: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Newest `bars` fully closed candles for a CoinDCX pair from its Binance perpetual, oldest first.

        Pages walk backward from the last closed candle, so the still-forming candle is never
        included. Prices are Binance's (a 1000x listing stays in 1000x units). Returns
        (candles, source); source is None when Binance has no market for the pair.
        """
        empty = pd.DataFrame(columns=self._HISTORY_COLS)
        market = self._require(symbol)
        if market is None:
            return empty, None

        step = interval_to_ms(interval)
        now_ms = int(time.time() * 1000) if end_time_ms is None else int(end_time_ms)
        last_closed_open = (now_ms // step) * step - step
        curr_end = last_closed_open + step - 1

        rows: List[list] = []
        while len(rows) < bars:
            want = min(bars - len(rows), _KLINE_PAGE)
            page = self._get("/fapi/v1/klines", {
                "symbol": market.symbol, "interval": interval, "endTime": curr_end, "limit": want,
            })
            if not page:
                break
            rows.extend(page)
            curr_end = min(int(r[0]) for r in page) - 1
            if len(page) < want:  # reached the listing date
                break
        if not rows:
            return empty, None

        df = pd.DataFrame(
            [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]),
              float(r[7]), int(r[8]), float(r[9]), float(r[10])) for r in rows],
            columns=["time", "open", "high", "low", "close", "volume", "quote_volume",
                     "count", "taker_buy_volume", "taker_buy_quote_volume"],
        )
        df = df[df["time"] <= last_closed_open].drop_duplicates("time").sort_values("time").tail(bars)
        df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df["symbol"] = symbol
        return df[self._HISTORY_COLS].reset_index(drop=True), f"binance:{market.symbol}"
