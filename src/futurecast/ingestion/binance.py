"""Binance USD-M futures public market data (Strictly Read-Only / Public Endpoints).

Every candle, funding rate, live price and order book for a CoinDCX pair comes from the
Binance perpetual behind it: B-SOL_USDT -> SOLUSDT, B-1000PEPE_USDT -> 1000PEPEUSDT.
Zero API keys or trading credentials required.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

from futurecast.ingestion.cache import CandleCache
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
_FUNDING_PAGE = 1000
_FUNDING_LIMIT_5M = 500  # fundingRate has its own budget, outside the weight count: 500 requests / 5 min / IP
_DAY_MS = 86_400_000  # funding settles every 1h, 4h or 8h, so a day back always holds a settlement


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
        funding_budget: int = int(_FUNDING_LIMIT_5M * 0.8),
        candle_cache: Optional[CandleCache] = None,
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
        self.funding_budget = funding_budget
        self._funding_lock = threading.Lock()
        self._funding_calls: deque = deque()
        # Binance symbol -> (covered from ms, downloaded at ms, events with columns time, funding_rate)
        self._funding: Dict[str, Tuple[int, int, pd.DataFrame]] = {}
        self.candle_cache = candle_cache

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

    # ───────── funding rates ─────────

    def _funding_slot(self) -> None:
        """Wait for room in the fundingRate budget (500 requests per 5 minutes, not reported in weight headers)."""
        with self._funding_lock:
            now = time.monotonic()
            calls = self._funding_calls
            while calls and now - calls[0] >= 300:
                calls.popleft()
            at = max(now, calls[-self.funding_budget] + 300) if len(calls) >= self.funding_budget else now
            calls.append(at)
        if at > now:
            logger.warning(f"Binance funding-rate budget used up; pausing {at - now:.0f}s")
            time.sleep(at - now)

    def fetch_funding_rates(self, symbol: str, start_ms: int, end_ms: Optional[int] = None) -> pd.DataFrame:
        """Settled funding rates of the pair's Binance perpetual between `start_ms` and `end_ms`, oldest first.

        Columns: timestamp (settlement time) and funding_rate (fraction paid per funding interval;
        a 1000x listing pays the same rate). `start_ms` is rounded down to a UTC day so nearby
        requests share a download. A download is reused while it was made at or after `end_ms`
        (default: now), since no settlement before `end_ms` can be missing from it; otherwise only
        the settlements after the newest one held are fetched.
        """
        empty = pd.DataFrame(columns=["timestamp", "funding_rate"])
        market = self._require(symbol)
        if market is None:
            return empty
        now_ms = int(time.time() * 1000)
        end_ms = now_ms if end_ms is None else int(end_ms)
        start_ms = int(start_ms) // _DAY_MS * _DAY_MS

        with self._funding_lock:
            cached = self._funding.get(market.symbol)
        if cached is None or cached[0] > start_ms:
            cached = (start_ms, 0, pd.DataFrame({"time": pd.Series(dtype="int64"), "funding_rate": pd.Series(dtype=float)}))
        covered_from, fetched_at, events = cached
        if fetched_at < end_ms:
            cursor = int(events["time"].iloc[-1]) + 1 if len(events) else covered_from
            rows: List[dict] = []
            while True:
                self._funding_slot()
                page = self._get("/fapi/v1/fundingRate", {
                    "symbol": market.symbol, "startTime": cursor, "limit": _FUNDING_PAGE,
                })
                rows.extend(page)
                if len(page) < _FUNDING_PAGE:
                    break
                cursor = int(page[-1]["fundingTime"]) + 1
            if rows:
                new = pd.DataFrame({
                    "time": [int(r["fundingTime"]) for r in rows],
                    "funding_rate": pd.to_numeric([r.get("fundingRate") for r in rows], errors="coerce"),
                }).dropna()
                events = (pd.concat([events, new]) if len(events) else new)
                events = events.drop_duplicates("time").sort_values("time").reset_index(drop=True)
            with self._funding_lock:
                self._funding[market.symbol] = (covered_from, now_ms, events)

        events = events[(events["time"] >= start_ms) & (events["time"] <= end_ms)]
        if events.empty:
            return empty
        return pd.DataFrame({
            "timestamp": pd.to_datetime(events["time"].to_numpy(), unit="ms", utc=True),
            "funding_rate": events["funding_rate"].to_numpy(dtype=float),
        })

    # ───────── model candle history (closed candles only) ─────────

    def fetch_candle_history(
        self,
        symbol: str,
        interval: str = "1m",
        bars: int = 1500,
        end_time_ms: Optional[int] = None,
        funding: bool = False,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Newest `bars` fully closed candles for a CoinDCX pair from its Binance perpetual, oldest first.

        Candles come from the disk cache when the client has one; only the candles it is
        missing are downloaded (see _closed_klines). The still-forming candle is never included.
        Prices are Binance's (a 1000x listing stays in 1000x units). With `funding`, a
        funding_rate column holds the latest rate settled before each candle closed (NaN before
        the first settlement on record). `progress(done, total)` counts downloaded candles.
        Returns (candles, source); source is None when Binance has no market for the pair.
        """
        empty = pd.DataFrame(columns=self._HISTORY_COLS)
        market = self._require(symbol)
        if market is None:
            return empty, None

        step = interval_to_ms(interval)
        now_ms = int(time.time() * 1000) if end_time_ms is None else int(end_time_ms)
        last_open = (now_ms // step) * step - step
        df = self._closed_klines(market.symbol, interval, last_open - (bars - 1) * step, last_open, progress)
        if df.empty:
            return empty, None

        df = df.tail(bars).copy()
        df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df["symbol"] = symbol
        cols = list(self._HISTORY_COLS)
        if funding:
            close_ms = df["time"].to_numpy() + step
            events = self.fetch_funding_rates(symbol, int(close_ms[0]) - step - _DAY_MS, int(close_ms[-1]))
            df["funding_rate"] = funding_known_at(close_ms, events)
            cols.append("funding_rate")
        return df[cols].reset_index(drop=True), f"binance:{market.symbol}"

    def _closed_klines(
        self,
        market: str,
        interval: str,
        first_open: int,
        last_open: int,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """Closed candles opening between `first_open` and `last_open` (ms), oldest first.

        With a cache, only the stretches outside its span are downloaded: newer candles up to
        `last_open`, then older ones back to `first_open`. A new span that cannot join the
        cached one (a request far older, or a gap longer than the cache keeps) is downloaded
        whole. Every finished download is merged into the cache; an interrupted walk back is
        kept too, since what it fetched still joins the span.
        """
        step = interval_to_ms(interval)
        cache = self.candle_cache
        have, span = cache.load(market, interval) if cache is not None else (CandleCache.empty(), None)
        if span is not None and (last_open < span[0] - step or (last_open - span[1]) // step > cache.max_rows):
            have, span = CandleCache.empty(), None

        if span is None:
            jobs = [(first_open, last_open)]
        else:
            jobs = []
            if last_open > span[1]:
                jobs.append((span[1] + step, last_open))
            if first_open < span[0]:
                jobs.append((first_open, span[0] - step))
        total = sum((hi - lo) // step + 1 for lo, hi in jobs)

        frames = [have]
        done = 0
        for lo, hi in jobs:
            rows: List[list] = []
            report = (lambda n, base=done: progress(base + n, total)) if progress is not None else None
            listed = None
            try:
                listed = self._walk_back(market, interval, lo, hi, rows, report)
            finally:
                got = self._kline_frame(rows, lo, hi)
                frames.append(got)
                done += (hi - lo) // step + 1
                if cache is not None and listed is not None:  # finished: the whole stretch is covered
                    cache.store(market, interval, step, got, 0 if listed else lo, hi)
                elif cache is not None and len(got) and (span is None or hi < span[0]):
                    # interrupted, but a walk back from the span's start still joins it
                    cache.store(market, interval, step, got, int(got["time"].iloc[0]), hi)

        df = pd.concat([f for f in frames if len(f)] or [CandleCache.empty()], ignore_index=True)
        df = df.drop_duplicates("time", keep="last").sort_values("time")
        return df[(df["time"] >= first_open) & (df["time"] <= last_open)].reset_index(drop=True)

    def _walk_back(
        self,
        market: str,
        interval: str,
        first_open: int,
        last_open: int,
        rows: List[list],
        report: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """Append klines opening in [first_open, last_open] to `rows`, newest page first.

        Returns True when Binance has no candles older than the ones fetched (the listing).
        """
        step = interval_to_ms(interval)
        curr_end = last_open + step - 1
        while curr_end >= first_open:
            want = min((curr_end - first_open) // step + 1, _KLINE_PAGE)
            page = self._get("/fapi/v1/klines", {
                "symbol": market, "interval": interval, "endTime": curr_end, "limit": want,
            })
            if not page:
                return True
            rows.extend(page)
            if report is not None:
                report(len(rows))
            curr_end = min(int(r[0]) for r in page) - 1
            if len(page) < want:  # reached the listing date
                return True
        return False

    @staticmethod
    def _kline_frame(rows: List[list], first_open: int, last_open: int) -> pd.DataFrame:
        """Raw kline rows as CandleCache columns, limited to open times first_open..last_open."""
        if not rows:
            return CandleCache.empty()
        df = pd.DataFrame(
            [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]),
              float(r[7]), int(r[8]), float(r[9]), float(r[10])) for r in rows],
            columns=CandleCache.COLUMNS,
        )
        df = df[(df["time"] >= first_open) & (df["time"] <= last_open)]
        return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def funding_known_at(close_ms: np.ndarray, events: pd.DataFrame) -> np.ndarray:
    """Latest funding rate settled strictly before each candle close time (ms), NaN before the first."""
    if events.empty:
        return np.full(len(close_ms), np.nan)
    settled_ms = events["timestamp"].to_numpy(dtype="datetime64[ms]").astype("int64")
    idx = np.searchsorted(settled_ms, close_ms, side="left") - 1
    rates = events["funding_rate"].to_numpy(dtype=float)
    return np.where(idx >= 0, rates[np.clip(idx, 0, None)], np.nan)
