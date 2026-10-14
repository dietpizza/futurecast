"""CoinDCX public market data client (Strictly Read-Only / Public Endpoints).

Zero API keys or trading credentials required. Transparent local caching via DataCache.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional
import httpx
import pandas as pd

from futurecast.ingestion.cache import DataCache

logger = logging.getLogger(__name__)

_INTERVAL_UNIT_MS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}


def interval_to_ms(interval: str) -> int:
    """Candle interval string ("1m", "5m", "15m", "1h", "1d") to milliseconds."""
    return int(interval[:-1]) * _INTERVAL_UNIT_MS[interval[-1]]


class CoinDCXPublicClient:
    """Public read-only client for CoinDCX Spot & Derivatives market data.
    
    Compatible with CoinDCX pairs like B-DOGE_USDT, B-BTC_USDT, B-SOL_USDT, I-BTC_INR, etc.
    """

    def __init__(
        self,
        base_url: str = "https://public.coindcx.com",
        timeout: int = 15,
        rate_limit_pause: float = 0.1,
        max_retries: int = 3,
        cache: Optional[DataCache] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit_pause = rate_limit_pause
        self.max_retries = max_retries
        self.cache = cache
        self._client: Optional[httpx.Client] = None

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

    def _get_with_retry(self, endpoint: str, params: dict[str, Any]) -> Any:
        for attempt in range(1, self.max_retries + 1):
            try:
                time.sleep(self.rate_limit_pause)
                resp = self.client.get(endpoint, params=params)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", "5"))
                    logger.warning(f"CoinDCX Rate limited (429). Sleeping {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                if resp.status_code == 422:
                    logger.debug(f"CoinDCX pair rejected (422) on {endpoint}: {params.get('pair')}")
                    return None
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, httpx.TimeoutException) as err:
                if attempt == self.max_retries:
                    logger.error(f"CoinDCX request failed {endpoint} after {attempt} attempts: {err}")
                    raise
                backoff = attempt * 1.5
                time.sleep(backoff)
        raise RuntimeError("Unreachable retry loop exit")

    def fetch_futures_rt_prices(self) -> dict[str, dict[str, Any]]:
        """Fetch live ticker and mark prices for all 540+ CoinDCX futures contracts.
        
        Returns dict keyed by CoinDCX symbol (e.g. 'B-TRUTH_USDT', 'B-DOGE_USDT')
        with fields: 'ls' (last price), 'mp' (mark price), 'h' (high), 'l' (low),
        'v' (volume), 'fr' (funding rate), 'pc' (% change), 'mkt' (underlying symbol).
        """
        now = time.time()
        if hasattr(self, "_rt_cache") and hasattr(self, "_rt_cache_time"):
            if (now - self._rt_cache_time) < 2.0:
                return self._rt_cache
        try:
            resp = self.client.get("https://public.coindcx.com/market_data/v3/current_prices/futures/rt")
            if resp.status_code == 200:
                data = resp.json()
                prices = data.get("prices", {}) if isinstance(data, dict) else {}
                self._rt_cache = prices
                self._rt_cache_time = now
                return prices
        except Exception as e:
            logger.debug(f"Failed to fetch CoinDCX RT futures prices: {e}")
        return getattr(self, "_rt_cache", {})

    def fetch_ticker(self, symbol: str) -> Optional[dict[str, Any]]:
        """Fetch real-time ticker data for a single CoinDCX futures symbol."""
        rt_prices = self.fetch_futures_rt_prices()
        if symbol in rt_prices:
            return rt_prices[symbol]
        sym_clean = symbol.strip().upper()
        if sym_clean in rt_prices:
            return rt_prices[sym_clean]
        return None

    def fetch_orderbook_depth(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        """Fetch live Level-2 orderbook depth for a futures contract.
        
        Tries CoinDCX native orderbook first; if 422/unavailable, falls back to
        the underlying Binance perpetual futures depth endpoint.
        """
        sym = symbol.strip().upper()
        now = time.time()
        if not hasattr(self, "_depth_cache"):
            self._depth_cache = {}
        cached = self._depth_cache.get(sym)
        if cached and (now - cached["_fetched_at"]) < 2.0:
            return cached["data"]

        # 1. Try CoinDCX native depth
        try:
            resp = self.client.get("/market_data/orderbook", params={"pair": sym})
            if resp.status_code == 200:
                data = resp.json()
                bids_raw = data.get("bids", {})
                asks_raw = data.get("asks", {})
                bids: list[tuple[float, float]] = []
                asks: list[tuple[float, float]] = []
                if isinstance(bids_raw, dict):
                    bids = [(float(p), float(q)) for p, q in bids_raw.items()]
                elif isinstance(bids_raw, list):
                    bids = [(float(item[0]), float(item[1])) for item in bids_raw]
                if isinstance(asks_raw, dict):
                    asks = [(float(p), float(q)) for p, q in asks_raw.items()]
                elif isinstance(asks_raw, list):
                    asks = [(float(item[0]), float(item[1])) for item in asks_raw]

                bids.sort(key=lambda x: x[0], reverse=True)
                asks.sort(key=lambda x: x[0])
                res = {
                    "symbol": sym,
                    "source": "coindcx",
                    "timestamp_ms": data.get("timestamp") or int(now * 1000),
                    "bids": bids[:limit],
                    "asks": asks[:limit],
                }
                self._depth_cache[sym] = {"data": res, "_fetched_at": now}
                return res
        except Exception as e:
            logger.debug(f"CoinDCX native orderbook failed for {sym}: {e}")

        # 2. Fallback to underlying Binance Futures depth
        underlying = None
        rt_prices = getattr(self, "_rt_cache", {})
        if sym in rt_prices and "mkt" in rt_prices[sym]:
            underlying = rt_prices[sym]["mkt"]
        if not underlying:
            clean = sym.replace("B-", "").replace("I-", "").replace("_", "")
            underlying = clean

        try:
            binance_url = f"https://fapi.binance.com/fapi/v1/depth?symbol={underlying}&limit={limit}"
            resp = self.client.get(binance_url)
            if resp.status_code == 200:
                data = resp.json()
                bids = [(float(item[0]), float(item[1])) for item in data.get("bids", [])]
                asks = [(float(item[0]), float(item[1])) for item in data.get("asks", [])]
                bids.sort(key=lambda x: x[0], reverse=True)
                asks.sort(key=lambda x: x[0])
                res = {
                    "symbol": sym,
                    "source": "binance_fallback",
                    "timestamp_ms": data.get("E") or int(now * 1000),
                    "bids": bids[:limit],
                    "asks": asks[:limit],
                }
                self._depth_cache[sym] = {"data": res, "_fetched_at": now}
                return res
        except Exception as e:
            logger.debug(f"Binance fallback orderbook failed for {underlying}: {e}")

        empty_res = {
            "symbol": sym,
            "source": "empty",
            "timestamp_ms": int(now * 1000),
            "bids": [],
            "asks": [],
        }
        self._depth_cache[sym] = {"data": empty_res, "_fetched_at": now}
        return empty_res

    def _fetch_klines_via_underlying(
        self,
        symbol: str,
        interval: str = "1m",
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fallback to Binance Futures or Bybit when CoinDCX market_data/candles is unavailable."""
        rt_prices = self.fetch_futures_rt_prices()
        underlying = None
        if symbol in rt_prices and "mkt" in rt_prices[symbol]:
            underlying = rt_prices[symbol]["mkt"]
        if not underlying:
            clean = symbol.strip().upper()
            if clean.startswith("B-") or clean.startswith("I-"):
                clean = clean[2:]
            underlying = clean.replace("_", "")

        # 1. Try Binance Futures (Strictly read-only public endpoint)
        try:
            params: dict[str, Any] = {
                "symbol": underlying,
                "interval": interval,
                "limit": min(limit, 1500),
            }
            if start_time_ms is not None:
                params["startTime"] = int(start_time_ms)
            if end_time_ms is not None:
                params["endTime"] = int(end_time_ms)

            with httpx.Client(base_url="https://fapi.binance.com", timeout=self.timeout) as b_client:
                resp = b_client.get("/fapi/v1/klines", params=params)
                if resp.status_code == 200:
                    raw = resp.json()
                    if raw and isinstance(raw, list):
                        cols = [
                            "open_time", "open", "high", "low", "close", "volume",
                            "close_time", "quote_volume", "count", "taker_buy_volume",
                            "taker_buy_quote_volume", "ignore"
                        ]
                        df = pd.DataFrame(raw, columns=cols)
                        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                        for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
                            df[c] = df[c].astype(float)
                        df["count"] = df["count"].astype(int)
                        df["symbol"] = symbol
                        res = df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]].sort_values("timestamp").reset_index(drop=True)
                        return res
        except Exception as e:
            logger.debug(f"Binance fallback failed for {underlying}: {e}")

        # 2. Try Bybit Linear Perpetuals (Strictly read-only public endpoint)
        try:
            bybit_interval = "1" if interval == "1m" else ("5" if interval == "5m" else ("15" if interval == "15m" else ("60" if interval == "1h" else "D")))
            params = {
                "category": "linear",
                "symbol": underlying,
                "interval": bybit_interval,
                "limit": min(limit, 1000),
            }
            if start_time_ms is not None:
                params["start"] = int(start_time_ms)
            if end_time_ms is not None:
                params["end"] = int(end_time_ms)

            with httpx.Client(base_url="https://api.bybit.com", timeout=self.timeout) as bybit_client:
                resp = bybit_client.get("/v5/market/kline", params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    rows = data.get("result", {}).get("list", [])
                    if rows:
                        cols = ["startTime", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "turnover"]
                        df = pd.DataFrame(rows, columns=cols)
                        df["timestamp"] = pd.to_datetime(df["startTime"].astype(int), unit="ms", utc=True)
                        for c, target_col in [("openPrice", "open"), ("highPrice", "high"), ("lowPrice", "low"), ("closePrice", "close"), ("volume", "volume"), ("turnover", "quote_volume")]:
                            df[target_col] = df[c].astype(float)
                        df["count"] = 100
                        df["taker_buy_volume"] = df["volume"] * 0.5
                        df["taker_buy_quote_volume"] = df["quote_volume"] * 0.5
                        df["symbol"] = symbol
                        res = df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]].sort_values("timestamp").reset_index(drop=True)
                        return res
        except Exception as e:
            logger.debug(f"Bybit fallback failed for {underlying}: {e}")

        return pd.DataFrame()

    # ───────── Model candle history (closed candles only) ─────────

    _HISTORY_COLS = [
        "timestamp", "symbol", "open", "high", "low", "close", "volume",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
    ]

    def underlying_symbol(self, symbol: str) -> str:
        """Underlying perpetual for a CoinDCX futures pair (e.g. B-DOGE_USDT -> DOGEUSDT)."""
        cache = self.__dict__.setdefault("_underlying_cache", {})
        if symbol not in cache:
            info = self.fetch_futures_rt_prices().get(symbol, {})
            mkt = info.get("mkt")
            if not mkt:
                clean = symbol.strip().upper()
                clean = clean[2:] if clean[:2] in ("B-", "I-") else clean
                mkt = clean.replace("_", "")
            cache[symbol] = mkt
        return cache[symbol]

    def _binance_page(self, market: str, interval: str, end_ms: int, limit: int) -> Optional[list]:
        resp = httpx.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": market, "interval": interval, "endTime": end_ms, "limit": min(limit, 1500)},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            return None
        return [
            (int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]),
             float(r[7]), int(r[8]), float(r[9]), float(r[10]))
            for r in resp.json()
        ]

    def _bybit_page(self, market: str, interval: str, end_ms: int, limit: int) -> Optional[list]:
        resp = httpx.get(
            "https://api.bybit.com/v5/market/kline",
            params={"category": "linear", "symbol": market, "interval": str(interval_to_ms(interval) // 60_000),
                    "end": end_ms, "limit": min(limit, 1000)},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            return None
        rows = resp.json().get("result", {}).get("list", [])
        # Bybit has no taker split or trade count: neutral placeholders
        return [
            (int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]),
             float(r[6]), 0, float(r[5]) * 0.5, float(r[6]) * 0.5)
            for r in rows
        ]

    def _coindcx_page(self, symbol: str, interval: str, end_ms: int, limit: int) -> Optional[list]:
        limit = min(limit, 1000)
        raw = self._get_with_retry("/market_data/candles", {
            "pair": symbol, "interval": interval, "limit": limit,
            "startTime": end_ms - (limit - 1) * interval_to_ms(interval), "endTime": end_ms,
        })
        if not raw:
            return None
        # CoinDCX has no taker split or trade count: neutral placeholders
        return [
            (int(c["time"]), float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"]),
             float(c["volume"]), float(c["volume"]) * float(c["close"]), 0,
             float(c["volume"]) * 0.5, float(c["volume"]) * float(c["close"]) * 0.5)
            for c in raw
        ]

    def fetch_candle_history(
        self,
        symbol: str,
        interval: str = "5m",
        bars: int = 1500,
        end_time_ms: Optional[int] = None,
    ) -> tuple[pd.DataFrame, Optional[str]]:
        """Newest `bars` fully closed candles for a CoinDCX futures pair, oldest first.

        Sources, first that answers wins: Binance USD-M futures (real taker flow),
        Bybit linear, then CoinDCX itself (1m/15m only). Pages walk backward from the
        last closed candle, so the still-forming candle is never included.
        Returns (candles, source); source is None when nothing could be fetched.
        """
        step = interval_to_ms(interval)
        now_ms = int(time.time() * 1000) if end_time_ms is None else int(end_time_ms)
        last_closed_open = (now_ms // step) * step - step
        market = self.underlying_symbol(symbol)

        attempts = [("binance", self._binance_page, market), ("binance", self._binance_page, f"1000{market}"),
                    ("bybit", self._bybit_page, market), ("bybit", self._bybit_page, f"1000{market}"),
                    ("coindcx", self._coindcx_page, symbol)]
        for source, page_fn, name in attempts:
            rows: list = []
            curr_end = last_closed_open + step - 1
            try:
                while len(rows) < bars:
                    want = bars - len(rows)
                    page = page_fn(name, interval, curr_end, want)
                    if not page:
                        break
                    rows.extend(page)
                    curr_end = min(r[0] for r in page) - 1
                    if len(page) < min(want, 1000):
                        break
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as err:
                logger.debug(f"{source} history failed for {name} {interval}: {err}")
            if not rows:
                continue

            df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume", "quote_volume",
                                             "count", "taker_buy_volume", "taker_buy_quote_volume"])
            df = df[df["time"] <= last_closed_open].drop_duplicates("time").sort_values("time").tail(bars)
            df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            df["symbol"] = symbol
            return df[self._HISTORY_COLS].reset_index(drop=True), f"{source}:{name}"

        return pd.DataFrame(columns=self._HISTORY_COLS), None

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch historical candles from CoinDCX public endpoint (with pagination support and underlying fallback)."""
        limit = min(limit, 1000)
        cache_key = None
        if self.cache is not None and start_time_ms is not None:
            cache_key = self.cache.build_kline_key(f"coindcx_{symbol}", interval, start_time_ms, end_time_ms, limit)
            cached = self.cache.get("klines", cache_key)
            if cached is not None:
                return cached

        all_candles: list[dict[str, Any]] = []
        step_ms = interval_to_ms(interval)

        if start_time_ms is not None and end_time_ms is not None and (end_time_ms - start_time_ms) >= (limit * step_ms):
            # Walk backward in windows of exactly `limit` candles. The endpoint's ordering
            # within a page is not relied on: each window is bounded on both sides.
            curr_end = int(end_time_ms)
            max_pages = min(25, (end_time_ms - start_time_ms) // (limit * step_ms) + 1)
            for _ in range(max_pages):
                window_start = max(int(start_time_ms), curr_end - (limit - 1) * step_ms)
                params: dict[str, Any] = {
                    "pair": symbol,
                    "interval": interval,
                    "limit": limit,
                    "startTime": window_start,
                    "endTime": curr_end,
                }
                raw = self._get_with_retry("/market_data/candles", params)
                times = [c["time"] for c in raw or [] if c.get("time") is not None]
                if not times:
                    break
                all_candles.extend(raw)
                earliest_time = min(times)
                if earliest_time <= start_time_ms:
                    break
                curr_end = earliest_time - step_ms
        else:
            params = {
                "pair": symbol,
                "interval": interval,
                "limit": limit,
            }
            if start_time_ms is not None:
                params["startTime"] = int(start_time_ms)
            if end_time_ms is not None:
                params["endTime"] = int(end_time_ms)
            raw = self._get_with_retry("/market_data/candles", params)
            if raw:
                all_candles = raw

        if not all_candles:
            # Fallback to underlying contract on Binance/Bybit
            res = self._fetch_klines_via_underlying(symbol, interval, start_time_ms, end_time_ms, limit)
            if not res.empty and self.cache is not None and cache_key is not None:
                self.cache.put("klines", cache_key, res)
            return res

        df = pd.DataFrame(all_candles)
        if "time" not in df.columns:
            res = self._fetch_klines_via_underlying(symbol, interval, start_time_ms, end_time_ms, limit)
            if not res.empty and self.cache is not None and cache_key is not None:
                self.cache.put("klines", cache_key, res)
            return res

        df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df["quote_volume"] = df["volume"] * df["close"]
        df["count"] = 100
        df["taker_buy_volume"] = df["volume"] * 0.5  # Neutral estimate if taker split not provided
        df["taker_buy_quote_volume"] = df["quote_volume"] * 0.5
        df["symbol"] = symbol

        # Sort ascending for continuous time series.
        result = df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]].sort_values("timestamp").reset_index(drop=True)

        if self.cache is not None and cache_key is not None and not result.empty:
            self.cache.put("klines", cache_key, result)

        return result
