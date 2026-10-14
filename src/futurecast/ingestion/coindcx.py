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
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, httpx.TimeoutException) as err:
                if attempt == self.max_retries:
                    logger.error(f"CoinDCX request failed {endpoint} after {attempt} attempts: {err}")
                    raise
                backoff = attempt * 1.5
                time.sleep(backoff)
        raise RuntimeError("Unreachable retry loop exit")

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch historical candles from CoinDCX public endpoint (with pagination support)."""
        limit = min(limit, 1000)
        cache_key = None
        if self.cache is not None and start_time_ms is not None:
            cache_key = self.cache.build_kline_key(f"coindcx_{symbol}", interval, start_time_ms, end_time_ms, limit)
            cached = self.cache.get("klines", cache_key)
            if cached is not None:
                return cached

        all_candles: list[dict[str, Any]] = []

        if start_time_ms is not None and end_time_ms is not None and (end_time_ms - start_time_ms) >= (limit * 60 * 1000):
            # Paginate backward from end_time_ms to start_time_ms
            curr_end = end_time_ms
            max_pages = 20  # Safeguard maximum pages
            for _ in range(max_pages):
                params: dict[str, Any] = {
                    "pair": symbol,
                    "interval": interval,
                    "limit": limit,
                    "startTime": int(start_time_ms),
                    "endTime": int(curr_end),
                }
                raw = self._get_with_retry("/market_data/candles", params)
                if not raw:
                    break
                all_candles.extend(raw)
                earliest_time = raw[-1].get("time")
                if earliest_time is None or len(raw) < limit:
                    break
                curr_end = earliest_time - 60000
                if curr_end < start_time_ms:
                    break
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
            return pd.DataFrame()

        df = pd.DataFrame(all_candles)
        if "time" not in df.columns:
            return pd.DataFrame()

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
