"""Binance Futures public market data client (Strictly Read-Only / Public Endpoints).

Zero private endpoints or signing methods are implemented.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Generator, Optional
import httpx
import pandas as pd
import numpy as np

from futurecast.ingestion.cache import DataCache

logger = logging.getLogger(__name__)


class BinancePublicClient:
    """Public read-only client for Binance USDT-M Perpetual market data.
    
    No API keys, no secrets, no signing methods. Supports transparent local caching in .cache/.
    """

    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
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
                    logger.warning(f"Rate limited (429). Sleeping {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, httpx.TimeoutException) as err:
                if attempt == self.max_retries:
                    logger.error(f"Failed {endpoint} after {attempt} attempts: {err}")
                    raise
                backoff = attempt * 1.5
                logger.warning(f"Attempt {attempt} failed ({err}). Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
        raise RuntimeError("Unreachable retry loop exit")

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        """Fetch historical 1-minute klines from Binance Public Futures endpoint with .cache/ caching."""
        cache_key = None
        if self.cache is not None and start_time_ms is not None:
            cache_key = self.cache.build_kline_key(symbol, interval, start_time_ms, end_time_ms, limit)
            cached = self.cache.get("klines", cache_key)
            if cached is not None:
                return cached

        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)

        raw = self._get_with_retry("/fapi/v1/klines", params)
        if not raw:
            return pd.DataFrame()

        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ]
        df = pd.DataFrame(raw, columns=cols)
        
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        num_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]
        for col in num_cols:
            df[col] = df[col].astype(float)
        df["count"] = df["count"].astype(int)
        df["symbol"] = symbol.upper()

        result = df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]]
        if self.cache is not None and cache_key is not None and not result.empty:
            self.cache.put("klines", cache_key, result)

        return result

    def fetch_funding_history(
        self,
        symbol: str,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch 8-hour funding rate history with .cache/ caching."""
        cache_key = None
        if self.cache is not None and start_time_ms is not None:
            cache_key = self.cache.build_funding_key(symbol, start_time_ms, end_time_ms, limit)
            cached = self.cache.get("funding", cache_key)
            if cached is not None:
                return cached

        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "limit": limit,
        }
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)

        raw = self._get_with_retry("/fapi/v1/fundingRate", params)
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df["symbol"] = symbol.upper()
        result = df[["timestamp", "symbol", "funding_rate"]].sort_values("timestamp")

        if self.cache is not None and cache_key is not None and not result.empty:
            self.cache.put("funding", cache_key, result)

        return result

    def fetch_open_interest_hist(
        self,
        symbol: str,
        period: str = "5m",
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch historical open interest snapshots with .cache/ caching."""
        cache_key = None
        if self.cache is not None and start_time_ms is not None:
            cache_key = self.cache.build_oi_key(symbol, period, start_time_ms, end_time_ms, limit)
            cached = self.cache.get("oi", cache_key)
            if cached is not None:
                return cached

        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "period": period,
            "limit": limit,
        }
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)

        raw = self._get_with_retry("/futures/data/openInterestHist", params)
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["open_interest"] = df["sumOpenInterest"].astype(float)
        df["open_interest_usd"] = df["sumOpenInterestValue"].astype(float)
        df["symbol"] = symbol.upper()
        result = df[["timestamp", "symbol", "open_interest", "open_interest_usd"]].sort_values("timestamp")

        if self.cache is not None and cache_key is not None and not result.empty:
            self.cache.put("oi", cache_key, result)

        return result


def generate_synthetic_data(
    symbol: str,
    start: datetime,
    periods: int,
    base_price: float,
    volatility: float,
    drift: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic 1-minute OHLCV, OI, and funding for offline testing."""
    np.random.seed(seed)
    timestamps = pd.date_range(start=start, periods=periods, freq="1min", tz="UTC")
    
    # Geometric Brownian Motion with student-t innovations
    dt = 1.0 / (365 * 24 * 60)
    returns = np.random.standard_t(df=5, size=periods) * volatility * np.sqrt(dt) + drift * dt
    price_series = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    intra_vol = volatility * np.sqrt(dt) * 0.5
    highs = price_series * (1.0 + np.abs(np.random.normal(0, intra_vol, size=periods)))
    lows = price_series * (1.0 - np.abs(np.random.normal(0, intra_vol, size=periods)))
    opens = np.roll(price_series, 1)
    opens[0] = base_price
    closes = price_series
    
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    
    # Volume
    base_vol = 1000.0 * (1.0 / base_price)
    volumes = np.random.lognormal(mean=np.log(base_vol), sigma=0.8, size=periods)
    quote_volumes = volumes * closes
    
    # Taker buy volume (fraction between 0.35 and 0.65)
    buy_ratio = np.clip(0.5 + 0.3 * (closes - opens) / (highs - lows + 1e-8), 0.2, 0.8)
    taker_buy_volumes = volumes * buy_ratio
    taker_buy_quote_volumes = quote_volumes * buy_ratio
    
    # Funding rate: changes every 480 minutes (8 hours)
    n_funding_intervals = (periods // 480) + 1
    funding_rates_raw = np.random.normal(0.0001, 0.0002, size=n_funding_intervals)
    funding_rates = np.repeat(funding_rates_raw, 480)[:periods]
    
    # Open interest: random walk around mean
    oi_base = 50000000.0
    oi_innovations = np.random.normal(0, 0.001, size=periods)
    open_interest = oi_base * np.exp(np.cumsum(oi_innovations))
    open_interest_usd = open_interest * closes
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": symbol,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "quote_volume": quote_volumes,
        "count": np.random.poisson(lam=120, size=periods),
        "taker_buy_volume": taker_buy_volumes,
        "taker_buy_quote_volume": taker_buy_quote_volumes,
        "funding_rate": funding_rates,
        "open_interest": open_interest,
        "open_interest_usd": open_interest_usd,
    })
    return df
