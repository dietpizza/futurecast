"""Continuous real-time public market data streamer (Read-Only / Public REST & WS)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import AsyncGenerator, Callable, List, Optional
import httpx
import pandas as pd
import numpy as np

from futurecast.ingestion.storage import align_cross_asset_data

logger = logging.getLogger(__name__)


class ContinuousKlineStreamer:
    """Streams continuous 1-minute closed bars for target and benchmarks in real-time.
    
    Uses high-frequency asynchronous public REST polling (or WebSocket) aligned to the
    minute boundary. Strictly read-only, zero authentication.
    """

    def __init__(
        self,
        target_symbol: str = "DOGEUSDT",
        benchmark_symbols: Optional[List[str]] = None,
        base_url: Optional[str] = None,
        poll_interval_sec: float = 3.0,
        exchange: str = "binance",
    ) -> None:
        self.exchange = exchange.lower()
        self.target_symbol = target_symbol.strip()
        self.benchmark_symbols = [b.strip() for b in (benchmark_symbols or ["BTCUSDT", "ETHUSDT"])]
        self.all_symbols = [self.target_symbol] + self.benchmark_symbols
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = "https://public.coindcx.com" if self.exchange == "coindcx" else "https://fapi.binance.com"
        self.poll_interval_sec = poll_interval_sec
        self.last_processed_timestamp: Optional[datetime] = None
        self._running = False

    async def _fetch_single_asset_latest(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        limit: int = 5,
    ) -> pd.DataFrame:
        """Fetch the most recent klines for a single symbol from Binance or CoinDCX."""
        if self.exchange == "coindcx":
            url = f"{self.base_url}/market_data/candles"
            params = {"pair": symbol, "interval": "1m", "limit": limit}
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            raw = resp.json()
            df = pd.DataFrame(raw)
            if df.empty or "time" not in df.columns:
                return pd.DataFrame()
            df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = df[c].astype(float)
            df["quote_volume"] = df["volume"] * df["close"]
            df["count"] = 100
            df["taker_buy_volume"] = df["volume"] * 0.5
            df["taker_buy_quote_volume"] = df["quote_volume"] * 0.5
            df["symbol"] = symbol
            return df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]].sort_values("timestamp").reset_index(drop=True)

        # Default: Binance
        url = f"{self.base_url}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": "1m", "limit": limit}
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        raw = resp.json()

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
        return df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]]

    async def stream_closed_bars(
        self,
        max_bars: Optional[int] = None,
    ) -> AsyncGenerator[pd.DataFrame, None]:
        """Continuously yield aligned 1-minute closed bars as soon as each minute finishes."""
        self._running = True
        bars_yielded = 0

        async with httpx.AsyncClient(timeout=10.0) as client:
            while self._running:
                try:
                    # Fetch all assets concurrently
                    tasks = [self._fetch_single_asset_latest(client, s, limit=5) for s in self.all_symbols]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    dfs: dict[str, pd.DataFrame] = {}
                    has_error = False
                    for symbol, res in zip(self.all_symbols, results):
                        if isinstance(res, Exception):
                            logger.warning(f"Error fetching {symbol}: {res}")
                            has_error = True
                            break
                        dfs[symbol] = res

                    if not has_error and len(dfs) == len(self.all_symbols):
                        # The last bar in klines is currently open. The second to last [-2] is the most recently closed bar.
                        target_df = dfs[self.target_symbol]
                        if len(target_df) >= 2:
                            latest_closed_bar = target_df.iloc[-2]
                            bar_time = latest_closed_bar["timestamp"]

                            # Check if this is a newly closed bar
                            if self.last_processed_timestamp is None or bar_time > self.last_processed_timestamp:
                                # Filter each asset for this exact closed bar timestamp
                                closed_dfs = {}
                                all_present = True
                                for s, df in dfs.items():
                                    match = df[df["timestamp"] == bar_time]
                                    if match.empty:
                                        all_present = False
                                        break
                                    closed_dfs[s] = match

                                if all_present:
                                    target_closed = closed_dfs[self.target_symbol]
                                    bench_closed = {k: v for k, v in closed_dfs.items() if k != self.target_symbol}
                                    aligned_bar = align_cross_asset_data(target_closed, bench_closed)

                                    if not aligned_bar.empty:
                                        self.last_processed_timestamp = bar_time
                                        bars_yielded += 1
                                        yield aligned_bar

                                        if max_bars is not None and bars_yielded >= max_bars:
                                            self._running = False
                                            break

                except Exception as err:
                    logger.error(f"Error in continuous streaming loop: {err}")

                await asyncio.sleep(self.poll_interval_sec)

    def stop(self) -> None:
        self._running = False
