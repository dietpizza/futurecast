"""Tests for .cache/ persistent disk caching mechanism."""

from datetime import datetime, timezone
import pandas as pd
import numpy as np
import pytest
from futurecast.ingestion.cache import DataCache
from futurecast.ingestion.binance import BinancePublicClient


def test_data_cache_put_get(tmp_path):
    cache = DataCache(cache_dir=tmp_path / ".cache")
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="1min", tz="UTC"),
        "symbol": ["DOGEUSDT"] * 10,
        "close": np.linspace(0.1, 0.15, 10),
    })

    key = cache.build_kline_key("DOGEUSDT", "1m", 1700000000000, 1700000600000, 10)
    
    # Cache miss
    assert cache.get("klines", key) is None

    # Put into cache
    cache.put("klines", key, df)

    # Cache hit
    cached_df = cache.get("klines", key)
    assert cached_df is not None
    assert len(cached_df) == 10
    assert np.allclose(cached_df["close"].values, df["close"].values)


def test_data_cache_clear_and_stats(tmp_path):
    cache = DataCache(cache_dir=tmp_path / ".cache")
    df = pd.DataFrame({"a": [1, 2, 3]})

    cache.put("klines", "k1", df)
    cache.put("funding", "f1", df)

    stats = cache.get_stats()
    assert stats["file_count"] == 2
    assert "klines" in stats["categories"]
    assert "funding" in stats["categories"]

    # Clear specific category
    removed = cache.clear("klines")
    assert removed == 1
    stats_after = cache.get_stats()
    assert stats_after["file_count"] == 1

    # Clear all
    cache.clear()
    assert cache.get_stats()["file_count"] == 0


def test_binance_client_uses_cache(tmp_path, monkeypatch):
    cache = DataCache(cache_dir=tmp_path / ".cache")
    client = BinancePublicClient(cache=cache)

    http_calls = 0

    def mock_get_with_retry(endpoint, params):
        nonlocal http_calls
        http_calls += 1
        # Return mock kline row
        return [[
            1700000000000, "0.1", "0.1", "0.1", "0.1", "100",
            1700000060000, "10", 5, "50", "5", "0"
        ]]

    monkeypatch.setattr(client, "_get_with_retry", mock_get_with_retry)

    # First fetch: should execute HTTP call
    df1 = client.fetch_klines("DOGEUSDT", start_time_ms=1700000000000, end_time_ms=1700000060000, limit=1)
    assert http_calls == 1
    assert len(df1) == 1

    # Second fetch with same parameters: should hit cache and NOT make HTTP call
    df2 = client.fetch_klines("DOGEUSDT", start_time_ms=1700000000000, end_time_ms=1700000060000, limit=1)
    assert http_calls == 1  # Still 1 because it hit cache!
    assert len(df2) == 1
    assert np.allclose(df1["close"].values, df2["close"].values)
