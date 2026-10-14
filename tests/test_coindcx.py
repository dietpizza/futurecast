"""Tests for CoinDCX integration, exchange detection, and benchmark routing."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from futurecast.ingestion.cache import DataCache
from futurecast.ingestion.coindcx import CoinDCXPublicClient
from futurecast.ingestion.factory import (
    detect_exchange,
    get_exchange_client,
    resolve_benchmarks,
    sanitize_symbol_for_filename,
)


def test_detect_exchange() -> None:
    assert detect_exchange("B-DOGE_USDT") == "coindcx"
    assert detect_exchange("B-SOL_USDT") == "coindcx"
    assert detect_exchange("I-BTC_INR") == "coindcx"
    assert detect_exchange("DOGEUSDT") == "binance"
    assert detect_exchange("BTCUSDT") == "binance"
    assert detect_exchange("SOLUSDT") == "binance"


def test_resolve_benchmarks() -> None:
    benchmarks_usdt = resolve_benchmarks("B-DOGE_USDT", "coindcx")
    assert benchmarks_usdt == ["B-BTC_USDT", "B-ETH_USDT"]

    benchmarks_inr = resolve_benchmarks("I-ETH_INR", "coindcx")
    assert benchmarks_inr == ["I-BTC_INR", "I-ETH_INR"]

    benchmarks_binance = resolve_benchmarks("DOGEUSDT", "binance")
    assert benchmarks_binance == ["BTCUSDT", "ETHUSDT"]


def test_dynamic_sector_benchmarks() -> None:
    from futurecast.ingestion.factory import get_symbol_sector

    # Meme tokens benchmark against DOGE
    assert get_symbol_sector("PEPEUSDT") == "MEME"
    assert resolve_benchmarks("PEPEUSDT", "binance") == ["DOGEUSDT", "BTCUSDT"]
    assert resolve_benchmarks("B-SHIB_USDT", "coindcx") == ["B-DOGE_USDT", "B-BTC_USDT"]

    # Alt-L1 tokens benchmark against SOL
    assert get_symbol_sector("AVAXUSDT") == "L1"
    assert resolve_benchmarks("AVAXUSDT", "binance") == ["SOLUSDT", "BTCUSDT"]
    assert resolve_benchmarks("B-SUI_USDT", "coindcx") == ["B-SOL_USDT", "B-BTC_USDT"]

    # DeFi tokens benchmark against ETH
    assert get_symbol_sector("UNIUSDT") == "DEFI"
    assert resolve_benchmarks("UNIUSDT", "binance") == ["ETHUSDT", "BTCUSDT"]
    assert resolve_benchmarks("B-AAVE_USDT", "coindcx") == ["B-ETH_USDT", "B-BTC_USDT"]

    # NFT / Gaming tokens benchmark against BLUR
    assert get_symbol_sector("SANDUSDT") == "NFT_GAMING"
    assert resolve_benchmarks("SANDUSDT", "binance") == ["BLURUSDT", "BTCUSDT"]

    # AI tokens benchmark against NEAR
    assert get_symbol_sector("RENDERUSDT") == "AI"
    assert resolve_benchmarks("RENDERUSDT", "binance") == ["NEARUSDT", "BTCUSDT"]

    # Sector leaders benchmark against market majors (BTC, ETH)
    assert resolve_benchmarks("SOLUSDT", "binance") == ["BTCUSDT", "ETHUSDT"]
    assert resolve_benchmarks("B-BLUR_USDT", "coindcx") == ["B-BTC_USDT", "B-ETH_USDT"]

    # Fallback when sector_aware is explicitly False
    assert resolve_benchmarks("PEPEUSDT", "binance", sector_aware=False) == ["BTCUSDT", "ETHUSDT"]


def test_sanitize_symbol() -> None:
    assert sanitize_symbol_for_filename("B-DOGE_USDT") == "b_doge_usdt"
    assert sanitize_symbol_for_filename("I-BTC_INR") == "i_btc_inr"
    assert sanitize_symbol_for_filename("DOGEUSDT") == "dogeusdt"


def test_coindcx_fetch_klines_mock(tmp_path) -> None:
    cache = DataCache(cache_dir=str(tmp_path), enabled=True)
    client = CoinDCXPublicClient(cache=cache)

    mock_candles = [
        {"open": 0.08, "high": 0.081, "low": 0.079, "close": 0.08, "volume": 100.0, "time": 1700000060000},
        {"open": 0.081, "high": 0.082, "low": 0.08, "close": 0.081, "volume": 150.0, "time": 1700000000000},
    ]

    with patch.object(client, "_get_with_retry", return_value=mock_candles):
        df = client.fetch_klines("B-DOGE_USDT", limit=10)

    assert not df.empty
    assert len(df) == 2
    assert "timestamp" in df.columns
    assert "open" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    assert "taker_buy_volume" in df.columns
    # Check that df was sorted chronologically ascending
    assert df["timestamp"].iloc[0] < df["timestamp"].iloc[1]


def test_coindcx_fetch_klines_pagination_mock(tmp_path) -> None:
    cache = DataCache(cache_dir=str(tmp_path), enabled=False)
    client = CoinDCXPublicClient(cache=cache)

    # Page 1 (newer)
    page1 = [
        {"open": 0.082, "high": 0.083, "low": 0.081, "close": 0.082, "volume": 120.0, "time": 1700000120000},
        {"open": 0.081, "high": 0.082, "low": 0.080, "close": 0.081, "volume": 110.0, "time": 1700000060000},
    ]
    # Page 2 (older)
    page2 = [
        {"open": 0.080, "high": 0.081, "low": 0.079, "close": 0.080, "volume": 100.0, "time": 1700000000000},
    ]

    with patch.object(client, "_get_with_retry", side_effect=[page1, page2]):
        # Request span that triggers pagination
        df = client.fetch_klines(
            "B-DOGE_USDT",
            start_time_ms=1700000000000,
            end_time_ms=1700000120000,
            limit=2,
        )

    assert len(df) == 3
    assert df["timestamp"].iloc[0] < df["timestamp"].iloc[-1]
