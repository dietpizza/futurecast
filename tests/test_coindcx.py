"""Tests for CoinDCX symbol handling: exchange detection and benchmark routing."""

from futurecast.config import load_config
from futurecast.ingestion.factory import (
    detect_exchange,
    resolve_benchmarks,
    sanitize_symbol_for_filename,
)
from futurecast.publisher.multi_manager import SymbolWorker


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
    assert benchmarks_inr == ["I-BTC_INR", "I-SOL_INR"]

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


def test_coin_is_never_its_own_benchmark() -> None:
    # BTC benchmarked against itself duplicated its candle columns and training failed
    assert resolve_benchmarks("B-BTC_USDT", "coindcx") == ["B-ETH_USDT", "B-SOL_USDT"]
    assert resolve_benchmarks("B-ETH_USDT", "coindcx") == ["B-BTC_USDT", "B-SOL_USDT"]
    assert resolve_benchmarks("BTCUSDT", "binance", sector_aware=False) == ["ETHUSDT", "SOLUSDT"]
    for sym in ("B-BTC_USDT", "B-ETH_USDT", "B-SOL_USDT", "B-DOGE_USDT", "B-BLUR_USDT", "I-BTC_INR", "B-PEPE_USDT"):
        benchmarks = resolve_benchmarks(sym, "coindcx")
        assert sym not in benchmarks and len(set(benchmarks)) == 2


def test_btc_features_build(tmp_path) -> None:
    worker = SymbolWorker("B-BTC_USDT", "coindcx", resolve_benchmarks("B-BTC_USDT", "coindcx"),
                          "MAJOR", load_config(), tmp_path / "models")
    worker.load_synthetic()

    feats = worker.pipes["1m"].generate_features(worker.candles["1m"])
    assert feats.columns.is_unique
    assert worker.latest_event is not None  # the live engine produced a forecast, so the order book shows


def test_sanitize_symbol() -> None:
    assert sanitize_symbol_for_filename("B-DOGE_USDT") == "b_doge_usdt"
    assert sanitize_symbol_for_filename("I-BTC_INR") == "i_btc_inr"
    assert sanitize_symbol_for_filename("DOGEUSDT") == "dogeusdt"

