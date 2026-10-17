"""Tests for CoinDCX symbol handling: exchange detection and benchmark routing."""

from futurecast.config import load_config
from futurecast.ingestion.factory import detect_exchange, get_symbol_sector, resolve_benchmark
from futurecast.publisher.multi_manager import SymbolWorker


def test_detect_exchange() -> None:
    assert detect_exchange("B-DOGE_USDT") == "coindcx"
    assert detect_exchange("B-SOL_USDT") == "coindcx"
    assert detect_exchange("I-BTC_INR") == "coindcx"
    assert detect_exchange("DOGEUSDT") == "binance"
    assert detect_exchange("BTCUSDT") == "binance"


def test_sector_benchmarks() -> None:
    # Meme tokens benchmark against DOGE
    assert get_symbol_sector("B-PEPE_USDT") == "MEME"
    assert resolve_benchmark("B-PEPE_USDT") == "B-DOGE_USDT"
    assert resolve_benchmark("B-SHIB_USDT") == "B-DOGE_USDT"

    # Alt-L1 tokens benchmark against SOL
    assert get_symbol_sector("B-AVAX_USDT") == "L1"
    assert resolve_benchmark("B-SUI_USDT") == "B-SOL_USDT"

    # DeFi tokens benchmark against ETH, NFT / Gaming against BLUR, AI against NEAR, infrastructure against LINK
    assert resolve_benchmark("B-AAVE_USDT") == "B-ETH_USDT"
    assert resolve_benchmark("B-SAND_USDT") == "B-BLUR_USDT"
    assert resolve_benchmark("B-RENDER_USDT") == "B-NEAR_USDT"
    assert resolve_benchmark("B-PYTH_USDT") == "B-LINK_USDT"

    # Sector leaders, majors and unlisted coins benchmark against BTC
    for sym in ("B-DOGE_USDT", "B-SOL_USDT", "B-BLUR_USDT", "B-LINK_USDT", "B-ETH_USDT", "B-WAL_USDT"):
        assert resolve_benchmark(sym) == "B-BTC_USDT"

    # INR pairs get the INR pair
    assert resolve_benchmark("I-ETH_INR") == "I-BTC_INR"


def test_coin_is_never_its_own_benchmark() -> None:
    # BTC benchmarked against itself duplicated its candle columns
    assert resolve_benchmark("B-BTC_USDT") == "B-ETH_USDT"
    assert resolve_benchmark("I-BTC_INR") == "I-ETH_INR"
    for sym in ("B-BTC_USDT", "B-ETH_USDT", "B-SOL_USDT", "B-DOGE_USDT", "B-BLUR_USDT", "B-NEAR_USDT", "B-LINK_USDT"):
        assert resolve_benchmark(sym) != sym


def test_btc_statistics_build_against_eth() -> None:
    worker = SymbolWorker("B-BTC_USDT", resolve_benchmark("B-BTC_USDT"), "MAJOR", load_config())
    worker.load_synthetic()

    assert all(df.columns.is_unique for df in worker.candles.values())
    assert worker.momentum["horizons"]["1d"]["versus"]["symbol"] == "B-ETH_USDT"
