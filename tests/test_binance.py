"""Tests for the Binance market data client: CoinDCX pair mapping, candles, prices."""

from typing import List

import httpx
import pandas as pd
import pytest

from deltarisk.ingestion import binance
from deltarisk.ingestion.binance import BinanceMarket, BinancePublicClient, binance_candidates
from deltarisk.ingestion.cache import CandleCache

STEP = 300_000  # 5m
NOW = 1_700_000_000_000 // STEP * STEP + 120_000  # two minutes into a 5m candle

EXCHANGE_INFO = {"symbols": [
    {"symbol": "SOLUSDT", "status": "TRADING", "contractType": "PERPETUAL"},
    {"symbol": "1000PEPEUSDT", "status": "TRADING", "contractType": "PERPETUAL"},
    {"symbol": "XAUUSDT", "status": "TRADING", "contractType": "TRADIFI_PERPETUAL"},
    {"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL"},
    {"symbol": "FTMUSDT", "status": "SETTLING", "contractType": "PERPETUAL"},
    {"symbol": "SOLUSDT_261225", "status": "TRADING", "contractType": "CURRENT_QUARTER"},
]}


def _kline(open_ms: int) -> list:
    px = 100.0 + open_ms / STEP % 7
    return [open_ms, str(px), str(px + 1), str(px - 1), str(px + 0.5), "10", open_ms + STEP - 1,
            "1000", 42, "6", "600", "0"]


def _client(handler, cache: CandleCache | None = None) -> BinancePublicClient:
    client = BinancePublicClient(candle_cache=cache)
    client._client = httpx.Client(base_url=binance.BINANCE_FAPI_URL, transport=httpx.MockTransport(handler))
    return client


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(binance.time, "sleep", lambda s: None)


class FakeBinance:
    """Serves exchangeInfo and a 5m kline series that runs up to the still-forming candle."""

    def __init__(self, first_open_ms: int = NOW // STEP * STEP - 4000 * STEP, now_ms: int = NOW) -> None:
        self.first_open_ms = first_open_ms
        self.now_ms = now_ms
        self.calls: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if request.url.path == "/fapi/v1/exchangeInfo":
            return httpx.Response(200, json=EXCHANGE_INFO)
        if request.url.path == "/fapi/v1/klines":
            end = int(request.url.params["endTime"])
            limit = int(request.url.params["limit"])
            forming = self.now_ms // STEP * STEP
            opens = [t for t in range(self.first_open_ms, forming + STEP, STEP) if t <= end][-limit:]
            return httpx.Response(200, json=[_kline(t) for t in opens])
        if request.url.path == "/fapi/v2/ticker/price":
            return httpx.Response(200, json=[{"symbol": "SOLUSDT", "price": "150.5"},
                                             {"symbol": "1000PEPEUSDT", "price": "0.004"}])
        return httpx.Response(404)

    def count(self, path: str) -> int:
        return sum(r.url.path == path for r in self.calls)


def test_binance_candidates() -> None:
    assert binance_candidates("B-SOL_USDT")[0] == BinanceMarket("SOLUSDT", 1)
    assert binance_candidates("B-1000PEPE_USDT")[0] == BinanceMarket("1000PEPEUSDT", 1)
    assert BinanceMarket("1000PEPEUSDT", 1000) in binance_candidates("B-PEPE_USDT")
    assert binance_candidates("I-BTC_INR")[0] == BinanceMarket("BTCUSDT", 1)


def test_resolve_maps_coindcx_pairs_to_trading_perpetuals() -> None:
    fake = FakeBinance()
    client = _client(fake)

    assert client.resolve("B-SOL_USDT") == BinanceMarket("SOLUSDT", 1)
    assert client.resolve("B-1000PEPE_USDT") == BinanceMarket("1000PEPEUSDT", 1)
    assert client.resolve("B-PEPE_USDT") == BinanceMarket("1000PEPEUSDT", 1000)
    assert client.resolve("B-XAU_USDT") == BinanceMarket("XAUUSDT", 1)
    assert client.resolve("I-BTC_INR") == BinanceMarket("BTCUSDT", 1)
    assert client.resolve("B-FTM_USDT") is None  # delisted
    assert client.resolve("B-NOPE_USDT") is None
    assert fake.count("/fapi/v1/exchangeInfo") == 1


def test_candle_history_pages_back_from_last_closed_candle() -> None:
    fake = FakeBinance()
    client = _client(fake)

    df, source = client.fetch_candle_history("B-SOL_USDT", "5m", bars=2500, end_time_ms=NOW)

    assert source == "binance:SOLUSDT"
    assert len(df) == 2500
    assert fake.count("/fapi/v1/klines") == 3
    assert df["timestamp"].is_monotonic_increasing and df["timestamp"].is_unique
    last_closed = (NOW // STEP - 1) * STEP
    assert df["timestamp"].iloc[-1].value // 1_000_000 == last_closed
    assert (df["symbol"] == "B-SOL_USDT").all()
    assert df["taker_buy_volume"].iloc[0] == 6.0 and df["count"].iloc[0] == 42


def test_candle_history_stops_at_listing_date() -> None:
    fake = FakeBinance(first_open_ms=NOW // STEP * STEP - 300 * STEP)
    df, _ = _client(fake).fetch_candle_history("B-SOL_USDT", "5m", bars=2000, end_time_ms=NOW)

    assert len(df) == 300  # every closed candle since listing, not the forming one
    assert fake.count("/fapi/v1/klines") == 1


def test_cached_candles_are_not_downloaded_again(tmp_path) -> None:
    cache = CandleCache(tmp_path / "candles")
    fake = FakeBinance()
    first, _ = _client(fake, cache).fetch_candle_history("B-SOL_USDT", "5m", bars=2500, end_time_ms=NOW)
    assert fake.count("/fapi/v1/klines") == 3

    # A new client (as each training run makes) reads the same candles from disk
    fake = FakeBinance()
    again, _ = _client(fake, cache).fetch_candle_history("B-SOL_USDT", "5m", bars=2500, end_time_ms=NOW)
    assert fake.count("/fapi/v1/klines") == 0
    pd.testing.assert_frame_equal(again, first)

    # Ten candles later only those ten are fetched, in one request
    fake = FakeBinance(now_ms=NOW + 10 * STEP)
    later, _ = _client(fake, cache).fetch_candle_history("B-SOL_USDT", "5m", bars=2500, end_time_ms=NOW + 10 * STEP)
    assert [int(r.url.params["limit"]) for r in fake.calls if r.url.path == "/fapi/v1/klines"] == [10]
    assert len(later) == 2500
    assert later["timestamp"].iloc[-1].value // 1_000_000 == (NOW // STEP + 9) * STEP

    # Deeper history fetches only the older stretch (the cache already holds 2,510)
    fake = FakeBinance(now_ms=NOW + 10 * STEP)
    deeper, _ = _client(fake, cache).fetch_candle_history("B-SOL_USDT", "5m", bars=3000, end_time_ms=NOW + 10 * STEP)
    assert [int(r.url.params["limit"]) for r in fake.calls if r.url.path == "/fapi/v1/klines"] == [490]
    assert len(deeper) == 3000 and deeper["timestamp"].is_unique
    pd.testing.assert_frame_equal(deeper.tail(2500).reset_index(drop=True), later)


def test_cache_remembers_the_listing_date(tmp_path) -> None:
    cache = CandleCache(tmp_path / "candles")
    listed = NOW // STEP * STEP - 300 * STEP
    _client(FakeBinance(first_open_ms=listed), cache).fetch_candle_history("B-SOL_USDT", "5m", bars=2000, end_time_ms=NOW)

    fake = FakeBinance(first_open_ms=listed)
    df, _ = _client(fake, cache).fetch_candle_history("B-SOL_USDT", "5m", bars=2000, end_time_ms=NOW)
    assert len(df) == 300
    assert fake.count("/fapi/v1/klines") == 0  # nothing older exists, so nothing is asked for


def test_interrupted_download_keeps_what_it_fetched(tmp_path) -> None:
    cache = CandleCache(tmp_path / "candles")
    fake = FakeBinance()

    def flaky(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/fapi/v1/klines" and fake.count("/fapi/v1/klines") >= 2:
            return httpx.Response(400)
        return fake(request)

    with pytest.raises(httpx.HTTPStatusError):
        _client(flaky, cache).fetch_candle_history("B-SOL_USDT", "5m", bars=2500, end_time_ms=NOW)

    fake = FakeBinance()
    df, _ = _client(fake, cache).fetch_candle_history("B-SOL_USDT", "5m", bars=2500, end_time_ms=NOW)
    assert len(df) == 2500
    assert [int(r.url.params["limit"]) for r in fake.calls if r.url.path == "/fapi/v1/klines"] == [500]


def test_download_progress_counts_candles(tmp_path) -> None:
    seen = []
    _client(FakeBinance(), CandleCache(tmp_path / "candles")).fetch_candle_history(
        "B-SOL_USDT", "5m", bars=2500, end_time_ms=NOW, progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1000, 2500), (2000, 2500), (2500, 2500)]


def test_candle_history_without_binance_market() -> None:
    fake = FakeBinance()
    df, source = _client(fake).fetch_candle_history("B-NOPE_USDT", "5m", bars=100)

    assert df.empty and source is None
    assert fake.count("/fapi/v1/klines") == 0


def test_live_prices_in_pair_units() -> None:
    prices = _client(FakeBinance()).fetch_live_prices(["B-SOL_USDT", "B-1000PEPE_USDT", "B-PEPE_USDT", "B-NOPE_USDT"])

    assert prices["B-SOL_USDT"] == 150.5
    assert prices["B-1000PEPE_USDT"] == 0.004
    assert prices["B-PEPE_USDT"] == pytest.approx(0.000004)
    assert "B-NOPE_USDT" not in prices


def test_rate_limit_is_retried() -> None:
    replies = iter([httpx.Response(429, headers={"Retry-After": "1"}),
                    httpx.Response(200, json=EXCHANGE_INFO, headers={"X-MBX-USED-WEIGHT-1M": "5"})])
    client = _client(lambda request: next(replies))

    assert client.resolve("B-SOL_USDT") == BinanceMarket("SOLUSDT", 1)
