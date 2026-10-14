"""Tests for the Binance market data client: CoinDCX pair mapping, candles, prices, order book."""

from typing import List

import httpx
import pytest

from futurecast.ingestion import binance
from futurecast.ingestion.binance import BinanceMarket, BinancePublicClient, binance_candidates

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


def _client(handler) -> BinancePublicClient:
    client = BinancePublicClient()
    client._client = httpx.Client(base_url=binance.BINANCE_FAPI_URL, transport=httpx.MockTransport(handler))
    return client


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(binance.time, "sleep", lambda s: None)


class FakeBinance:
    """Serves exchangeInfo and a 5m kline series that runs up to the still-forming candle."""

    def __init__(self, first_open_ms: int = NOW // STEP * STEP - 4000 * STEP) -> None:
        self.first_open_ms = first_open_ms
        self.calls: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if request.url.path == "/fapi/v1/exchangeInfo":
            return httpx.Response(200, json=EXCHANGE_INFO)
        if request.url.path == "/fapi/v1/klines":
            end = int(request.url.params["endTime"])
            limit = int(request.url.params["limit"])
            forming = NOW // STEP * STEP
            opens = [t for t in range(self.first_open_ms, forming + STEP, STEP) if t <= end][-limit:]
            return httpx.Response(200, json=[_kline(t) for t in opens])
        if request.url.path == "/fapi/v2/ticker/price":
            return httpx.Response(200, json=[{"symbol": "SOLUSDT", "price": "150.5"},
                                             {"symbol": "1000PEPEUSDT", "price": "0.004"}])
        if request.url.path == "/fapi/v1/depth":
            return httpx.Response(200, json={"E": 123, "bids": [["0.0039", "10"], ["0.0040", "5"]],
                                             "asks": [["0.0042", "7"], ["0.0041", "3"]]})
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


def test_orderbook_sorted_and_scaled() -> None:
    client = _client(FakeBinance())

    book = client.fetch_orderbook_depth("B-1000PEPE_USDT", limit=20)
    assert book["source"] == "binance:1000PEPEUSDT"
    assert book["bids"] == [(0.0040, 5.0), (0.0039, 10.0)]
    assert book["asks"] == [(0.0041, 3.0), (0.0042, 7.0)]

    scaled = client.fetch_orderbook_depth("B-PEPE_USDT", limit=1)
    assert scaled["bids"][0] == pytest.approx((0.0040 / 1000, 5_000.0))

    assert client.fetch_orderbook_depth("B-NOPE_USDT")["source"] == "empty"


def test_rate_limit_is_retried() -> None:
    replies = iter([httpx.Response(429, headers={"Retry-After": "1"}),
                    httpx.Response(200, json=EXCHANGE_INFO, headers={"X-MBX-USED-WEIGHT-1M": "5"})])
    client = _client(lambda request: next(replies))

    assert client.resolve("B-SOL_USDT") == BinanceMarket("SOLUSDT", 1)
