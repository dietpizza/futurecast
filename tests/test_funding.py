"""Funding rates: Binance download, point-in-time merge onto candles, and the funding signal."""

import math
from typing import List

import httpx
import pandas as pd
import pytest

from deltarisk.config import load_config
from deltarisk.ingestion import binance
from deltarisk.ingestion.binance import BinancePublicClient, interval_to_ms
from deltarisk.ingestion.factory import resolve_benchmark
from deltarisk.publisher.multi_manager import CHART_MINUTES, SymbolWorker

MINUTE, HOUR, DAY = 60_000, 3_600_000, 86_400_000
SETTLE = 1_700_006_400_000  # a real 8h BTCUSDT funding settlement (16:00 UTC)
NOW = SETTLE + 30_000  # the candles closing at the settlement have just closed
SYMBOL = "B-SOL_USDT"


def _phase(market: str) -> int:
    return sum(map(ord, market)) % 17


def _price(market: str, t: int) -> float:
    x, p = t / (5 * MINUTE), _phase(market)
    return 100 * math.exp(0.02 * math.sin(x / 37 + p) + 0.01 * math.sin(x / 11 + 2 * p) + 0.004 * math.sin(x / 3.1 + p))


def rate(market: str, t: int) -> float:
    """Funding settled at `t`: different every settlement and per coin, never zero; 8 decimals like Binance."""
    return round(1e-4 * math.sin(t / (8 * HOUR) * 0.7 + _phase(market)) + 2e-5, 8)


class Clock:
    """Stands in for the `time` module inside the Binance client."""

    def __init__(self, now_ms: int) -> None:
        self.now_ms = now_ms
        self.slept: List[float] = []

    def time(self) -> float:
        return self.now_ms / 1000

    def monotonic(self) -> float:
        return self.now_ms / 1000

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


class FakeBinance:
    """exchangeInfo, klines up to the forming candle and funding settled every `funding_hours`."""

    def __init__(self, clock: Clock, funding_hours: int = 8, listed_ms: int = NOW - 120 * DAY) -> None:
        self.clock = clock
        self.funding_step = funding_hours * HOUR
        self.listed_ms = listed_ms
        self.calls: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        params, now = request.url.params, self.clock.now_ms
        if request.url.path == "/fapi/v1/exchangeInfo":
            return httpx.Response(200, json={"symbols": [
                {"symbol": s, "status": "TRADING", "contractType": "PERPETUAL"}
                for s in ("SOLUSDT", "BTCUSDT", "ETHUSDT")
            ]})
        if request.url.path == "/fapi/v1/klines":
            step, market = interval_to_ms(params["interval"]), params["symbol"]
            last = min(int(params["endTime"]) // step * step, now // step * step)
            first = max(-(-self.listed_ms // step) * step, last - (int(params["limit"]) - 1) * step)
            rows = []
            for t in range(first, last + step, step):
                o, c = _price(market, t), _price(market, t + step)
                rows.append([t, str(o), str(max(o, c) * 1.001), str(min(o, c) * 0.999), str(c), "1000",
                             t + step - 1, str(1000 * c), 50, "500", str(500 * c), "0"])
            return httpx.Response(200, json=rows)
        if request.url.path == "/fapi/v1/fundingRate":
            start = max(int(params["startTime"]), self.listed_ms)
            first = -(-start // self.funding_step) * self.funding_step
            times = list(range(first, now + 1, self.funding_step))[: int(params["limit"])]
            return httpx.Response(200, json=[
                {"symbol": params["symbol"], "fundingTime": t, "fundingRate": f"{rate(params['symbol'], t):.8f}",
                 "markPrice": "100.0", "rateType": "Regular"} for t in times
            ])
        return httpx.Response(404)

    def funding_calls(self) -> List[httpx.Request]:
        return [r for r in self.calls if r.url.path == "/fapi/v1/fundingRate"]


@pytest.fixture
def clock(monkeypatch) -> Clock:
    c = Clock(NOW)
    monkeypatch.setattr(binance, "time", c)
    return c


def _client(fake: FakeBinance, **kwargs) -> BinancePublicClient:
    client = BinancePublicClient(**kwargs)
    client._client = httpx.Client(base_url=binance.BINANCE_FAPI_URL, transport=httpx.MockTransport(fake))
    return client


def _worker() -> SymbolWorker:
    return SymbolWorker(SYMBOL, resolve_benchmark(SYMBOL), "L1", load_config())


def test_funding_rates_page_forward_and_reuse_downloads(clock) -> None:
    fake = FakeBinance(clock, funding_hours=1)
    client = _client(fake)
    start = NOW - 50 * DAY

    events = client.fetch_funding_rates(SYMBOL, start)

    first = (start // DAY * DAY + HOUR - 1) // HOUR * HOUR
    expected = list(range(first, NOW + 1, HOUR))
    assert len(expected) > 1000 and len(fake.funding_calls()) == 2  # 1000-row pages walking forward
    assert [t.value // 1_000_000 for t in events["timestamp"]] == expected
    assert events["funding_rate"].iloc[-1] == pytest.approx(rate("SOLUSDT", expected[-1]))

    # Nothing can have settled since the download, so a request up to now is served from it
    recent = client.fetch_funding_rates(SYMBOL, NOW - DAY, NOW)  # start rounds down to the UTC day
    assert len(recent) == sum(t >= (NOW - DAY) // DAY * DAY for t in expected)
    assert len(fake.funding_calls()) == 2

    # An hour later only the settlements after the newest one held are fetched
    clock.now_ms += HOUR
    events = client.fetch_funding_rates(SYMBOL, start)
    assert len(fake.funding_calls()) == 3
    assert int(fake.funding_calls()[-1].url.params["startTime"]) == expected[-1] + 1
    assert events["timestamp"].iloc[-1].value // 1_000_000 == expected[-1] + HOUR


def test_funding_downloads_survive_a_restart(clock, tmp_path) -> None:
    from deltarisk.ingestion.cache import FundingCache

    fake = FakeBinance(clock, funding_hours=1)
    start = NOW - 50 * DAY
    first = _client(fake, funding_cache=FundingCache(tmp_path / "funding")).fetch_funding_rates(SYMBOL, start)
    assert len(fake.funding_calls()) == 2 and (tmp_path / "funding" / "SOLUSDT.parquet").exists()

    # A new client (a restarted app) at the same moment reads the file and downloads nothing
    restarted = _client(fake, funding_cache=FundingCache(tmp_path / "funding"))
    pd.testing.assert_frame_equal(restarted.fetch_funding_rates(SYMBOL, start), first)
    assert len(fake.funding_calls()) == 2

    # An hour later it fetches only the settlement since
    clock.now_ms += HOUR
    later = _client(fake, funding_cache=FundingCache(tmp_path / "funding")).fetch_funding_rates(SYMBOL, start)
    assert len(fake.funding_calls()) == 3 and len(later) == len(first) + 1
    assert int(fake.funding_calls()[-1].url.params["startTime"]) == first["timestamp"].iloc[-1].value // 1_000_000 + 1

    # A request reaching further back than the file covers downloads again from there
    _client(fake, funding_cache=FundingCache(tmp_path / "funding")).fetch_funding_rates(SYMBOL, start - 30 * DAY)
    older = [int(r.url.params["startTime"]) for r in fake.funding_calls()[3:]]
    assert older[0] == (start - 30 * DAY) // DAY * DAY


def test_funding_requests_stay_within_their_own_budget(clock) -> None:
    client = BinancePublicClient(funding_budget=2)
    for _ in range(3):
        client._funding_slot()
    assert clock.slept == [300.0]  # the third request waits for the first to leave the 5-minute window


def test_candles_carry_the_rate_settled_before_they_closed(clock) -> None:
    client = _client(FakeBinance(clock))
    step = 5 * MINUTE

    candles, _ = client.fetch_candle_history(SYMBOL, "5m", bars=400, funding=True)

    close = candles["timestamp"].map(lambda t: t.value // 1_000_000) + step
    settled = close.map(lambda c: (c - 1) // (8 * HOUR) * (8 * HOUR))
    assert candles["funding_rate"].tolist() == pytest.approx([rate("SOLUSDT", s) for s in settled])
    # The candle closing at a settlement still shows the previous rate; the next one the new rate
    at = close[close == SETTLE - 8 * HOUR].index[0]
    assert candles["funding_rate"][at] == pytest.approx(rate("SOLUSDT", SETTLE - 16 * HOUR))
    assert candles["funding_rate"][at + 1] == pytest.approx(rate("SOLUSDT", SETTLE - 8 * HOUR))
    assert close.iloc[-1] == SETTLE


def test_history_and_live_candles_carry_the_settled_funding_rate(clock, monkeypatch) -> None:
    fake = FakeBinance(clock)
    client = _client(fake)
    worker = _worker()
    bench = worker.benchmark
    train = SymbolWorker.train
    monkeypatch.setattr(SymbolWorker, "train", lambda self, lgbm=True, progress=False: train(self, False, progress))

    worker.load_history(client)

    assert worker.status == "ready" and worker.progress is None
    assert worker.risk_view["8h"]["long"]["model"] == "har"  # trained after the download (baseline only here)
    funding_pct = worker.risk.windows["1h"].funding_pct
    assert funding_pct is not None and 0 <= funding_pct <= 100
    hourly = worker.candles["1h"]
    assert f"{SYMBOL}_funding_rate" in hourly and f"{bench}_funding_rate" not in hourly
    assert f"{SYMBOL}_funding_rate" not in worker.candles["15m"]  # only the coin's, only where it is read
    # The candle closing at the settlement still carries the previous rate
    assert worker.funding["rate_pct"] == pytest.approx(100 * rate("SOLUSDT", SETTLE - 8 * HOUR))

    # An hour later both charts gain a candle; the new hourly one carries the new rate, fetched incrementally
    clock.now_ms += HOUR
    before = len(fake.funding_calls())
    assert worker.refresh_candles(client, now=pd.Timestamp(clock.now_ms, unit="ms", tz="UTC")) == list(CHART_MINUTES)
    assert worker.funding["rate_pct"] == pytest.approx(100 * rate("SOLUSDT", SETTLE))
    new_calls = fake.funding_calls()[before:]
    assert len(new_calls) == 1 and int(new_calls[0].url.params["startTime"]) > SETTLE - 8 * HOUR


def test_offline_candles_need_no_funding_download() -> None:
    worker = _worker()
    worker.load_synthetic()

    assert worker.funding is None
    assert worker.risk_view["4h"]["long"]["score"] is not None
