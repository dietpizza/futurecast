"""The add-pair drawer's market board: prices, volume and changes over several windows."""

import pytest
from starlette.testclient import TestClient

from deltarisk.ingestion.binance import BinanceMarket
from deltarisk.publisher.market_board import HISTORY_MINUTES, MarketBoard
from deltarisk.publisher.web_server import create_app

MIN = 60_000
NOW = 1_800_000_000_000  # a whole minute


class FakeClient:
    """Two perpetuals; SOL rose 1% a minute-ish, 1000PEPE flat. Closes every minute for the last 4h+."""

    def __init__(self, now_ms):
        self.now_ms = now_ms
        self.kline_calls = []

    def perpetuals(self):
        return {"SOLUSDT": {}, "1000PEPEUSDT": {}}

    def resolve(self, symbol):
        return {"B-SOL_USDT": BinanceMarket("SOLUSDT", 1), "B-PEPE_USDT": BinanceMarket("1000PEPEUSDT", 1000),
                "B-1000PEPE_USDT": BinanceMarket("1000PEPEUSDT", 1)}.get(symbol)

    def fetch_tickers_24h(self):
        return [
            {"symbol": "SOLUSDT", "lastPrice": "200", "priceChangePercent": "5.5", "quoteVolume": "7.1e8"},
            {"symbol": "1000PEPEUSDT", "lastPrice": "0.012", "priceChangePercent": "-2", "quoteVolume": "3e8"},
            {"symbol": "DELISTEDUSDT", "lastPrice": "1", "priceChangePercent": "0", "quoteVolume": "1"},
        ]

    def fetch_recent_closes(self, market, minutes):
        self.kline_calls.append(market)
        # SOL: 100 four hours ago, rising linearly to 200 now; PEPE flat
        out = []
        for k in range(minutes, 0, -1):
            t = self.now_ms - k * MIN
            out.append((t, 200 - 100 * k / 240 if market == "SOLUSDT" else 0.012))
        return out

    def close(self):
        pass


def _board(now_ms=NOW):
    client = FakeClient(now_ms)
    board = MarketBoard(lambda: client, clock=lambda: now_ms / 1000)
    board.client = client
    return board, client


def test_changes_over_every_window_from_backfill_and_ticker():
    board, client = _board()
    board.poll()
    board.backfill()
    assert client.kline_calls == []  # nothing asked for yet
    board.view(["B-SOL_USDT", "B-PEPE_USDT"])
    board.backfill()
    assert sorted(client.kline_calls) == ["1000PEPEUSDT", "SOLUSDT"]  # one klines call per market the page shows
    view = board.view(["B-SOL_USDT", "B-PEPE_USDT", "B-1000PEPE_USDT", "B-NOPE_USDT"])
    assert view["markets"] == 2 and view["backfilled"] == 2 and view["updated_at"] == NOW
    sol = view["coins"]["B-SOL_USDT"]
    assert sol["price"] == 200 and sol["volume_24h"] == 7.1e8
    assert sol["change"]["24h"] == 5.5
    assert sol["change"]["4h"] == pytest.approx(100.0)  # 100 → 200
    assert sol["change"]["1h"] == pytest.approx(100 * (200 / (200 - 100 * 60 / 240) - 1))
    assert sol["change"]["5m"] == pytest.approx(100 * (200 / (200 - 100 * 5 / 240) - 1))
    pepe = view["coins"]["B-PEPE_USDT"]
    assert pepe["price"] == pytest.approx(0.012 / 1000) and pepe["change"]["15m"] == pytest.approx(0.0)
    assert view["coins"]["B-1000PEPE_USDT"]["price"] == 0.012  # the 1000x listing in its own units
    assert "B-NOPE_USDT" not in view["coins"]


def test_windows_the_history_does_not_reach_are_none():
    board, _ = _board()
    board.poll()  # ticker only: one point, no backfill yet
    change = board.view(["B-SOL_USDT"])["coins"]["B-SOL_USDT"]["change"]
    assert change == {"5m": None, "15m": None, "1h": None, "4h": None, "24h": 5.5}


def test_change_needs_a_point_close_to_the_window_start():
    times, prices = [NOW - 20 * MIN, NOW - 4 * MIN], [100.0, 110.0]
    assert MarketBoard.change(times, prices, NOW, 5, 121.0) is None  # nothing near 5 minutes ago
    assert MarketBoard.change(times, prices, NOW, 4, 121.0) == pytest.approx(10.0)
    assert MarketBoard.change(times, prices, NOW, 21, 121.0) is None  # before the history starts
    assert HISTORY_MINUTES >= 240


def test_offline_has_no_market_data(tmp_path):
    app = create_app(initial_symbols=[], live_mode=False, persistence_path=tmp_path / "t.json")
    with TestClient(app) as client:
        out = client.get("/api/market").json()
    assert out["offline"] is True and out["coins"] == {} and out["windows"] == ["5m", "15m", "1h", "4h", "24h"]
