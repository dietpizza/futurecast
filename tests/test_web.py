"""Tests for Web UI, REST endpoints, and WebSocket broadcasting."""

import json
import pytest
from starlette.testclient import TestClient
from futurecast.publisher.web_server import create_app


@pytest.fixture
def test_app(tmp_path):
    app = create_app(
        initial_symbols=["B-DOGE_USDT", "B-SOL_USDT"],
        live_mode=False,
        persistence_path=tmp_path / "test_tracked_symbols.json",
    )
    return app


def test_web_serves_the_built_page(tmp_path):
    """/ serves web/dist/index.html (revalidated on every load); hashed assets may be cached for good."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<title>FutureCast</title><script src="/assets/index-abc.js"></script>')
    (dist / "assets" / "index-abc.js").write_text("console.log(1)")
    (dist / "favicon.svg").write_text("<svg/>")
    app = create_app(initial_symbols=[], live_mode=False, persistence_path=tmp_path / "t.json", web_dist=dist)
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "FutureCast" in page.text
        assert page.headers["cache-control"] == "no-cache"

        asset = client.get("/assets/index-abc.js")
        assert asset.status_code == 200
        assert "immutable" in asset.headers["cache-control"]
        assert client.get("/assets/missing.js").status_code == 404

        icon = client.get("/favicon.svg")
        assert icon.status_code == 200
        assert icon.headers["cache-control"] == "no-cache"


def test_web_without_a_build_says_how_to_make_one(tmp_path):
    app = create_app(initial_symbols=[], live_mode=False, persistence_path=tmp_path / "t.json",
                     web_dist=tmp_path / "missing")
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 503
        assert "mise run build" in page.text
        assert client.get("/api/coins").status_code == 200


def test_api_coindcx_futures_catalog(test_app):
    """The add-coin search gets every pair in one list and filters it in the page."""
    with TestClient(test_app) as client:
        response = client.get("/api/coindcx/futures")
        assert response.status_code == 200
        catalog = response.json()
        assert len(catalog) >= 30

        by_symbol = {item["symbol"]: item for item in catalog}
        assert {"B-DOGE_USDT", "B-SOL_USDT", "B-PEPE_USDT", "B-BTC_USDT", "B-ZK_USDT"} <= set(by_symbol)
        zk = by_symbol["B-ZK_USDT"]
        assert zk == {"symbol": "B-ZK_USDT", "base": "ZK", "name": "zkSync", "sector": zk["sector"],
                      "description": zk["description"], "coindcx_url": "https://coindcx.com/futures/B-ZK_USDT"}
        assert by_symbol["I-BTC_INR"]["coindcx_url"] == "https://coindcx.com/trade/I-BTC_INR"
        assert sum(item["sector"] == "MEME" for item in catalog) >= 5


def test_api_coins_serve_momentum_statistics(test_app):
    from futurecast.publisher.multi_manager import MOMENTUM_SPEC

    with TestClient(test_app) as client:
        response = client.get("/api/coins")
        assert response.status_code == 200
        coins = response.json()
        assert len(coins) >= 2

        doge_coin = next((c for c in coins if "DOGE" in c["symbol"]), None)
        assert doge_coin is not None
        assert doge_coin["sector"] == "MEME"

        assert any("SOL" in c["symbol"] for c in coins)

        for coin in coins:
            # Offline mode computes on synthetic candles straight away
            assert coin["has_data"] is True and coin["status"] == "ready"
            assert set(coin) == {"symbol", "sector", "has_data", "status", "progress", "error", "history",
                                 "current_price", "momentum", "signals", "orderbook_depth"}

            m = coin["momentum"]
            assert m["round_trip_pct"] == pytest.approx(MOMENTUM_SPEC.round_trip_pct)
            assert set(m["horizons"]) == {h.name for h in MOMENTUM_SPEC.horizons}
            for name, h in m["horizons"].items():
                assert h["signal"] in ("LONG", "SHORT", "NEUTRAL")
                assert h["reason"] in ("flat", "few", "noise", "fees", "unstable", "edge")
                assert (h["signal"] != "NEUTRAL") == (h["reason"] == "edge")
                assert isinstance(h["z"], float) and 0 <= h["percentile"] <= 100
                a = h["after"]
                assert a["ci_low_pct"] <= a["mean_pct"] <= a["ci_high_pct"]
                assert 1 <= a["bucket"] <= a["buckets"] == MOMENTUM_SPEC.quantiles
                assert h["persistence"]["character"] in ("trending", "mean_reverting", "random")
                assert 0 < h["flow"]["buy_pct"] < 100
                assert h["versus"]["symbol"] == "B-BTC_USDT"  # DOGE and SOL lead their sectors

            s = coin["signals"]
            assert 0 <= s["rsi"]["15m"] <= 100 and 0 <= s["rsi"]["1h"] <= 100
            assert s["ema"]["stack"] in ("up", "down", "mixed")
            assert all(0 <= r["position"] <= 1 for r in s["ranges"].values() if r)
            assert s["volatility"]["daily_pct"] > 0
            assert s["funding"] is None  # synthetic candles carry no funding


def test_reload_endpoint(test_app):
    with TestClient(test_app) as client:
        assert client.post("/api/coins/B-DOGE_USDT/reload").json() == {"status": "offline", "symbol": "B-DOGE_USDT"}
        assert client.post("/api/coins/B-NOPE_USDT/reload").status_code == 404


def test_api_add_and_delete_coin(test_app):
    with TestClient(test_app) as client:
        # Non-CoinDCX symbol must be rejected with 400
        rej_resp = client.post("/api/coins", json={"symbol": "BTCUSDT"})
        assert rej_resp.status_code == 400
        assert "Only CoinDCX symbols are supported" in rej_resp.json()["detail"]

        # Add new CoinDCX meme token (should route sector benchmark to DOGE)
        add_resp = client.post("/api/coins", json={"symbol": "B-PEPE_USDT"})
        assert add_resp.status_code == 200
        added = add_resp.json()
        assert added["symbol"] == "B-PEPE_USDT"
        assert added["sector"] == "MEME"
        assert added["momentum"]["horizons"]["4h"]["versus"]["symbol"] == "B-DOGE_USDT"  # meme coins against DOGE

        # Add NFT token B-BLUR_USDT: statistics straight away, nothing to train
        blur_resp = client.post("/api/coins", json={"symbol": "B-BLUR_USDT"})
        assert blur_resp.status_code == 200
        blur_added = blur_resp.json()
        assert blur_added["symbol"] == "B-BLUR_USDT"
        assert blur_added["status"] == "ready" and "4h" in blur_added["momentum"]["horizons"]

        # Check it is in active list
        get_resp = client.get("/api/coins")
        symbols = [c["symbol"] for c in get_resp.json()]
        assert "B-PEPE_USDT" in symbols
        assert "B-BLUR_USDT" in symbols

        # Delete tokens
        del_resp = client.delete("/api/coins/B-PEPE_USDT")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "removed"

        del_resp2 = client.delete("/api/coins/B-BLUR_USDT")
        assert del_resp2.status_code == 200
        assert del_resp2.json()["status"] == "removed"

        # Verify deletion
        get_resp2 = client.get("/api/coins")
        symbols2 = [c["symbol"] for c in get_resp2.json()]
        assert "B-PEPE_USDT" not in symbols2
        assert "B-BLUR_USDT" not in symbols2


def test_websocket_stream_initial_state(test_app):
    with TestClient(test_app) as client:
        with client.websocket_connect("/ws") as websocket:
            init_data = websocket.receive_text()
            payload = json.loads(init_data)
            assert payload["type"] == "INIT_STATE"
            assert len(payload["coins"]) >= 2

            # Test sending client action
            websocket.send_text(json.dumps({"action": "add_coin", "symbol": "B-BLUR_USDT"}))
            # No disconnect, stream remains healthy


def test_tracked_coins_persistence_across_app_reboots(tmp_path):
    """Test that coins added/removed are saved to disk and restored upon closing and reopening the app."""
    storage_file = tmp_path / "saved_coins.json"

    # 1. Start App 1 with initial symbols
    app1 = create_app(
        initial_symbols=["B-DOGE_USDT", "B-SOL_USDT"],
        live_mode=False,
        persistence_path=storage_file,
    )
    with TestClient(app1) as client1:
        # Add a new coin: B-TRUTH_USDT
        add_resp = client1.post("/api/coins", json={"symbol": "B-TRUTH_USDT"})
        assert add_resp.status_code == 200

        # Verify it was persisted to disk
        assert storage_file.exists()
        with open(storage_file, "r") as f:
            saved_data = json.load(f)
        assert "B-TRUTH_USDT" in saved_data["symbols"]
        assert "B-DOGE_USDT" in saved_data["symbols"]
        assert "B-SOL_USDT" in saved_data["symbols"]

    # 2. Simulate closing the app, then starting App 2 WITHOUT specifying initial_symbols
    # It must restore exactly what was saved previously
    app2 = create_app(
        initial_symbols=None,
        live_mode=False,
        persistence_path=storage_file,
    )
    with TestClient(app2) as client2:
        resp = client2.get("/api/coins")
        assert resp.status_code == 200
        coins = [c["symbol"] for c in resp.json()]
        assert "B-TRUTH_USDT" in coins
        assert "B-DOGE_USDT" in coins
        assert "B-SOL_USDT" in coins

        # Now remove B-DOGE_USDT
        del_resp = client2.delete("/api/coins/B-DOGE_USDT")
        assert del_resp.status_code == 200

    # 3. Simulate closing the app again, then starting App 3
    # B-DOGE_USDT must be gone, and B-TRUTH_USDT must still be present
    app3 = create_app(
        initial_symbols=None,
        live_mode=False,
        persistence_path=storage_file,
    )
    with TestClient(app3) as client3:
        resp3 = client3.get("/api/coins")
        coins3 = [c["symbol"] for c in resp3.json()]
        assert "B-DOGE_USDT" not in coins3
        assert "B-TRUTH_USDT" in coins3
        assert "B-SOL_USDT" in coins3



def test_watchlist_starts_empty_without_default_coins(tmp_path):
    """No saved coins means an empty watchlist, including after the user removes every coin."""
    storage_file = tmp_path / "saved_coins.json"

    with TestClient(create_app(live_mode=False, persistence_path=storage_file)) as client:
        assert client.get("/api/coins").json() == []
        assert client.post("/api/coins", json={"symbol": "B-WAL_USDT"}).status_code == 200

    with TestClient(create_app(live_mode=False, persistence_path=storage_file)) as client:
        assert [c["symbol"] for c in client.get("/api/coins").json()] == ["B-WAL_USDT"]
        assert client.delete("/api/coins/B-WAL_USDT").status_code == 200

    with TestClient(create_app(live_mode=False, persistence_path=storage_file)) as client:
        assert client.get("/api/coins").json() == []
