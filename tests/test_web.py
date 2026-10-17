"""Tests for Web UI, REST endpoints, and WebSocket broadcasting."""

import json
import pytest
from starlette.testclient import TestClient
from deltarisk.publisher.web_server import create_app


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
    (dist / "index.html").write_text('<title>DeltaRisk</title><script src="/assets/index-abc.js"></script>')
    (dist / "assets" / "index-abc.js").write_text("console.log(1)")
    (dist / "favicon.svg").write_text("<svg/>")
    app = create_app(initial_symbols=[], live_mode=False, persistence_path=tmp_path / "t.json", web_dist=dist)
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "DeltaRisk" in page.text
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


def test_api_coins_serve_only_what_the_page_reads(test_app):
    with TestClient(test_app) as client:
        response = client.get("/api/coins")
        assert response.status_code == 200
        coins = response.json()
        assert {c["symbol"] for c in coins} == {"B-DOGE_USDT", "B-SOL_USDT"}
        assert next(c for c in coins if "DOGE" in c["symbol"])["sector"] == "MEME"

        for coin in coins:
            # Offline mode trains the baseline on synthetic candles straight away
            assert coin["has_data"] is True and coin["status"] == "ready"
            assert set(coin) == {"symbol", "sector", "has_data", "status", "progress", "error", "history",
                                 "current_price", "tier", "tier_auto", "leverage", "overrides", "risk", "funding"}
            assert set(coin["history"]) == {"5m", "15m", "1h"}


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
        assert added["tier"] == "shitcoin"  # meme coins

        # Add NFT token B-BLUR_USDT: offline, readings straight away
        blur_resp = client.post("/api/coins", json={"symbol": "B-BLUR_USDT"})
        assert blur_resp.status_code == 200
        blur_added = blur_resp.json()
        assert blur_added["symbol"] == "B-BLUR_USDT"
        assert blur_added["status"] == "ready" and "4h" in blur_added["risk"]["windows"]

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
        saved = [c["symbol"] for c in saved_data["symbols"]]
        assert "B-TRUTH_USDT" in saved
        assert "B-DOGE_USDT" in saved
        assert "B-SOL_USDT" in saved

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


def test_api_coins_serve_risk_readings(test_app):
    from deltarisk.stats.regime import WINDOWS

    with TestClient(test_app) as client:
        coins = {c["symbol"]: c for c in client.get("/api/coins").json()}
        doge, sol = coins["B-DOGE_USDT"], coins["B-SOL_USDT"]
        assert (doge["tier"], doge["leverage"]) == ("shitcoin", 3.0)  # meme sector
        assert (sol["tier"], sol["leverage"]) == ("megacap", 20.0)
        assert sol["overrides"] == {"tier": None, "leverage": None}
        risk = sol["risk"]
        assert risk["stop_pct"] == 5.0 and risk["mmr"] == 0.004 and risk["trained_at"]
        assert set(risk["windows"]) == {w.name for w in WINDOWS}
        for name, w in risk["windows"].items():
            assert w["state"] in ("trending", "choppy", "quiet", "wild")
            assert w["momentum_label"] in ("bullish", "bearish", "neutral") and -100 <= w["momentum"] <= 100
            assert w["volatility"] in ("calm", "normal", "high", "wild")
            rec = w["trend_record"]
            assert rec["base"]["n"] > 0 and rec["round_trip_pct"] == pytest.approx(0.177)
            assert set(rec) == {"long", "short", "base", "round_trip_pct"}
            for side in ("long", "short"):
                r = w[side]
                assert r["model"] == "har"  # offline: the baseline only
                assert r["score"] == round(100 * r["stop_hit_chance"])
                assert r["bad_case_margin_pct"] == pytest.approx(20 * r["bad_case_pct"])
                assert 0 <= r["liquidation_chance"] <= r["stop_hit_chance"] <= 1
                assert isinstance(r["checked"], bool) and r["reason"]
                q = r["quantiles"]
                assert len(q["alphas"]) == len(q["moves_pct"]) and q["moves_pct"] == sorted(q["moves_pct"])
        assert sol["funding"] is None  # synthetic candles carry no funding


def test_tier_and_leverage_overrides_round_trip(tmp_path):
    storage_file = tmp_path / "saved_coins.json"
    with TestClient(create_app(initial_symbols=["B-SOL_USDT"], live_mode=False, persistence_path=storage_file)) as client:
        before = client.get("/api/coins").json()[0]["risk"]["windows"]["1h"]["long"]
        out = client.patch("/api/coins/B-SOL_USDT", json={"tier": "altcoin"}).json()
        assert (out["tier"], out["tier_auto"], out["leverage"]) == ("altcoin", "megacap", 10.0)
        after = out["risk"]["windows"]["1h"]["long"]
        assert after["stop_hit_chance"] < before["stop_hit_chance"]  # 10x: the -5% stop is twice as far
        out = client.patch("/api/coins/B-SOL_USDT", json={"leverage": 5}).json()
        assert (out["tier"], out["leverage"], out["overrides"]) == ("altcoin", 5.0, {"tier": "altcoin", "leverage": 5.0})
        assert client.patch("/api/coins/B-SOL_USDT", json={"tier": "moon"}).status_code == 400
        assert client.patch("/api/coins/B-SOL_USDT", json={"leverage": 500}).status_code == 400
        assert client.patch("/api/coins/B-NOPE_USDT", json={"leverage": 5}).status_code == 404

    saved = json.loads(storage_file.read_text())["symbols"]
    assert saved == [{"symbol": "B-SOL_USDT", "tier": "altcoin", "leverage": 5.0}]
    with TestClient(create_app(live_mode=False, persistence_path=storage_file)) as client:
        coin = client.get("/api/coins").json()[0]
        assert (coin["tier"], coin["leverage"]) == ("altcoin", 5.0)
        out = client.patch("/api/coins/B-SOL_USDT", json={"tier": None, "leverage": None}).json()
        assert (out["tier"], out["leverage"]) == ("megacap", 20.0)


def test_old_watchlist_format_still_loads(tmp_path):
    storage_file = tmp_path / "saved_coins.json"
    storage_file.write_text(json.dumps({"symbols": ["B-ETH_USDT", "b-sol_usdt"], "updated_at": "2026-10-01"}))
    with TestClient(create_app(live_mode=False, persistence_path=storage_file)) as client:
        assert [c["symbol"] for c in client.get("/api/coins").json()] == ["B-ETH_USDT", "B-SOL_USDT"]
    assert json.loads(storage_file.read_text())["symbols"][0] == {"symbol": "B-ETH_USDT", "tier": None, "leverage": None}
