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
        models_dir=tmp_path / "models",
    )
    return app


def test_web_index_page(test_app):
    with TestClient(test_app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "FutureCast" in response.text
        assert "models-table" in response.text
        # Verify tooltips and beginner-friendly descriptions
        assert "global-tooltip" in response.text
        assert "data-tooltip" in response.text
        assert "data-tooltip-title" in response.text
        assert "basis points" in response.text
        assert "Residual Alpha" in response.text
        # Verify centered floating timeframe selector
        assert "timeframe-selector" in response.text
        assert "5m" in response.text
        assert "15m" in response.text
        assert "4h" in response.text
        assert "8h" in response.text
        # Verify CoinDCX link and glossary footer
        assert "coindcx.com/trade/" in response.text
        assert "plain-english-glossary" in response.text
        assert "The \"Not a Finance-Bro\" Cheat Sheet" in response.text

        # Verify CoinDCX Futures Search Sidebar elements
        assert "coindcx-sidebar" in response.text
        assert "sidebar-search-input" in response.text
        assert "sidebar-toggle-btn" in response.text

        # Verify global light/dark theme toggle
        assert 'id="theme-toggle"' in response.text
        assert "/static/js/theme.js" in response.text

        # Verify modular CSS and JS static files are served
        for asset in [
            "/static/css/main.css",
            "/static/css/tooltips.css",
            "/static/css/glossary.css",
            "/static/css/sidebar.css",
            "/static/js/theme.js",
            "/static/js/tooltips.js",
            "/static/js/timeframe.js",
            "/static/js/app.js",
            "/static/js/sidebar.js",
        ]:
            asset_resp = client.get(asset)
            assert asset_resp.status_code == 200, f"Static asset {asset} failed to load"
            # Browsers must revalidate so edited CSS/JS is never served stale
            assert asset_resp.headers.get("cache-control") == "no-cache", f"{asset} missing no-cache"


def test_api_coindcx_futures_catalog(test_app):
    """Test CoinDCX futures catalog search endpoint."""
    with TestClient(test_app) as client:
        # Full catalog
        response = client.get("/api/coindcx/futures")
        assert response.status_code == 200
        catalog = response.json()
        assert len(catalog) >= 30

        symbols = [item["symbol"] for item in catalog]
        assert "B-DOGE_USDT" in symbols
        assert "B-SOL_USDT" in symbols
        assert "B-PEPE_USDT" in symbols
        assert "B-BTC_USDT" in symbols
        assert "B-ZK_USDT" in symbols

        # Search query filter (e.g. pepe)
        q_resp = client.get("/api/coindcx/futures?q=pepe")
        assert q_resp.status_code == 200
        pepe_results = q_resp.json()
        assert any("PEPE" in item["symbol"] for item in pepe_results)

        # Search query for B-ZK_USDT and zkSync
        zk_resp = client.get("/api/coindcx/futures?q=zk")
        assert zk_resp.status_code == 200
        zk_results = zk_resp.json()
        assert any(item["symbol"] == "B-ZK_USDT" for item in zk_results)
        zk_item = next(item for item in zk_results if item["symbol"] == "B-ZK_USDT")
        assert zk_item["name"] == "zkSync"
        assert zk_item["coindcx_url"] == "https://coindcx.com/futures/B-ZK_USDT"

        # Search query using direct CoinDCX URL paste
        url_resp = client.get("/api/coindcx/futures?q=https://coindcx.com/futures/B-ZK_USDT")
        assert url_resp.status_code == 200
        assert any(item["symbol"] == "B-ZK_USDT" for item in url_resp.json())

        # Sector filter
        sec_resp = client.get("/api/coindcx/futures?sector=MEME")
        assert sec_resp.status_code == 200
        meme_results = sec_resp.json()
        assert len(meme_results) >= 5
        assert all(item["sector"] == "MEME" for item in meme_results)
        assert all(item["is_shitcoin"] is True for item in meme_results)


def test_api_status_air_gapped_guarantee(test_app):
    with TestClient(test_app) as client:
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["execution_disabled"] is True
        assert data["air_gapped"] is True
        assert len(data["active_coins"]) >= 2


def test_api_coins_and_multi_model_predictions(test_app):
    with TestClient(test_app) as client:
        response = client.get("/api/coins")
        assert response.status_code == 200
        coins = response.json()
        assert len(coins) >= 2

        doge_coin = next((c for c in coins if "DOGE" in c["symbol"]), None)
        assert doge_coin is not None
        assert doge_coin["sector"] == "MEME"
        assert doge_coin["is_shitcoin"] is True
        assert "benchmarks" in doge_coin

        sol_coin = next((c for c in coins if "SOL" in c["symbol"]), None)
        assert sol_coin is not None
        assert sol_coin["is_shitcoin"] is False

        for coin in coins:
            # Offline mode renders synthetic candles and says so
            assert coin["has_data"] is True
            assert coin["data_source"] == "synthetic (offline mode)"
            assert "trade_risk" in coin
            assert "orderbook_depth" in coin
            # Untrained coins never get invented models, metrics or probabilities
            assert coin["is_custom_trained"] is False
            assert coin["training_status"] == "untrained"
            assert coin["model_predictions"] == {}
            assert coin["cross_verification"]["models_count"] == 0
            assert coin["trade_probabilities"] == {}
            assert "timeframe_forecasts" not in coin


def test_trained_coin_serves_calibrated_trade_probabilities(tmp_path):
    from futurecast.features.pipeline import FeaturePipeline
    from futurecast.ingestion.factory import resolve_benchmarks, sanitize_symbol_for_filename
    from futurecast.ingestion.storage import align_cross_asset_data
    from futurecast.models.trade_probability import TradeModelConfig, TradeProbabilityModel, trade_model_path
    from futurecast.publisher.multi_manager import _synthetic_candles
    import numpy as np

    sym = "B-DOGE_USDT"
    benches = resolve_benchmarks(sym, "coindcx", sector_aware=True)
    rng = np.random.default_rng(0)
    target = _synthetic_candles(sym, 5, 4000, rng)
    aligned = align_cross_asset_data(target, {b: _synthetic_candles(b, 5, 4000, rng) for b in benches}, freq="5min")
    feats = FeaturePipeline(target_symbol=sym, primary_benchmark=benches[0], secondary_benchmark=benches[-1]).generate_features(aligned)
    model = TradeProbabilityModel(TradeModelConfig(interval="5m", horizons={"15m": 3}, max_rounds=40))
    model.fit(feats, aligned[f"{sym}_close"], aligned[f"{sym}_high"], aligned[f"{sym}_low"])
    model.save(trade_model_path(tmp_path / "models", f"{sanitize_symbol_for_filename(sym)}_5m"))

    app = create_app(initial_symbols=[sym], live_mode=False,
                     persistence_path=tmp_path / "coins.json", models_dir=tmp_path / "models")
    with TestClient(app) as client:
        coin = client.get("/api/coins").json()[0]
        probs = coin["trade_probabilities"]
        assert set(probs) == {"15m"}  # only horizons with a trained model, nothing scaled or invented
        h = probs["15m"]
        assert h["candle_interval"] == "5m" and h["horizon_bars"] == 3
        assert h["signal"] in ("LONG", "SHORT", "NEUTRAL")
        assert h["tp_pct"] > 0 and h["sl_pct"] > 0
        for side in ("long", "short"):
            assert 0.0 <= h[side]["probability"] <= 1.0
            assert 0.0 < h[side]["break_even"] < 1.0
            assert h[side]["status"] in ("edge", "no_edge", "insufficient_data")
            if h[side]["status"] != "edge":
                assert h[side]["probability"] == h[side]["base_rate"]


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
        assert added["benchmarks"] == ["B-DOGE_USDT", "B-BTC_USDT"]

        # Add NFT token B-BLUR_USDT: untrained, so no borrowed or invented models
        blur_resp = client.post("/api/coins", json={"symbol": "B-BLUR_USDT"})
        assert blur_resp.status_code == 200
        blur_added = blur_resp.json()
        assert blur_added["symbol"] == "B-BLUR_USDT"
        assert blur_added["model_predictions"] == {}
        assert blur_added["trade_probabilities"] == {}

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
            assert payload["execution_disabled"] is True
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
