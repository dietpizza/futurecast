"""Tests for Web UI, REST endpoints, and WebSocket broadcasting."""

import json
import pytest
from starlette.testclient import TestClient
from futurecast.publisher.web_server import create_app


@pytest.fixture
def test_app():
    app = create_app(
        initial_symbols=["B-DOGE_USDT", "B-SOL_USDT"],
        live_mode=False,
    )
    return app


def test_web_index_page(test_app):
    with TestClient(test_app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "futurecast" in response.text
        assert "models-table" in response.text
        assert "AIR-GAPPED" in response.text


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
        assert "benchmarks" in doge_coin

        # Verify multi-model predictions exist
        assert "model_predictions" in doge_coin
        assert "cross_verification" in doge_coin
        assert doge_coin["execution_disabled"] is True


def test_api_add_and_delete_coin(test_app):
    with TestClient(test_app) as client:
        # Add new meme token (should route sector benchmark to DOGE)
        add_resp = client.post("/api/coins", json={"symbol": "B-PEPE_USDT"})
        assert add_resp.status_code == 200
        added = add_resp.json()
        assert added["symbol"] == "B-PEPE_USDT"
        assert added["sector"] == "MEME"
        assert added["benchmarks"] == ["B-DOGE_USDT", "B-BTC_USDT"]

        # Check it is in active list
        get_resp = client.get("/api/coins")
        symbols = [c["symbol"] for c in get_resp.json()]
        assert "B-PEPE_USDT" in symbols

        # Delete token
        del_resp = client.delete("/api/coins/B-PEPE_USDT")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "removed"

        # Verify deletion
        get_resp2 = client.get("/api/coins")
        symbols2 = [c["symbol"] for c in get_resp2.json()]
        assert "B-PEPE_USDT" not in symbols2


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
