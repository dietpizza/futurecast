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
    from futurecast.publisher.multi_manager import SETUP_SPEC

    with TestClient(test_app) as client:
        response = client.get("/api/coins")
        assert response.status_code == 200
        coins = response.json()
        assert len(coins) >= 2

        doge_coin = next((c for c in coins if "DOGE" in c["symbol"]), None)
        assert doge_coin is not None
        assert doge_coin["sector"] == "MEME"
        assert "benchmarks" in doge_coin

        assert any("SOL" in c["symbol"] for c in coins)

        for coin in coins:
            # Offline mode renders synthetic candles and says so
            assert coin["has_data"] is True
            assert coin["data_source"] == "synthetic (offline mode)"
            assert coin["trade_risk"] == {}  # risk comes only from trained models
            assert "orderbook_depth" in coin
            # Untrained coins never get invented models, metrics or probabilities
            assert coin["is_custom_trained"] is False
            assert coin["training_status"] == "untrained"
            assert coin["model_predictions"] == {}
            assert coin["cross_verification"]["models_count"] == 0
            # Structure and setups come from rules, so they show before training; win chances do not
            assert coin["setup_model"] is None
            assert coin["structure"]["htf"]["trend"] in ("up", "down", "none")
            assert coin["structure"]["ltf"]["interval"] == SETUP_SPEC.entry_interval
            assert coin["setups"]["signal"] == "NEUTRAL"
            for side in ("long", "short"):
                if coin["setups"][side]:
                    assert coin["setups"][side]["probability"] is None
            assert "timeframe_forecasts" not in coin


def test_trained_coin_serves_setups_with_win_chances_and_risk(tmp_path):
    from futurecast.backtest import BacktestConfig, dashboard_view, run_backtest
    from futurecast.engine.setups import setup_dataset
    from futurecast.features.pipeline import FeaturePipeline
    from futurecast.ingestion.factory import resolve_benchmarks, sanitize_symbol_for_filename
    from futurecast.ingestion.storage import align_cross_asset_data
    from futurecast.models.setup_model import SetupModel, SetupModelConfig, setup_model_path
    from futurecast.models.trade_risk import TradeRiskConfig, TradeRiskModel, risk_model_path
    from futurecast.publisher.multi_manager import SETUP_SPEC, _synthetic_candles
    import numpy as np

    sym = "B-DOGE_USDT"
    benches = resolve_benchmarks(sym, "coindcx", sector_aware=True)
    rng = np.random.default_rng(0)
    aligned = {
        minutes: align_cross_asset_data(_synthetic_candles(sym, minutes, bars, rng),
                                        {b: _synthetic_candles(b, minutes, bars, rng) for b in benches},
                                        freq=f"{minutes}min")
        for minutes, bars in ((1, 6000), (SETUP_SPEC.ltf_minutes, 6000), (SETUP_SPEC.htf_minutes, 600))
    }
    pipe = FeaturePipeline(target_symbol=sym, primary_benchmark=benches[0], secondary_benchmark=benches[-1])
    feats = pipe.generate_features(aligned[1])
    entry, context = aligned[SETUP_SPEC.ltf_minutes], aligned[SETUP_SPEC.htf_minutes]
    model = SetupModel(SetupModelConfig(spec=SETUP_SPEC, max_rounds=40))
    setups, outcomes, X = setup_dataset(entry, context, pipe.generate_features(entry), sym, SETUP_SPEC)
    model.fit(X, setups, outcomes)
    model.backtests = {"3": dashboard_view(run_backtest(setups, outcomes, entry.index, SETUP_SPEC, BacktestConfig()))}
    clean = sanitize_symbol_for_filename(sym)
    model.save(setup_model_path(tmp_path / "models", clean))
    risk = TradeRiskModel(TradeRiskConfig(interval="1m", horizons={"15m": 15}, max_rounds=40))
    risk.fit(feats, aligned[1][f"{sym}_close"], aligned[1][f"{sym}_high"], aligned[1][f"{sym}_low"])
    risk.save(risk_model_path(tmp_path / "models", f"{clean}_1m"))

    app = create_app(initial_symbols=[sym], live_mode=False,
                     persistence_path=tmp_path / "coins.json", models_dir=tmp_path / "models")
    with TestClient(app) as client:
        coin = client.get("/api/coins").json()[0]
        evidence = coin["setup_model"]
        assert (evidence["entry_interval"], evidence["context_interval"]) == (SETUP_SPEC.entry_interval,
                                                                            SETUP_SPEC.context_interval)
        assert set(evidence["targets"]) == {"2", "3", "4"}
        bt = evidence["backtests"]["3"]
        assert "trades" not in bt and bt["summary"]["grades"] == ["A+"] and bt["equity"][0]["equity"] == 100.0
        assert bt["summary"]["costs"] == pytest.approx(SETUP_SPEC.costs())
        for t in evidence["targets"].values():
            assert t["status"] in ("edge", "no_edge", "insufficient_data")
            assert set(t["base_rates"]) == {"A+", "B"}
            assert "trades" in t["rules_replay"]

        setups = coin["setups"]
        assert setups["signal"] in ("LONG", "SHORT", "NEUTRAL")
        assert (setups["entry_interval"], setups["context_interval"]) == (SETUP_SPEC.entry_interval,
                                                                        SETUP_SPEC.context_interval)
        # Per-leg fees for the page's notes: a stop (taker + slippage out) costs more than a win (maker out)
        assert setups["costs"] == pytest.approx(SETUP_SPEC.costs())
        assert setups["costs"]["lost_pct"] > setups["costs"]["timeout_pct"] > setups["costs"]["won_pct"]
        assert coin["structure"]["htf"]["interval"] == SETUP_SPEC.context_interval
        shown = [setups[side] for side in ("long", "short") if setups[side]]
        assert shown, "synthetic candles should hold at least one setup"
        for d in shown:
            assert d["grade"] in ("A+", "B") and d["state"] in ("waiting", "in_trade", "won", "lost", "timeout",
                                                                "expired", "cancelled")
            up = d["side"] == "long"
            assert (d["stop"] < d["entry"] < d["take_profit"]) if up else (d["stop"] > d["entry"] > d["take_profit"])
            assert [t["target_r"] for t in d["targets"]] == [2.0, 3.0, 4.0]
            assert 0.0 <= d["probability"] <= 1.0 and 0.0 < d["break_even"] < 1.0
            if d["status"] != "edge":  # no proven edge: the rules' own win rate for the grade
                assert d["probability_source"] == "rules" and d["probability"] == d["rules_probability"]
            assert d["max_leverage"] in (1, 2, 3, 5, 10, 20) and d["size_for_1pct_risk"] > 0

        risk = coin["trade_risk"]
        assert set(risk) == {"15m"}
        r = risk["15m"]
        assert r["bias"] in ("LONG_RISKIER", "SHORT_RISKIER", "BALANCED")
        assert r["regime"] in ("calm", "normal", "stormy")
        assert sum(r["regime_probs"].values()) == pytest.approx(1.0, abs=1e-3)
        assert 0.0 <= r["storm_chance"] <= 1.0
        for side in ("long", "short"):
            assert r[side]["worst_move_pct"] > 0
            assert 1 <= r[side]["score"] <= 99
            assert r[side]["level"] in ("LOW", "MODERATE", "HIGH", "EXTREME")
            assert r[side]["max_leverage"] in (1, 2, 3, 5, 10, 20)
            assert r[side]["status"] in ("edge", "no_edge", "insufficient_data")


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
        assert blur_added["setup_model"] is None

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
