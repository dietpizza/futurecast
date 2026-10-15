# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, read-only web dashboard for CoinDCX futures pairs. For each tracked coin it trains LightGBM / XGBoost / CatBoost on real closed candles and serves calibrated long/short win probabilities per horizon, plus a secondary cross-asset "residual alpha" forecast. It must never trade: there are no API keys, no order routing, and `AppConfig.enforce_zero_execution_policy()` (`config.py`) refuses to start if `execution_enabled` is true or `air_gapped` is false. Keep it that way.

## Commands

Python 3.12, managed with `uv`. No build step, linter or formatter is configured.

```bash
uv run pytest -q                                   # full suite (~70 tests, ~15s, no network)
uv run pytest tests/test_web.py -q                 # one file
uv run pytest tests/test_web.py::test_web_index_page -q   # one test
uv run pytest -k persistence -q                    # by name

./run.sh                                           # dashboard on 0.0.0.0:8000 (wraps `uv run futurecast`)
uv run futurecast --host 127.0.0.1 --port 8765     # local only, other port
uv run futurecast --offline                        # synthetic random-walk candles, no network, no training
```

Runtime state lives in `~/.futurecast`, or `$FUTURECAST_HOME` if set (`paths.py`). When running the server to check a change, point `FUTURECAST_HOME` at a temp folder so the user's watchlist and trained models aren't touched. Tests already do this through an autouse fixture in `tests/conftest.py`, which also stubs the Binance catalog so tests stay offline. To check training end to end, build a `SymbolWorker` with `FUTURECAST_HOME` set to a temp folder and call `train_sync()`. It needs network access and takes about 80s for BTC.

## Architecture

**Data flow.** `cli.py` → `publisher/web_server.py` (FastAPI app, REST + `/ws` WebSocket) → `publisher/multi_manager.py`, which is the hub. `MultiCoinStreamManager` holds one `SymbolWorker` per tracked coin and runs the stream loop: live price and order book on every poll (~3s), and candles, features and model outputs only when a candle closes. Every snapshot is pushed to browsers as a `FORECAST_UPDATE` WebSocket message.

**Symbols.** CoinDCX names (`B-SOL_USDT`, `I-BTC_INR`) are the IDs everywhere: API, persistence, model file names (via `sanitize_symbol_for_filename`). All market data (candles, settled funding rates, prices, order books) actually comes from the Binance USD-M perpetual behind the pair (`ingestion/binance.py`: `B-SOL_USDT` → `SOLUSDT`, `B-1000PEPE_USDT` → `1000PEPEUSDT`, INR pairs → their USDT perpetual). CoinDCX is never called; the add-coin catalog (`ingestion/coindcx_catalog.py`) is Binance perpetuals renamed to CoinDCX form plus a curated registry. Each coin gets sector-aware benchmarks (`ingestion/factory.py`: `get_symbol_sector`, `resolve_benchmarks`, e.g. meme coins → `[DOGE, BTC]`) that are aligned with it before feature generation. A coin is never its own benchmark (BTC → `[ETH, SOL]`, ETH → `[BTC, SOL]`). Listing it would duplicate its candle columns, and feature generation fails with `'arg1' columns are not unique`. Feature names containing `btc` (`beta_btc_*`, `funding_rate_spread_btc`, …) refer to the primary benchmark, not literally BTC.

**Three model families per coin:**
- **Win probabilities (primary output):** `models/trade_probability.py`. `TradeProbabilityModel` is trained per candle interval. `HORIZONS_BY_INTERVAL` maps 5m candles → 5m/15m/1h horizons and 15m candles → 4h/8h. A "win" means the volatility-scaled take-profit is touched before the stop-loss (`labels/barriers.py`). The take-profit and stop are not model output. `barrier_distances` sets both to 0.5× the trailing horizon volatility, floored at 2× the 10 bps round-trip cost, so calm coins like BTC sit on the 0.20% floor. The models only estimate the odds of reaching them. The module docstring describes the purged fit → early-stop → calibrate → test split, Platt calibration, the bootstrap "edge" test, the trade replay that gates long/short signals, and out-of-sample shrink. Read it before changing anything there.
- **Trade risk:** `models/trade_risk.py`. `TradeRiskModel` uses the same intervals, horizons and purged slices. Quantile regressors learn the 95th percentile of the adverse excursion (worst move against each side, `labels.barriers.adverse_excursion`), measured in volatility units, with a conformal calibration step. The model is served only if it beats a volatility-only baseline on the test slice; otherwise the baseline is served. It outputs a worst-move %, a 0–100 score ranking it against the coin's recent history, and a max leverage. There is no rule-based risk score: untrained coins show none.
- **Residual alpha (secondary):** `models/ensemble.py` (`MultiModelEnsemble` regressors) driven by `engine/forecaster.py` (`PureForecastEngine`) on 5m candles with a 15-minute horizon. It writes daily JSONL forecast logs through `publisher/broadcaster.py`. The dashboard no longer shows it, but it still trains and logs.

**Training.** `SymbolWorker.train_sync` fetches up to `HISTORY_BARS` real candles per interval and needs at least `MIN_TRAIN_BARS`. It never trains on synthetic data. It runs in a thread executor behind one asyncio lock (one coin at a time), saves models to `<home>/models/` (`trade_prob_<sym>_<interval>.joblib`, `trade_risk_<sym>_<interval>.joblib`, `lightgbm_<sym>_residual_alpha.txt`, …), and hot-swaps them under the worker's lock. A coin counts as trained only when both its win and risk models load. In live mode untrained coins are queued automatically when streaming starts. A model whose `FORMAT_VERSION` doesn't match is ignored, so bumping the version forces a retrain.

**Watchlist.** `<home>/tracked_symbols.json` holds the coins the user added and is rewritten on every add/remove. There are no default coins: with nothing saved the watchlist starts empty. `create_app(initial_symbols=...)` overrides the saved list, which tests use.

## Gotchas

- **Config:** `load_config()` reads `<home>/config/*.yaml`, not the repo's `config/` folder. The repo YAML files only document the defaults; the real defaults are the pydantic models in `config.py`. Editing `config/*.yaml` in the repo changes nothing at runtime.
- **Window units:** feature window settings named `*_min` (e.g. `beta_windows_min: [60, 1440]`) are passed straight to `rolling(n)`, so they count candles, not minutes. On 5m candles, 1440 is 5 days. `LIVE_BARS` must stay above the longest window. The exception is `funding_ema_window_min`: it is a clock-time window in minutes (a 3-day trailing mean), so the live buffer reproduces the training value. `BarrierSpec.vol_window` (240) also counts candles, despite its "1m bars" comment.
- **README:** it still describes 1-minute bars and a DOGE-specific target in places. The code (5m/15m candles, any coin) is the source of truth.

## Frontend

Plain HTML/CSS/JS in `publisher/web/`, with no bundler or framework. `index.html` loads `static/js/*.js` in order as classic scripts that share globals; the markup calls them from inline `onclick` handlers. Static files are served with `Cache-Control: no-cache` (`RevalidatingStaticFiles`), so edits show on reload without restarting the server. Colors are CSS custom properties defined twice in `static/css/main.css`: light in `:root` and dark in `:root[data-theme="dark"]`. Any new color token needs both. The theme is set before first paint by an inline script in `<head>` and toggled by `theme.js`.

**Coin card.** `app.js` re-renders each card with `innerHTML` on every `FORECAST_UPDATE` (~3s), so UI state that must survive a refresh lives in window-level maps (`expandedCards`, `openReliability`).
- **Layout:** the expanded card opens with the trade block (`tradeBlock`) for the selected horizon: verdict, take-profit, stop, worst case and max leverage at full contrast. The horizon table, Evidence and the order book line below it use `.detail-quiet` and stay low-contrast; keep that split when adding things.
- **Verdicts:** `sideVerdict` re-derives each side's verdict from the server fields and mirrors the signal rules in `trade_probability.py`. It and `UNPROVEN_TIP` hard-code the 10-trade replay minimum (`min_signal_trades`), so change them together.
- **Order book:** display-only; no model uses it.

**Checking UI changes.** `--offline` has no trained models, so the trade block never renders there. If the user's dashboard is running on :8000, it serves the working tree's static files, so a reload shows edits. View it only: don't train, add or remove coins there. To preview signal states, first close the socket in the page (`ws.onmessage = null; ws.close()`), then edit `coinsData` and call `renderCoinCard`. Otherwise the next update overwrites your edits within seconds.
