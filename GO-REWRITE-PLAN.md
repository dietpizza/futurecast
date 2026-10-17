# Plan: rewrite DeltaRisk's backend in Go, without LightGBM

Written 2026-10-18 at the user's request. Phase 0 done 2026-10-18 (fixtures: ETH/BTC, 5m 20 days, 15m 40, 1h 120
ending 2026-10-17 20:00 UTC, 10.5 MB; `internal/testdata/parity/inputs.json` has the spans). Read `CLAUDE.md` first; this file only says what
changes and in what order.

## Goal

Replace the Python package (`src/deltarisk`, ~4,500 lines, plus ~2,000 lines of tests) with a Go program
that does the same things: the dashboard server, `deltarisk risk` and `deltarisk backtest`. The Svelte page
in `web/` stays. It only talks to `/api` and `/ws`, so it keeps working as long as the JSON stays the same.
Then rewire the `mise` tasks and delete the Python code.

**LightGBM is dropped.** Every window and side uses the HAR baseline. On 2026-10-18 we compared the two on the
walk-forward test windows of BTC, ETH, SOL, LINK, DASH and ZK at −5% and −10% stops. LightGBM shipped in 9 of
96 window-sides, all on BTC/ETH/SOL. It moved scores by about 2 points on average and improved stop-hit
accuracy (Brier score) by 0–4%. Where the two disagreed about a reading being under 50, the stop was hit
about half the time. No other gradient-boosting library replaces it.

## Decisions (defaults; change them before starting)

| Topic | Default |
|---|---|
| Go version | 1.25 (installed through mise; add `go = "1.25"` to `mise.toml`) |
| Module path | `deltarisk` (local module; the git remote keeps its old name) |
| Command line | `github.com/spf13/cobra`: subcommands, `--help`, and `--breakeven/--no-breakeven`-style flags like typer's |
| HTTP | standard `net/http` with Go 1.22+ method/path patterns (`GET /api/coins`, `PATCH /api/coins/{symbol}`) |
| WebSocket | `github.com/coder/websocket` |
| YAML | `gopkg.in/yaml.v3`; same `<home>/config/assets.yaml`, unknown keys ignored |
| Parquet caches | `github.com/apache/arrow-go/v18` (`parquet/pqarrow`). The existing files are flat (`time` int64 + float64/int64 columns) with `covered_from`/`covered_to`/`fetched_at` in the file metadata, so the Go code reads and writes the same files and the user's cache in `~/.deltarisk/cache` keeps working |
| Linear algebra | `gonum.org/v1/gonum/mat` for the HAR least-squares fit (9 columns, use QR, not normal equations) |
| Data frames | none. A chart is a struct of equal-length slices (`OpenTime []int64` in ms, then `Close`, `High`… per symbol). Rolling windows, shifts, `merge_asof` and percentiles are small hand-written helpers in one package |
| Model cache | dropped (`<home>/cache/risk/*.pkl`, `MODEL_CACHE_VERSION`). HAR training is least squares and sorting, well under a second per coin, so restarts just retrain. Daily retraining stays, so readings behave as today |
| Frontend files | served from disk as today (`web/dist`, override with `--web-dist`). Embedding them in the binary is possible later (needs a Go file inside `web/` because `go:embed` can't reach parent folders) |
| Logging | `log/slog` |

## What doesn't change

- Every API route and its JSON (see "API contract"), the WebSocket `FORECAST_UPDATE` message, and the home
  folder layout: `DELTARISK_HOME`, legacy `FUTURECAST_HOME`, the one-time `~/.futurecast` move,
  `tracked_symbols.json` (both formats), the candle and funding caches.
- Binance only, public endpoints, no keys. Same retry rules (429/418 `Retry-After`), the per-minute weight
  budget (80% of 2,400), the funding-rate budget, paging, 3 market-board fills in flight.
- The numbers: windows, tiers, stops, HAR inputs, walk-forward folds, coverage band, regime rules, trend
  record, funding view, the setup and wedge rules, fees, backtest accounting. Checked against Python
  outputs (see "Parity").
- Offline mode: random-walk candles, baseline only. Go's random numbers differ from numpy's, so offline values
  differ. Nothing should depend on the exact values.

## What changes for the user

- `model` is always the baseline, so the field goes away. Page edits: drop `model` from `SideRisk` in
  `web/src/lib/types.ts`, the "Model: …" part of the tooltip in `web/src/lib/risk.ts`, and the LightGBM
  sentence in `Glossary.svelte`.
- `deltarisk risk` loses the `LGBM`, `gain` and `model` columns.
- Commands become `go run ./cmd/deltarisk …` or the built `bin/deltarisk …` instead of `uv run deltarisk …`.

## Layout

| Python | Go package |
|---|---|
| `cli.py`, `__init__.py` | `cmd/deltarisk` (main, `risk`, `backtest`, dashboard as the default command, `web` alias) |
| `paths.py`, `config.py` | `internal/paths`, `internal/config` |
| `ingestion/binance.py` | `internal/binance` (client, rate limits, klines, funding, tickers, `fetch_recent_closes`) |
| `ingestion/cache.py` | `internal/cache` (candle and funding Parquet caches, a mutex per file) |
| `ingestion/storage.py` | `internal/candles` (the chart type, `Align`: common grid, forward-fill ≤ 3 candles, drop longer gaps) |
| `ingestion/coindcx_catalog.py`, `ingestion/factory.py` | `internal/catalog` (catalog, sectors, benchmarks, symbol mapping) |
| (pandas/numpy helpers) | `internal/series` (rolling mean/sum/std/min/max/rank, EWM, shift, diff, quantile, as-of join) |
| `stats/regime.py` | `internal/regime` |
| `stats/risk.py` | `internal/risk` (inputs, targets, HAR, walk-forward, `CoinRisk`, view, tiers, `exceed_chance`) |
| `stats/signals.py` | `internal/funding` |
| `features/structure.py`, `features/setups.py`, `labels/setups.py` | `internal/setups` |
| `features/wedges.py`, `labels/wedges.py` | `internal/wedges` |
| `backtest.py` | `internal/backtest` |
| `publisher/multi_manager.py` | `internal/manager` (workers, download and training queues, stream loop) |
| `publisher/market_board.py` | `internal/board` |
| `publisher/web_server.py` | `internal/server` (routes, static files, `/ws` hub) |

## pandas behaviour to copy exactly

These are where a port drifts silently. Each gets a helper in `internal/series` with its own test:

- `rolling(n)` with the default `min_periods = n`: NaN until n values, and NaN when any value in the
  window is NaN. With an explicit `min_periods` (regime `sigma` and `vol_pct`, funding percentile) it counts
  non-NaN values.
- `rolling().std()` uses ddof = 1.
- `rolling().rank(pct=True)`: the latest value's rank in its window, ties averaged, divided by the number of
  non-NaN values.
- `ewm(span=L, adjust=False)`: α = 2 / (L + 1), seeded with the first value.
- `np.quantile` default (linear interpolation) for the baseline's quantiles and `PAGE_ALPHAS`.
- `np.searchsorted(side="left")` in the baseline's stop-hit chance.
- `merge_asof(direction="backward")` on close times for funding inputs.
- `.where(cond, other)` / `.clip` NaN handling: a NaN condition keeps NaN, `clip` leaves NaN alone.
- Times: everything in int64 milliseconds (open times), close = open + interval. The pandas 3 µs/ns issue
  goes away.
- JSON: NaN and ±Inf become `null`, as the Python snapshot code does today.

## API contract

Routes (all called by the page): `GET /` (index.html, `Cache-Control: no-cache`), `GET /favicon.svg`,
`GET /assets/*` (immutable cache), `GET /api/coins`, `POST /api/coins`, `POST /api/coins/{symbol}/reload`,
`PATCH /api/coins/{symbol}` (tier/leverage, null resets, missing fields stay), `DELETE /api/coins/{symbol}`,
`GET /api/coindcx/futures`, `GET /api/market` (the first request waits up to 5 s for the first poll),
`WS /ws`. Without a build, `/` answers 503 with how to make one.

The coin snapshot is `Coin` in `web/src/lib/types.ts` (minus `model`). The Go structs get JSON tags that
match it field for field, and `web/src/lib/types.ts`'s header comment moves to point at the Go type. Status codes and error
bodies follow `tests/test_web.py`.

## Parity: capture Python's answers before deleting it

Phase 0 writes a throwaway script (`scripts/parity_dump.py`, deleted in phase 9) that runs the Python code on
fixed inputs and saves its outputs under `internal/testdata/parity/`, small enough to commit (a few MB):

- One coin and its benchmark (ETH with BTC) from the user's candle cache, cut to the most recent ~30 days per chart,
  plus its funding events. Store them as Parquet in the cache format, so the same reader loads them.
- Python's outputs on them, as JSON or Parquet: the aligned charts; `regime_frame` per window; `risk_inputs`,
  `adverse_moves`, `future_vol` per window; `walk_forward(..., lgbm=False)` per window; `CoinRisk.view` at
  three stops and two leverages; `trend_record`; `funding_view`; `exceed_chance` on a grid; the setup and
  wedge trade lists and backtest reports for one config each.

Go tests load these and compare: labels exactly, numbers to a relative 1e-9 (1e-6 where a least-squares fit
or long sums are involved). A mismatch means a semantics bug, not a reason to loosen the tolerance.

## Phases

Each phase ends with `go test ./...` green and a commit. Python stays runnable until phase 9.

0. **Prepare.** Commit or drop pending work (the unused-code cleanup from 2026-10-18 is uncommitted). Write
   and run `scripts/parity_dump.py`; commit the fixtures. Add `go = "1.25"` to `mise.toml`, `go mod init deltarisk`.
1. **Basics.** `internal/paths` (home, legacy env and folder move, `Resolve`), `internal/config`
   (defaults as structs, YAML overrides). Port `test_paths.py`, `test_config.py`.
2. **Market data.** `internal/binance`, `internal/cache`, `internal/candles` (`Align`), `internal/catalog`.
   The test helper is an `httptest.Server` faking Binance, replacing `conftest.py`'s catalog stub and the
   fake transports in `test_binance.py`/`test_funding.py`. Port `test_binance.py`, `test_catalog.py`,
   `test_coindcx.py`, `test_ingestion_storage.py`, and the cache parts of `test_funding.py`. Check reading
   and appending to a copy of the user's real cache files.
3. **Statistics.** `internal/series`, `internal/regime`, `internal/funding`, `internal/risk` (baseline only,
   folds run in parallel with goroutines). The `deltarisk risk` command. Port `test_regime.py`,
   `test_signals.py`, `test_risk.py` (minus the LightGBM tests; keep `test_inputs_never_use_later_candles`
   and the `exceed_chance` checks), plus the parity tests.
4. **Backtest.** `internal/setups`, `internal/wedges`, `internal/backtest`, the `deltarisk backtest` command
   with every flag and `--csv`. Port `test_structure.py`, `test_setups.py`, `test_wedges.py`,
   `test_backtest.py`, plus the parity trade lists.
5. **Dashboard.** `internal/manager`: one worker per coin with the same status flow (pending → queued →
   downloading → training → ready/failed) and progress, a download semaphore and a training semaphore (one
   coin each, so the next downloads while the previous trains), 3 s price polls, rescoring on 5m/15m/1h
   closes, daily retraining while the old models serve, watchlist persistence and overrides.
   `internal/board`, `internal/server`, the `/ws` hub. Port `test_web.py`, `test_queues.py`, the worker
   parts of `test_funding.py`, `test_market_board.py`.
6. **Page.** Drop `model` and the LightGBM text (see "What changes for the user"); `mise run check`.
7. **Tasks and docs.** Rewire `mise.toml` (below). Update `.gitignore` (`bin/`; drop the Python and model
   entries). Rewrite `CLAUDE.md` (commands, architecture, gotchas for Go, LightGBM history kept as one line)
   and `README.md` (`uv run …` → `go run ./cmd/deltarisk …`, `./run.sh` → `mise start`). Add a line at the top of
   `docs/risk-plan.md` saying LightGBM was removed on this date.
8. **Side-by-side check.** Copy the user's candle cache into a temp `DELTARISK_HOME` and run the Python and
   Go dashboards against it on spare ports with the same coins (at least BTC, ETH, SOL and one shitcoin).
   Compare `/api/coins`: every reading within the parity tolerance, the same keys and nulls. Run
   `deltarisk risk` for both on 3 coins and compare the HAR columns. Open the page on the Go server and
   click through: add/remove pair, tier/leverage override, reload, window and stop changes, market board.
9. **Delete Python.** Remove `src/`, `tests/`, `pyproject.toml`, `uv.lock`, `.python-version`, `.venv/`,
   `.pytest_cache/`, `.ruff_cache/`, `scripts/parity_dump.py`. Keep `config/assets.yaml` (it still documents
   the defaults) and the parity fixtures (they become plain regression tests). Run `go test ./...`,
   `go vet ./...`, `mise run check`, `mise run build`, then start the dashboard once more.

## mise tasks after the rewrite

| Task | Runs |
|---|---|
| `install` | `go mod download`, then `install:web` |
| `install:web` | unchanged |
| `dev` | `dev:api` + `dev:web` |
| `dev:api` | `go run ./cmd/deltarisk --host 127.0.0.1 --port $DELTARISK_API_PORT` (arguments pass through, e.g. `--offline`). `air` is installed if live reload is wanted later |
| `dev:web` | unchanged |
| `build` | `build:web` (today's `npm run build`) and `build:go` (`go build -o bin/deltarisk ./cmd/deltarisk`) |
| `start` | `build`, then `bin/deltarisk` on 0.0.0.0:8000 (`PORT` overrides), replacing `run.sh` |
| `check` | `npm run check` and `go vet ./...` |
| `test` | `go test ./...` |

## Risks

- **Rolling-window semantics** are the most likely source of drift. The parity fixtures are there to catch it.
- **Cache compatibility.** If arrow-go can't append to a pyarrow-written file, the fallback is to rewrite
  the file in Go's format on first load. Never delete the user's cache.
- **Concurrency.** The Python manager relies on asyncio's single thread for some state. In Go, a worker's
  candles, models and snapshot need a mutex or a single owning goroutine. Run `go test -race`.
- **Size.** Expect 6–9k lines of Go including tests. Phases 3 and 5 are the largest.
