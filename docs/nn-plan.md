# Plan: a small coin-agnostic neural network for long/short risk

Written 2026-10-19 with the user (decisions below are theirs). Not started. Read `CLAUDE.md` first, especially
"Risk models": this plan adds a second model next to the HAR baseline and changes nothing that works today.

## Goal

Train one small neural network, offline in Python, on many coins at once, that predicts the same thing the HAR
baseline predicts: the quantiles of the adverse move over the next window, in swing units (the HAR volatility
forecast). Ship its weights inside the Go binary and run it on the CPU in pure Go. Per coin, window and side, the
network serves a reading only where it beats HAR on data it never saw; everywhere else HAR keeps serving. Done
right, the dashboard can never get worse than it is today.

Be honest about the bar: LightGBM on per-coin data beat HAR in 9 of 96 window-sides, by at most 3.7%. A pooled
network sees far more data (30 coins instead of one) and can learn things one coin can't (how meme coins' tails
differ from majors' at the same volatility, what crowded funding does to the next hour), so it may do better, but
"it rarely passes the gate" is an acceptable outcome. The gate is what makes it safe to try.

## Decisions

| Topic | Decision |
|---|---|
| Output | Quantiles of the adverse move in swing units, long and short, at fixed levels (below). Everything downstream keeps working: bad case, `ExceedChance` for any stop, the page's quantile re-read, the coverage check |
| Relation to HAR | Ship where better: per coin, window and side, the network serves only if it beats HAR out-of-sample (pinball ≥ 2% lower) and its bad case passes the 7–13% coverage check, on that coin's data after the training cutoff |
| Features | Go computes them. A new `deltarisk dataset` command writes the network's final inputs and targets to Parquet; Python only trains. Inference uses the same Go function, so training and serving can't drift |
| Architecture | A small MLP: ~40 inputs → 32 → 32 → 2 × 9 outputs, ReLU, monotone quantile head. About 3,000 weights |
| Windows | One network for all eight windows: the window is an input (one-hot plus its length) |
| Coins | A fixed list of ~30 liquid Binance perpetuals across sectors and tiers for training, plus ~5 held out entirely |
| Gate | Per coin at runtime, on the coin's windows after the cutoff stored in the model file |
| Training stack | PyTorch (CPU) in a uv-managed package `nn/`; export to `spec.json` + little-endian float32 `weights.bin`, read in Go with the standard library (no cgo, no ONNX) |
| Weights | Committed in `engine/nn/model/` and always embedded with `go:embed` (no build tag: a few tens of KB, and a retrain is a reviewable diff) |
| Page | A visible "NN" mark next to readings the network gave, and a tooltip line naming the model and its cutoff |
| Retraining | Manual: `mise run nn` (dataset, train, export), then commit. The risk command and the startup log say how old the cutoff is |

## Layout

```
nn/                          Python package (uv, Python 3.12): training only, never imported by Go
  pyproject.toml, uv.lock
  coins.toml                 training coins, held-out coins, cutoff offset, quantile levels, seeds
  deltarisk_nn/
    data.py                  read the Parquet dataset, splits, standardisation, window-balanced batches
    model.py                 the MLP and the monotone quantile head
    train.py                 training loop, early stopping on validation pinball loss
    evaluate.py              pinball loss, coverage, calibration vs the HAR baseline, per window and side
    export.py                spec.json, weights.bin, testvectors.json
    cli.py                   `uv run deltarisk-nn train|evaluate|export`
  tests/                     pytest: head monotonicity, pinball, export round trip
engine/nn/                   Go: features, model loading, inference (imports nothing from engine/risk)
  features.go                the network's inputs from a window's data (shared by dataset export and inference)
  model.go                   parse spec.json + weights.bin, validate, Forward
  embed.go                   //go:embed model/spec.json model/weights.bin
  model/                     the committed model: spec.json, weights.bin, testvectors.json
engine/risk/                 gains the gate and the "nn" choice per side (uses engine/nn)
cmd/deltarisk/dataset.go     the `dataset` command
```

`engine/nn` must not import `engine/risk` (risk imports nn). Feature code therefore takes plain columns: give it a
small struct (or the `risk.Frame` columns by name through an interface) rather than `*risk.WindowData`.

## The network's inputs (engine/nn/features.go)

Coin-agnostic means no input may carry a coin's price level or its absolute volatility level; everything is
relative to the coin's own swing or bounded. Built from `risk.Inputs` columns, the HAR swing and the window, at a
candle's close (nothing from later candles: add a test like `TestInputsNeverUseLaterCandles` for these):

| Input | From | Why |
|---|---|---|
| `rv_k − rv_96`, `pk_k − pk_96` for k = 1, 4, 24 | HAR columns | volatility term structure: is it rising or falling |
| `pk_96 − rv_96` | HAR columns | range vs close-to-close volatility (wickiness) |
| `log(swing / typical_swing)` | HAR forecast, `Model.TypicalSwing` | how unusual today's forecast is for this coin |
| `momentum / 100`, `efficiency`, `vol_pct / 100` | regime | trend and state |
| `move` (clipped to ±5) | inputs | last window's move in typical moves |
| `taker_1 − 0.5`, `taker_4 − 0.5` | inputs | order flow (0 when missing) |
| `range_pos − 0.5` | inputs | position in the 24-window range |
| `sin`, `cos` of `2π·hour/24`; `to_funding / 8` | inputs | time of day, time to funding |
| `funding × 1e4` (clipped to ±10), `funding_pct / 100 − 0.5` | inputs | crowding (0 and a "missing" flag when the coin has no funding column) |
| `market_rv − rv_24` | inputs | benchmark volatility relative to the coin's |
| `has_funding` | | 1/0 |
| window one-hot (8), `log(window minutes) / log(480)` | window | one net for every window |
| tier one-hot (3) | `risk.CoinTier` | megacap / altcoin / shitcoin behave differently |

NaN handling: the dataset keeps only rows where every input is finite after these rules (missing taker or funding
becomes 0 plus a flag); inference returns "no reading" (HAR serves) when an input is not finite.

Standardisation (mean and standard deviation per input over the training rows) is computed in Python and stored in
`spec.json`; Go applies it. The one-hot and flag inputs are left as they are (mean 0, std 1 in the spec).

## Targets and the HAR baseline in the dataset

Per coin and window, the dataset command fits the HAR regression on that coin's rows **before the cutoff**
(`risk.FitHar` on pre-cutoff trainable rows), and computes for every trainable row:

- `swing`: the HAR forecast (price fraction over the window), and `typical_swing` (median over the fit rows);
- `y_long`, `y_short`: the adverse moves divided by `swing` (the target);
- `har_q{level}` long and short: the HAR baseline's quantiles in swing units (the empirical quantiles of the
  pre-cutoff training rows), so Python can report the gain over HAR on exactly the same rows.

At runtime the dashboard's HAR model is fitted on all of a coin's history (as today), so the swing unit the network
sees at serving time differs slightly from the training one. Both are HAR forecasts of the same quantity; the
evaluation in phase 5 measures what that costs.

## The dataset command (cmd/deltarisk/dataset.go)

```bash
go run ./cmd/deltarisk dataset --coins nn/coins.toml --out nn/data --cutoff 2026-09-04
```

- Reads the coin lists from `coins.toml` (training and held-out), downloads history through `history.Client`
  (the disk cache makes reruns fast) with longer spans than the dashboard: 5m 180 days, 15m 1 year, 1h 2 years
  (all within `history.CacheBars`). Each coin with its usual benchmark (`catalog.Benchmark`), aligned as today.
- Per window, keeps every `stride`-th trainable row, `stride = ceil(h / 2)` (overlapping windows add little):
  about 65,000 rows a coin, ~2.3 million in all.
- Writes one Parquet file per coin (`engine/pq`): `time` (close, ms), `window` (index 0–7), `coin` (index into
  `coins.json`), `split` (0 train, 1 validation, 2 post-cutoff test), every input by name, `swing`,
  `typical_swing`, `y_long`, `y_short`, `har_long_q*`, `har_short_q*`. Plus `dataset.json`: coins, windows,
  input names in order, quantile levels, cutoff, the command line and the git commit.
- Splits by time: rows whose window ends after the cutoff are test; the last 20% of each coin's pre-cutoff rows are
  validation, with a gap of h + 8h before them (`risk.Spec.Gap`); the rest is train. Held-out coins are all test.
- `--cutoff` defaults to 45 days before today, so a freshly trained model already has ~45 days of unseen data per
  coin at release (enough for the runtime gate on every window: ≥ 500 rows even at 8h).

## Model and training (nn/)

- **Network:** inputs → Linear(32) → ReLU → Linear(32) → ReLU → Linear(2 × 9). ReLU, not GELU: it is exact in Go.
- **Monotone quantile head:** per side, the 9 raw outputs `a` become `q_1 = softplus(a_1)`,
  `q_k = q_{k−1} + softplus(a_k)`: non-negative (an adverse move is ≥ 0) and never crossing.
- **Levels:** 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.97, 0.99. The bad case is 0.90; the denser tail keeps
  `ExceedChance` accurate for far stops and liquidation.
- **Loss:** mean pinball loss over both sides and all levels, in swing units.
- **Batches:** window-balanced (each batch draws equally from the eight windows), so the 5m rows don't swamp the 8h
  ones. Adam, lr 1e-3 with cosine decay, weight decay 1e-4, batch 4,096, up to 30 epochs, early stopping on
  validation pinball (patience 3). Fixed seeds; five seeds trained, the best on validation kept.
- **Size budget:** under 10,000 weights. If a wider net doesn't beat this one on validation by a clear margin, keep
  the small one.

## Evaluation (nn/deltarisk_nn/evaluate.py, also printed by `train`)

On the post-cutoff test rows, for training coins and held-out coins separately, per window and side:

- pinball loss of the network and of HAR, and the gain `1 − nn / har`;
- coverage of the 0.90 quantile (share of actual moves above it; want 7–13%);
- stop-hit calibration at the dashboard's stops (−5% and −10% margin at each tier's leverage), via the same
  `ExceedChance` interpolation the Go side uses (port it to Python with a test against the Go parity values).

Write the table to `nn/reports/<cutoff>.md`. Acceptance to export: on held-out coins the network's coverage is 7–13%
in most window-sides and its pinball loss is not worse than HAR's on average. If it isn't, don't export: tune or
stop.

## Export format (engine/nn/model/)

`spec.json`:

```json
{
  "version": 1,
  "trained_at": "2026-10-19T10:00:00Z",
  "cutoff_ms": 1788480000000,
  "git_commit": "…",
  "inputs": ["rv_1_rel", "…"],
  "mean": [0.0], "std": [1.0],
  "layers": [
    {"in": 41, "out": 32, "activation": "relu"},
    {"in": 32, "out": 32, "activation": "relu"},
    {"in": 32, "out": 18, "activation": "quantile_head"}
  ],
  "levels": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99],
  "windows": ["5m", "15m", "30m", "1h", "2h", "3h", "4h", "8h"],
  "report": {"heldout_pinball_gain": 0.031, "…": "…"}
}
```

`weights.bin`: for each layer in order, the weight matrix row-major (`out × in`, PyTorch's layout) then the bias
(`out`), float32 little-endian, nothing else. `Load` checks the file length against the spec exactly.

`testvectors.json`: 50 rows of standardised-before inputs from the test split, with the network's outputs (both
sides, all levels) as PyTorch computed them. The Go test checks `Forward` against them (absolute error ≤ 1e-5).

## Go inference (engine/nn/model.go)

- `Load(spec, weights []byte) (*Model, error)`: parse, validate (input names must equal `features.go`'s list in
  order: refuse to load otherwise), keep weights as float32, compute in float64.
- `(*Model).Forward(x []float64) (long, short []float64)`: standardise, the dense layers, the monotone head.
  No allocation in the hot path beyond two small buffers; one forward pass is ~3,000 multiply-adds.
- `Embedded()`: the committed model through `go:embed`, loaded once (`sync.Once`); a load error is logged and the
  dashboard runs HAR only.

## Serving and the gate (engine/risk)

- `Train` (per coin, per window), after the HAR walk-forward check: take the rows that are both in HAR's
  walk-forward test blocks and after the model's cutoff (out-of-sample for both models); need ≥ 500 of them. On those
  rows compute the network's pinball loss and coverage, and HAR's pinball loss from the walk-forward predictions.
  The network serves a side when its pinball loss is ≥ 2% lower (`Spec.MinGain`, restored) **and** its coverage is
  7–13%; otherwise HAR, exactly as today.
- `WindowState` keeps the choice per side; `Update` also runs the network at the latest close when it serves;
  `View` reads quantiles, bad case, stop-hit and liquidation chances from whichever model serves (`ExceedChance`
  over the network's quantiles × swing, as LightGBM did), and sets `checked` from that model's coverage.
- The snapshot gets back `model: "har" | "nn"` per side, and the risk block gets `nn_cutoff` (date) so the page
  can say how old the network is.

## Page

- `types.ts`: `SideRisk.model: "har" | "nn"`, `Risk.nn_cutoff: string | null`.
- `RiskCell.svelte` and the windows table in `CoinDetails.svelte`: a small "NN" mark next to the score when
  `model === "nn"` (caps label in the page's style, `--accent` text; check both themes and the phone layout).
- `risk.ts` tooltip: a line "Model: neural network trained to 2026-09-04" or "Model: volatility baseline (HAR)".
- `Glossary.svelte`: one sentence on the network and the gate.

## CLI

- `deltarisk risk`: columns for the network's coverage, pinball loss and gain beside HAR's, and the model chosen;
  a header line with the network's cutoff and age, or "no network loaded".
- `deltarisk dataset`: as above.

## mise tasks

| Task | Runs |
|---|---|
| `nn:install` | `cd nn && uv sync` |
| `nn:dataset` | `go run ./cmd/deltarisk dataset --coins nn/coins.toml --out nn/data` |
| `nn:train` | `cd nn && uv run deltarisk-nn train --data data --out ../engine/nn/model` (trains, evaluates, exports if accepted) |
| `nn` | `nn:dataset` then `nn:train` |
| `test` | also `cd nn && uv run pytest -q` |

`nn/data/` and `nn/reports/` are gitignored (the report is summarised in `spec.json`); `engine/nn/model/` is committed.

## Phases

Each phase ends with `go test -race ./...` (and `pytest` from phase 3) green and a commit.

1. **Features in Go.** `engine/nn/features.go` with its input list, the no-lookahead test and a test that every
   input is bounded or standardisable on real ETH data (the parity fixtures' candles).
2. **Dataset command.** `cmd/deltarisk/dataset.go`, `nn/coins.toml` (verify every coin has a Binance perpetual
   with enough history; replace those that don't). Run it once for real and check row counts, splits and that
   `har_*` quantiles reproduce HAR's walk-forward coverage roughly.
3. **Python package.** `nn/` with uv, data loading, the model, the loss, pytest.
4. **Training and evaluation.** The loop, the report. Iterate on inputs and size here, not later.
5. **Export and Go inference.** `export.py`, `engine/nn/model.go`, `embed.go`, the test vectors test. Commit the
   first model only if phase 4's acceptance passes.
6. **Gate and serving.** `engine/risk` changes, snapshot `model` and `nn_cutoff`, unit tests with a tiny hand-made
   model (one that always beats HAR, one that never does) so the gate is tested without the real weights.
7. **CLI and page.** `deltarisk risk` columns, the NN mark, the tooltip, the glossary; `mise run check`.
8. **Live check.** On a temp `DELTARISK_HOME`, run the dashboard on BTC, ETH, SOL, a meme coin and a held-out
   coin; compare `deltarisk risk` HAR vs network columns; look at the marks on the page in both themes.
9. **Docs.** `CLAUDE.md` (the network, the gate, the dataset command, retraining, the new tasks), `README.md`,
   a dated note at the top of `docs/risk-plan.md`.

## Risks

- **Distribution shift in the swing unit** between training (pre-cutoff HAR fit) and serving (full-history fit).
  Measured in phase 4 on the test split; if it hurts, fit the serving HAR on pre-cutoff rows too.
- **Staleness:** markets change; the gate catches a model that has stopped working (it falls back to HAR per coin),
  and the risk command shows the cutoff's age. Retrain every month or two.
- **A gate on few rows is noisy**, especially at 8h: keep the 500-row minimum and the 2% margin.
- **Survivorship:** the coin list is today's liquid coins. The held-out coins and the per-coin gate limit how much
  that matters; note it in the report.
- **PyTorch vs Go numerics:** float32 weights, float64 arithmetic in Go, test vectors at 1e-5; ReLU only.
