# futurecast ⚡

A local **web dashboard for CoinDCX futures**, built around a fair-value-gap strategy: a higher chart
for context, a lower chart for entries (1-hour and 15-minute by default). For every coin you add it shows:

- **market structure** on both charts (swing points, breaks of structure, changes of character) and
  the open **fair value gaps** nearest to price, found by exact rules,
- the latest long and short **setup** with its limit entry, stop and 3R / 4R take-profit, an A+ checklist,
  and what happened to its orders,
- a **backtest** of the rules on two years of the coin's history: win rate, average R, profit factor,
  drawdown and an equity curve, risking 1% per trade,
- each setup's **win chance** from LightGBM / XGBoost / CatBoost trained on the coin's own past setups,
  next to the break-even after fees, and whether models or rules made money on history they never saw,
- the **worst move** to expect against each side per horizon (5m · 15m · 1h · 4h · 8h), a risk score and a
  maximum leverage.

A secondary 15-minute cross-asset "residual alpha" forecast still trains and is logged, but is not shown.

> **Zero-Execution Security Guarantee**: This system is architecturally constrained to **forecasting, signal calibration, telemetry streaming, and drift auditing only**. It contains zero exchange trading credentials, zero order routing routines, and zero signing methods, preventing live trade execution or capital risk under any condition.

---

## Architecture Overview

```
[ Binance USD-M Futures Public REST (Read-Only) ]
  (closed 1m / 15m / 1h candles and settled funding of the coin and two benchmarks,
   live price, order book; candles cached on disk)
               │
               ▼
   [ Alignment & Feature Engineering ]   38 features: returns, betas, volatility,
               │                         volume, taker flow, funding, price geometry
               ▼
   [ Rules ]   swing points, BOS / CHoCH, fair value gaps on 1h and 15m → setups
               │
               ▼
   [ Labels ]  setup filled? take-profit before stop (3R, 4R); worst move;
               │  15-minute residual alpha
               ▼
   [ Time-ordered fit → calibrate → test, no outcome crossing a slice boundary ]
   [ Volatility regimes: hidden Markov model (risk) ]
               │
               ▼
┌────────────────────────────────────────────────────────────────────────┐
│      READ-ONLY WEB DASHBOARD (FastAPI + WebSocket, port 8000)          │
│                                                                        │
│  • Strictly NO Private Keys / NO Trading APIs                          │
│  • Output: structure, gaps, setups with entry / stop / target, win    │
│    chance, worst move, max leverage, and the evidence behind each      │
│  • Audit: JSONL forecast log in ~/.futurecast/logs/forecasts/ (≤ 32 MB) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Features

Window lengths count candles of the interval being modelled (1m, 15m or 1h). "btc" and "doge" in
feature names mean the primary benchmark and the target coin.

| Group | Count | Features |
|---|---|---|
| **Cross-asset** | 19 | Log returns over 1 / 5 / 15 / 60 candles for the coin and both benchmarks, rolling betas (60 and 1440 candles), beta-residual momentum, return spreads to both benchmarks, RSI spread. |
| **Volume & volatility** | 11 | Volume z-scores (60, 1440), relative volume ratio, Parkinson high-low volatility (15, 60, 240) and the benchmark's (60), volatility ratio, taker order-flow imbalance (5, 15, z-score over 60). |
| **Funding** | 3 | Settled funding rate, its deviation from the 3-day mean, spread to the benchmark's funding. |
| **Price geometry** | 5 | Distance to the 60-candle range, 240-candle VWAP basis, upper / lower wick and body ratios. |

---

## Setups

The rules run on a pair of charts: 15-minute entries in 1-hour context by default (`SetupSpec`).

1. **Context:** the context chart's trend points the trade's way (its latest structure break), and the
   entry chart's pullback reached into a context-chart fair value gap that existed before price got
   there and that no context candle has closed through since.
2. **Shift:** an entry-chart change of character (CHoCH) in the trade's direction.
3. **Gap:** the move that broke structure left an entry-chart fair value gap behind.
4. **Orders:** a limit at that gap's midpoint (waiting up to 30 entry candles), a stop just past the
   pullback's extreme, take-profits at 3R and 4R. Once filled the trade is left alone for up to 240
   entry candles.

Every entry-chart CHoCH that leaves a gap is a setup; those passing both context checks are grade
**A+**, the rest grade B. Only A+ setups raise a signal. Swings, breaks and gaps use only candles that
have closed.

## Backtest

```bash
uv run futurecast backtest B-ETH_USDT                                  # 15m / 1h, 2 years, A+, 3R, 1% risk
uv run futurecast backtest B-BTC_USDT --entry 5m --context 1h --days 365 --target 4 --csv trades.csv
```

It replays every setup with what its orders really did, as one account: A+ setups only, one at a time
(a waiting order or an open trade blocks the next), 1% of equity risked per trade, compounding, after
fees. Fees follow how the orders are placed: the limit entry and the take-profit pay the maker fee, the
stop pays the taker fee plus slippage, a timeout exit the taker fee. The defaults are CoinDCX USDT-M
futures (0.025% maker, 0.075% taker, plus 18% GST) and 2 bps of stop slippage, so a win costs 0.059%
round trip and a stop 0.138%; change them with `--maker`, `--taker` and `--slippage` (basis points).
Each trained coin's card shows the same backtest on its training history.

Results to October 2026, A+ setups, 3R target, average result per trade after a flat 0.10% round-trip
cost (the model before per-leg fees):

| Charts (entry / context) | History | BTC | ETH | SOL | BAT |
|---|---|---|---|---|---|
| 1m / 15m | 69 days | −0.88R | −0.60R | −0.51R | −0.42R |
| 5m / 1h | 1 year | −0.37R | −0.28R | −0.43R | −0.14R |
| **15m / 1h** | 2 years | −0.50R | +0.08R | +0.02R | +0.02R |
| 15m / 4h | 2 years | −0.36R | −0.16R | −0.14R | −0.15R |

With per-leg fees, 15m / 1h is a little worse on every coin, because about 70% of trades end at the
stop, which costs more than the flat rate: BTC −0.56R, ETH +0.06R, SOL −0.01R, BAT +0.01R at 3R
(4R: −0.57R, +0.06R, −0.00R, −0.06R).

Higher charts help mostly because fees shrink relative to the stop (0.4–0.7R per trade on 1m entries,
about 0.2R on 15m). Before fees, A+ setups average close to 0R almost everywhere: as coded, the strategy
has no clear edge, and none of these results is large enough to trust over 170–200 trades.

## What the models predict

- **Win chance:** whether a filled setup reaches its take-profit before its stop (3R and 4R each get
  their own models), from the setup's geometry, its 15m context, the time of day and the 38 market
  features. Touching both in one candle counts as a loss.
- **Worst move:** quantile regressors estimate the 95th percentile of how far price moves against each
  side within the horizon, which sets the risk score and maximum leverage.
- **Residual alpha (secondary):** the coin's 15-minute return minus beta times its benchmark's return.

---

## Validation (Zero-Leakage Protocol)

Every model uses the coin's history strictly in time order, and no trade still running in one slice
resolves in the next:

```
| fit | early-stop |  calibrate  |    test    |
oldest ------------------------------------> newest
```

* **Calibration:** the three models' average is mapped onto observed win frequencies with Platt scaling.
* **Baseline:** the rules alone: how often filled setups of the same grade (A+ or B) won before the test slice.
* **Edge test:** the models have an edge only when, on the untouched test slice, they beat that baseline
  (lower bound of a block-bootstrap interval of their Brier skill above zero) and rank winners better than
  the grade does. Otherwise the rules' win rate is shown.
* **Trading check:** the test stretch is replayed one setup at a time after fees: every A+ setup (the
  rules), and the A+ setups whose win chance cleared break-even (the models). A signal is raised only when
  the replay behind the chance shown made money on at least 10 trades.
* **Worst move:** its 95% bound is calibrated separately in each volatility regime, so it also holds in
  stormy stretches, and it is shown only if it beats a volatility-only baseline.

See `src/futurecast/features/setups.py`, `src/futurecast/models/setup_model.py`, `src/futurecast/backtest.py`
and `trade_risk.py` for the full procedure.

---

## Running the Dashboard

```bash
./run.sh
```

That starts the dashboard on port 8000 on every network interface: open **http://127.0.0.1:8000**
on the same machine, or the machine's IP from elsewhere. The dashboard has no login, so anyone who
can reach the port can add coins and start training; firewall the port or pass `--host 127.0.0.1`
to keep it local. `./run.sh` is a thin wrapper around the `futurecast` command, so these are
equivalent:

```bash
uv run futurecast                    # same as ./run.sh
uv run futurecast --port 8002        # or: PORT=8002 ./run.sh
uv run futurecast --host 127.0.0.1   # this machine only
uv run futurecast --offline          # simulated candles, no network (no training)
```

`futurecast web` is still accepted and does the same thing.

In the dashboard, add any CoinDCX futures pair (e.g. `B-DOGE_USDT`, `B-SOL_USDT`). New coins are
trained automatically from their real candle history (the card shows the download, then training
progress) and can be retrained from the dashboard. All market data comes from the Binance perpetual
behind the pair (`B-SOL_USDT` → `SOLUSDT`, `B-1000PEPE_USDT` → `1000PEPEUSDT`, INR pairs → their
`USDT` perpetual, priced in USDT). The add-coin search lists every Binance crypto and commodity
perpetual under its CoinDCX name; CoinDCX itself is never called.

Everything the app keeps lives in `~/.futurecast` (set `FUTURECAST_HOME` to use another folder):

| Path | Contents |
|---|---|
| `tracked_symbols.json` | Coins on the dashboard |
| `models/` | Trained models |
| `cache/candles/` | Downloaded closed candles, so retrains and restarts only fetch new ones |
| `logs/forecasts/` | Residual-alpha forecasts as JSONL, at most 32 MB in total (oldest deleted first) |
| `config/*.yaml` | Optional overrides; any file left out uses the defaults shown in this repo's `config/` |

---

### Sector Benchmarks

Each coin is aligned with two benchmarks, chosen by sector. A coin is never its own benchmark.

| Sector | Benchmarks | Example coins |
|---|---|---|
| **Meme** | `DOGE`, `BTC` | `PEPE`, `SHIB`, `FLOKI`, `BONK`, `WIF` |
| **L1** | `SOL`, `BTC` | `AVAX`, `SUI`, `APT`, `ADA`, `NEAR` |
| **DeFi** | `ETH`, `BTC` | `UNI`, `AAVE`, `CRV`, `LDO`, `PENDLE` |
| **NFT / Gaming** | `BLUR`, `BTC` | `SAND`, `MANA`, `AXS`, `GALA`, `APE` |
| **AI** | `NEAR`, `BTC` | `RENDER`, `FET`, `TAO`, `AR` |
| **Leaders and others** | `BTC`, `ETH` | `DOGE`, `SOL`, `BLUR` and unlisted coins; `BTC` uses `ETH`, `SOL` and `ETH` uses `BTC`, `SOL` |

---

### Run the Test Suite
```bash
uv run pytest -v
```
