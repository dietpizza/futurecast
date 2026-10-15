# futurecast ⚡

A local **web dashboard for CoinDCX futures**. For every coin you add it trains LightGBM / XGBoost /
CatBoost on the coin's real closed 1m and 15m candles and shows, per horizon (5m · 15m · 1h · 4h · 8h):

- calibrated **long / short win probabilities**, with a take-profit and stop the models pick for the current candle,
- the **worst move** to expect against each side (95% of cases), a risk score and a maximum leverage,
- whether any of it beat simpler baselines on recent history the models never saw.

A secondary 15-minute cross-asset "residual alpha" forecast still trains and is logged, but is not shown.

> **Zero-Execution Security Guarantee**: This system is architecturally constrained to **forecasting, signal calibration, telemetry streaming, and drift auditing only**. It contains zero exchange trading credentials, zero order routing routines, and zero signing methods, preventing live trade execution or capital risk under any condition.

---

## Architecture Overview

```
[ Binance USD-M Futures Public REST (Read-Only) ]
  (closed 1m / 15m candles and settled funding of the coin and two benchmarks,
   live price, order book; candles cached on disk)
               │
               ▼
   [ Alignment & Feature Engineering ]   38 features: returns, betas, volatility,
               │                         volume, taker flow, funding, price geometry
               ▼
   [ Labels ]  take-profit before stop (4 distance pairs per horizon), worst move,
               │  15-minute residual alpha
               ▼
   [ Time-ordered fit → calibrate → test, one horizon gap between slices ]
   [ Volatility regimes: hidden Markov model; volatility-only odds: absorbing Markov chain ]
               │
               ▼
┌────────────────────────────────────────────────────────────────────────┐
│      READ-ONLY WEB DASHBOARD (FastAPI + WebSocket, port 8000)          │
│                                                                        │
│  • Strictly NO Private Keys / NO Trading APIs                          │
│  • Output: win odds, take-profit / stop, worst move, max leverage,     │
│    volatility regime, and the evidence behind each                     │
│  • Audit: JSONL forecast log in ~/.futurecast/logs/forecasts/ (≤ 32 MB) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Features

Window lengths count candles of the interval being modelled (1m or 15m). "btc" and "doge" in
feature names mean the primary benchmark and the target coin.

| Group | Count | Features |
|---|---|---|
| **Cross-asset** | 19 | Log returns over 1 / 5 / 15 / 60 candles for the coin and both benchmarks, rolling betas (60 and 1440 candles), beta-residual momentum, return spreads to both benchmarks, RSI spread. |
| **Volume & volatility** | 11 | Volume z-scores (60, 1440), relative volume ratio, Parkinson high-low volatility (15, 60, 240) and the benchmark's (60), volatility ratio, taker order-flow imbalance (5, 15, z-score over 60). |
| **Funding** | 3 | Settled funding rate, its deviation from the 3-day mean, spread to the benchmark's funding. |
| **Price geometry** | 5 | Distance to the 60-candle range, 240-candle VWAP basis, upper / lower wick and body ratios. |

---

## What the models predict

- **Win probability:** a long wins if price touches the take-profit before the stop within the horizon;
  a timeout or touching both in one candle counts as a loss. Four take-profit / stop pairs, scaled to
  recent volatility (1:1 tight and wide, 2:1, 1:2), each get their own models; for the current candle
  the pair with the highest expected profit after fees is shown.
- **Worst move:** quantile regressors estimate the 95th percentile of how far price moves against each
  side within the horizon, which sets the risk score and maximum leverage.
- **Residual alpha (secondary):** the coin's 15-minute return minus beta times its benchmark's return.

---

## Validation (Zero-Leakage Protocol)

Every model uses the coin's candle history strictly in time order, with a one-horizon gap
between slices so overlapping trade windows cannot leak outcomes across a boundary:

```
| fit | early-stop |  calibrate  |    test    |
oldest ------------------------------------> newest
```

* **Calibration:** the three models' average is mapped onto observed win frequencies with Platt scaling.
* **Baseline:** the odds someone with no view on direction would quote. A hidden Markov model of
  calm / normal / stormy volatility forecasts how much the coin will move, and an absorbing Markov
  chain turns that into the chance of touching the take-profit first.
* **Edge test:** a side has an edge only when, on the untouched test slice, it beats that baseline
  (lower bound of a block-bootstrap interval of its Brier skill above zero) *and* its long / short
  split predicts outcomes. Otherwise the volatility-only odds are shown.
* **Trading check:** the test slice is replayed as a strategy (pick the best pair each candle, trade
  when the odds clear break-even); a long / short signal is raised only for sides whose replay made money
  after fees on at least 10 trades.
* **Worst move:** its 95% bound is calibrated separately in each volatility regime, so it also holds in
  stormy stretches, and it is shown only if it beats a volatility-only baseline.

See `src/futurecast/models/trade_probability.py` and `trade_risk.py` for the full procedure.

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
