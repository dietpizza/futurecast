# futurecast ⚡

A local **web dashboard for CoinDCX futures**: it tracks any number of coins, trains LightGBM / XGBoost / CatBoost models on their real 5m and 15m candles, and serves calibrated long / short win probabilities per horizon next to a cross-asset residual alpha forecast.

> **Zero-Execution Security Guarantee**: This system is architecturally constrained to **forecasting, signal calibration, telemetry streaming, and drift auditing only**. It contains zero exchange trading credentials, zero order routing routines, and zero signing methods, preventing live trade execution or capital risk under any condition.

---

## Architecture Overview

```
[ Binance USD-M Futures Public REST (Read-Only) ]
  (closed 5m / 15m candles of the coin and its sector benchmarks, live price, order book)
               │
               ▼
   [ Alignment & Vectorized Feature Engineering ]
  (Lead-Lags, Volume Z-Scores, Volatility, Basis, OFI Proxies, Order Book)
               │
               ▼
   [ Labels ]
  (Residual Alpha, Take-Profit / Stop-Loss Barrier Outcomes per Horizon)
               │
               ▼
 [ Time-Ordered Fit → Calibrate → Test, One Horizon Gap Between Slices ]
               │
               ▼
[ Per-Coin Model Store: LightGBM · XGBoost · CatBoost + Platt Calibration ]
               │
               ▼
┌────────────────────────────────────────────────────────────────────────┐
│      READ-ONLY WEB DASHBOARD (FastAPI + WebSocket, localhost)          │
│                                                                        │
│  • Strictly NO Private Keys / NO Trading APIs                          │
│  • Input: Closed Candles, Live Price & Order Book                      │
│  • Output: Calibrated Win Probabilities (5m · 15m · 1h · 4h · 8h),     │
│    Residual Alpha, Beta Hedge Ratio, Trade Risk, Model Agreement       │
│  • Audit: JSONL Forecast Log in logs/forecasts/                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 35 Core Stationary Predictors

| Feature Group | Count | Key Features |
|---|---|---|
| **Group A: Cross-Asset Dynamics** | 10 | Multi-horizon log returns ($k \in \{1, 5, 15, 60\}$ min for DOGE, BTC, ETH), Rolling Betas ($\beta_{60\text{m}}$, $\beta_{1440\text{m}}$), Beta Residual Momentum ($15\text{m}$), RSI Spreads ($\Delta \text{RSI}_{14}$). |
| **Group B: Microstructure & Volatility** | 12 | Rolling Volume Z-Scores ($60\text{m}, 1440\text{m}$), Relative Volume Ratio ($\text{RVR}_{15}$), Parkinson High-Low Volatility ($15\text{m}, 60\text{m}, 240\text{m}$), Realized Volatility Ratio, Order Flow Imbalance (OFI) taker buy/sell proxy. |
| **Group C: Derivative Positioning** | 8 | 8-hour Funding Rate deviations from 3-day EMA, Cross-Asset Funding Basis ($\text{DOGE} - \text{BTC}$), Open Interest % Deltas ($15\text{m}, 60\text{m}$), OI-Price Divergence ($\text{sign}(\Delta P) \times \text{sign}(\Delta \text{OI})$), Liquidation Flush Proxy. |
| **Group D: Price Geometry & Technical** | 5 | Distance to Rolling Extremes (Stochastic %K), 4-hour VWAP Basis (bps), Upper Wick Ratio, Lower Wick Ratio, Body-to-Range Ratio. |

---

## Target Formulation

The primary training target is **15-Minute Forward Residual Alpha**:

$$y_t = R_{\text{DOGE}, t \to t+15} - \beta_{t, 1440} \cdot R_{\text{BTC}, t \to t+15}$$

* $\beta_{t, 1440}$ is strictly point-in-time at bar close $t$ (backward-looking).
* Loss function: **Huber Loss / L1 (MAE)** to mitigate crypto liquidation wicks and fat-tailed shocks.

---

## Validation (Zero-Leakage Protocol)

Training a coin uses its candle history strictly in time order, with a one-horizon gap
between slices so overlapping trade windows cannot leak outcomes across a boundary:

```
| fit | early-stop |  calibrate  |    test    |
oldest ------------------------------------> newest
```

* **Calibration:** the three classifiers' average is mapped onto observed win frequencies with Platt scaling.
* **Edge test:** a side is shown as having edge only when the lower bound of a block-bootstrap interval of its Brier skill on the untouched test slice is above zero; otherwise the recent base rate is served.
* **Trading check:** the test slice is replayed as a strategy, and a long / short signal is raised only for sides whose replay made money after costs.

See `src/futurecast/models/trade_probability.py` for the full procedure.

---

## Running the Dashboard

```bash
./run.sh
```

That starts the dashboard on **http://127.0.0.1:8000**. `./run.sh` is a thin wrapper around the
`futurecast` command, so these are equivalent:

```bash
uv run futurecast                    # same as ./run.sh
uv run futurecast --port 8002        # or: PORT=8002 ./run.sh
uv run futurecast --host 0.0.0.0     # listen on every interface
uv run futurecast --offline          # simulated candles, no network (no training)
```

`futurecast web` is still accepted and does the same thing.

In the dashboard, add any CoinDCX futures pair (e.g. `B-DOGE_USDT`, `B-SOL_USDT`). New coins are
trained automatically from their real candle history and can be retrained from the dashboard.
All market data comes from the Binance perpetual behind the pair (`B-SOL_USDT` → `SOLUSDT`,
`B-1000PEPE_USDT` → `1000PEPEUSDT`, INR pairs → their `USDT` perpetual, priced in USDT); CoinDCX is
only asked for its list of contracts to search.
Tracked coins are remembered in `data/tracked_symbols.json`, trained models are saved under
`models/`, and every forecast is appended to `logs/forecasts/`.

**Key Features of the Web Terminal:**
- **Dynamic Coin Management**: Add any CoinDCX futures pair on the fly (e.g. `B-DOGE_USDT`, `B-SOL_USDT`, `I-BTC_INR`) with automatic sector routing.
- **Grouped Coin Cards & Multi-Model Breakdown**: Each symbol has its own grouped section displaying real-time price, benchmark price, 15m residual alpha, and a nested table with rows for all predictive models (`LightGBM`, `XGBoost`, `CatBoost`, and `Consensus Ensemble`).
- **Cross-Verification Concordance**: Displays real-time model agreement scores (e.g. `100% Agreement (3/3 Models Bullish)` vs `⚠ Divergence Warning`).
- **Microstructure Telemetry Strip**: Real-time Parkinson Volatility, Relative Volume Ratio (RVR), Funding Rate Basis, and Order Flow Imbalance (OFI).

---

### Dynamic Sector Benchmarks

The system dynamically routes category leaders based on token taxonomy:

| Sector | Category Leader | Benchmark Routing | Tokens |
|---|---|---|---|
| **MEME** | `DOGE` | `[DOGE, BTC]` | `PEPE`, `SHIB`, `FLOKI`, `BONK`, `WIF`, `MEME`, `POPCAT` |
| **L1 / Alt-L1** | `SOL` | `[SOL, BTC]` | `AVAX`, `SUI`, `APT`, `ADA`, `DOT`, `FTM`, `NEAR`, `SEI` |
| **DEFI** | `ETH` | `[ETH, BTC]` | `UNI`, `AAVE`, `CRV`, `MKR`, `SNX`, `COMP`, `LDO`, `PENDLE` |
| **NFT / GAMING** | `BLUR` | `[BLUR, BTC]` | `BLUR`, `SAND`, `MANA`, `AXS`, `GALA`, `APE` |
| **AI / COMPUTE** | `NEAR` | `[NEAR, BTC]` | `RENDER`, `FET`, `TAO`, `AGIX`, `AR`, `AKT` |
| **MAJORS** | `BTC / ETH` | `[BTC, ETH]` | `BTC`, `ETH`, `SOL`, `DOGE` |

---

### Multi-Model Cross-Verification Suite

Combines three distinct gradient boosting architectures to eliminate single-model bias and reject flash liquidation noise:
- **LightGBM**: Fast leaf-wise tree growth with Huber / L1 objective.
- **XGBoost**: Depth-wise exact splits with pseudo-Huber regression loss and column subsampling.
- **CatBoost**: Oblivious symmetric trees with MAE loss to reject crypto flash wicks.
- **Consensus Ensemble**: Combines residual alpha predictions, computes concordance ratios, and flags directional divergence.

---

### Run the Test Suite
```bash
uv run pytest -v
```
