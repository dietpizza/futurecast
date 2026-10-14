# futurecast ⚡

A high-performance, **air-gapped cross-asset Gradient Boosted Decision Tree (LightGBM) forecasting engine** for high-beta cryptocurrency tokens (such as `DOGEUSDT` relative to `BTCUSDT` and `ETHUSDT`).

> **Zero-Execution Security Guarantee**: This system is architecturally constrained to **forecasting, signal calibration, telemetry streaming, and drift auditing only**. It contains zero exchange trading credentials, zero order routing routines, and zero signing methods, preventing live trade execution or capital risk under any condition.

---

## Architecture Overview

```
[ Public Market Data Feeds (Read-Only) ]
  (Binance / Bybit Public REST & WebSockets: DOGE, BTC, ETH)
               │
               ▼
   [ Data Ingestion & Storage Engine ]
  (Continuous UTC Alignment, Gap Handling, Parquet Storage)
               │
               ▼
   [ Vectorized Feature Engineering ]
  (35 Stationary Features: Lead-Lags, Volume Z-Scores, Basis, OFI Proxies)
               │
               ▼
   [ Target Generator (Offline/Training) ]
  (15-Minute Residual Alpha & Triple Barrier Labels)
               │
               ▼
 [ Purged & Embargoed Walk-Forward CV ] ◄── [ Hyperparameter Tuning (Optuna) ]
               │
               ▼
[ Model Registry & Artifact Store (LightGBM) ]
               │
               ▼
┌────────────────────────────────────────────────────────────────────────┐
│      AIR-GAPPED REAL-TIME FORECASTING & TELEMETRY ENGINE               │
│                                                                        │
│  • Strictly NO Private Keys / NO Trading APIs                          │
│  • Input: Real-Time Ingested Bars & Orderflow Proxies                  │
│  • Output: Calibrated Residual Alpha (\hat{y}_t), Beta Hedge Ratio,    │
│    Regime Indicators, Confidence Intervals                             │
│  • Delivery: In-Memory Bus, Local SQLite/Parquet Audit Log,           │
│    WebSocket / Webhook Event Stream, Terminal Rich Feed                │
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

## Validation Strategy (Zero-Leakage Protocol)

```
[ Window 1: Train (4 Months) ] ──[ Purge (60m) ]──► [ Test (1 Month) ] ──[ Embargo (24h) ]
         └── Step Forward 1 Month ──►
         [ Window 2: Train (4 Months) ] ──[ Purge (60m) ]──► [ Test (1 Month) ] ──[ Embargo (24h) ]
```

* **Purging (60 min):** Strips the final 60 bars of training to prevent target overlap ($t \to t+15$).
* **Embargoing (24 hours):** Discards 24 hours post-test window before beginning the next split.
* **Evaluation Metrics:** Spearman Rank Information Coefficient (IC), IC Information Ratio ($IC\_IR$), and Theoretical Long/Short Friction-Free Sharpe.

---

## Quickstart & CLI Commands

### ⚡ Quick Start (One Command)
```bash
# Default (Binance DOGEUSDT):
./run.sh

# Or any CoinDCX pair (auto-builds decision tree if model doesn't exist, then streams live):
./run.sh --symbol B-DOGE_USDT
./run.sh --symbol B-SOL_USDT
./run.sh --symbol I-BTC_INR
```

### 1. Modular Decision Tree Builder (`futurecast build`)
Paste or specify **ANY symbol** from CoinDCX or Binance. The engine automatically:
1. Detects the exchange (`coindcx` or `binance`)
2. Resolves primary and secondary benchmark assets (e.g. `B-BTC_USDT`/`B-ETH_USDT` or `I-BTC_INR`)
3. Fetches paginated klines with transparent disk caching in `.cache/`
4. Computes the 35 core cross-asset and microstructure features
5. Formulates 15m residual alpha targets and runs Purged Walk-Forward CV (or 80/20 train/val split)
6. Trains and saves the LightGBM model to `models/lightgbm_<symbol>_residual_alpha.txt`

```bash
# Build decision tree for any CoinDCX symbol:
uv run futurecast build --symbol B-DOGE_USDT --days 7
uv run futurecast build --symbol B-SOL_USDT --days 14

# Build decision tree for any Binance symbol:
uv run futurecast build --symbol DOGEUSDT --days 14
```

### 2. Ingestion & Local Disk Caching (.cache/)
```bash
# Ingest live public market data
uv run futurecast ingest --days 30

# Inspect or clear the .cache/ folder
uv run futurecast cache
uv run futurecast cache --clear

# Or generate realistic synthetic data for offline testing
uv run futurecast ingest --synthetic --days 35
```

### 2. Feature Generation & Stationarity Audit
```bash
uv run futurecast features --audit
```

### 3. Purged Walk-Forward Cross-Validation
```bash
uv run futurecast validate
```

### 4. Production Model Training
```bash
uv run futurecast train --output-model models/lightgbm_doge_residual_alpha.txt
```

### 5. Real-Time Multi-Coin Web Terminal (`futurecast web`)

Launch the local interactive Web UI dashboard to stream real-time signals across multiple coins concurrently:

```bash
# Launch Web UI on http://127.0.0.1:8000 (or via ./run.sh --web):
uv run futurecast web
./run.sh --web

# Custom host, port, or initial coin list:
uv run futurecast web --port 8000 --symbols B-DOGE_USDT,B-SOL_USDT,B-BLUR_USDT,DOGEUSDT
```

**Key Features of the Web Terminal:**
- **Dynamic Coin Management**: Add any symbol on the fly (e.g. `B-DOGE_USDT`, `PEPEUSDT`, `B-SOL_USDT`, `I-BTC_INR`) with automatic exchange detection and sector routing.
- **Grouped Coin Cards & Multi-Model Breakdown**: Each symbol has its own grouped section displaying real-time price, benchmark price, 15m residual alpha, and a nested table with rows for all predictive models (`LightGBM`, `XGBoost`, `CatBoost`, and `Consensus Ensemble`).
- **Cross-Verification Concordance**: Displays real-time model agreement scores (e.g. `100% Agreement (3/3 Models Bullish)` vs `⚠ Divergence Warning`).
- **Microstructure Telemetry Strip**: Real-time Parkinson Volatility, Relative Volume Ratio (RVR), Funding Rate Basis, and Order Flow Imbalance (OFI).

---

### 6. Dynamic Sector Benchmarks

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

### 7. Multi-Model Cross-Verification Suite

Combines three distinct gradient boosting architectures to eliminate single-model bias and reject flash liquidation noise:
- **LightGBM**: Fast leaf-wise tree growth with Huber / L1 objective.
- **XGBoost**: Depth-wise exact splits with pseudo-Huber regression loss and column subsampling.
- **CatBoost**: Oblivious symmetric trees with MAE loss to reject crypto flash wicks.
- **Consensus Ensemble**: Combines residual alpha predictions, computes concordance ratios, and flags directional divergence.

---

### 8. Run Test Suite (32/32 Passing)
```bash
uv run pytest -v
```
