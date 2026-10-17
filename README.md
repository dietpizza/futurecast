# futurecast ⚡

A local, read-only **web dashboard for CoinDCX futures** that runs plain statistics on each coin's
price momentum. No models are trained. For every coin you add, at four horizons (1h · 4h · 1d · 3d), it shows:

- the **last move** in volatility units (z) and its percentile among every past move of that length,
- **what followed similar moves** on this coin: the next window's average return, how often it rose,
  and a 95% range for that average that allows for the overlap between windows,
- whether the coin's price **trends or mean-reverts** at that horizon (variance-ratio test) and the
  correlation between each move and the next,
- a **Long / Short signal** only when the average after similar moves beats a market round trip in
  fees, its 95% range excludes zero, and it holds in both halves of the history,
- **supporting signals**: taker flow, moves against the sector benchmark after taking out beta, RSI, the 1h
  EMA 20/50/200 stack, position in the 24h / 7d / 30d range, realized volatility, volume against a
  typical day, funding rate percentile, and the order book.

The `futurecast backtest` command also replays two rule-based setups (fair-value-gap setups and wedge
breakouts) on history. They are not on the dashboard.

> **Zero-Execution Security Guarantee**: This system is read-only. It contains zero exchange trading credentials, zero order routing routines, and zero signing methods, preventing live trade execution or capital risk under any condition. `AppConfig.enforce_zero_execution_policy()` refuses to start otherwise.

---

## Architecture Overview

```
[ Binance USD-M Futures Public REST (Read-Only) ]
  (closed 15m / 1h candles of the coin and its benchmark, the coin's settled
   funding, live price, order book; candles cached on disk)
               │
               ▼
   [ Alignment ]   coin and benchmark on one candle grid per chart
               │
               ▼
   [ stats.momentum ]  z, event study, variance ratio, IC, signal   (per horizon)
   [ stats.signals ]   RSI, EMA stack, ranges, volatility, volume, funding
               │
               ▼
┌────────────────────────────────────────────────────────────────────────┐
│      READ-ONLY WEB DASHBOARD (FastAPI + WebSocket, port 8000)          │
│                                                                        │
│  • Strictly NO Private Keys / NO Trading APIs                          │
│  • Recomputed whenever a 15m or 1h candle closes                       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Momentum statistics

Each horizon is a number of candles on one chart: 1h and 4h use 15-minute candles (180 days of
history), 1d and 3d use hourly candles (2 years). Everything is computed on closed candles, in
`src/futurecast/stats/momentum.py` and `stats/signals.py`.

- **Move:** z = the log return over the last h candles ÷ (σ √h), where σ is the standard deviation
  of one-candle returns over the trailing 14 days. z is roughly standard normal under a random walk.
- **After similar moves:** every past z goes into a fifth (quintile). For the fifth the latest move falls
  in: the mean, median and share-up of the following h-candle return, and the same over every window
  (the coin's drift). Windows start every candle, so they overlap; the 95% range of the mean uses
  Hansen–Hodrick standard errors (autocovariances up to h − 1 lags), which are exact for that overlap.
  The mean is also split into the older and newer half of the history.
- **Persistence:** the Lo–MacKinlay variance ratio VR(h) with its heteroskedasticity-robust z
  (above 1 trending, below 1 mean-reverting, p < 0.05 to count), and the correlation of z with the next
  window's volatility-scaled return (IC) with an overlap-corrected t.
- **Signal:** Long or Short when the latest move is outside the middle fifth and, after moves in its
  fifth, the next window averaged more than a market round trip (2 × the 0.0885% CoinDCX taker fee
  with GST), the 95% range excludes zero, both halves agree in sign, and at least 30 non-overlapping
  windows stand behind it. A **continuation** follows the move; a **reversal** fades it.

The signal comes from an event study over the history on hand (the latest windows included), not
from a backtest of trading it. On 17 October 2026 every horizon of BTC, ETH, SOL, BAT, INJ and DASH
read "no edge": after moves like the latest ones, the averages were a few hundredths of a percent,
inside their 95% ranges and mostly under the fees.

---

## Setups

The rules run on a pair of charts: 15-minute entries in 1-hour context by default (`SetupSpec`).

1. **Context:** the context chart's trend points the trade's way (its latest structure break), and the
   entry chart's pullback reached into a context-chart fair value gap that existed before price got
   there and that no context candle has closed through since.
2. **Shift:** an entry-chart change of character (CHoCH) in the trade's direction.
3. **Gap:** the move that broke structure left an entry-chart fair value gap behind.
4. **Orders:** a limit at that gap's midpoint (waiting up to 30 entry candles), a stop just past the
   pullback's extreme, take-profits at 2R, 3R and 4R (the limit is cancelled if price runs 3R away first). Once filled the trade is left alone for up to 240
   entry candles.

Every entry-chart CHoCH that leaves a gap is a setup; those passing both context checks are grade
**A+**, the rest grade B. Backtests take A+ setups unless told otherwise (`--grade all`). Swings,
breaks and gaps use only candles that have closed.

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

Results to October 2026, A+ setups, 3R target, average result per trade after a flat 0.10% round-trip
cost (the model before per-leg fees):

| Charts (entry / context) | History | BTC | ETH | SOL | BAT |
|---|---|---|---|---|---|
| 1m / 15m | 69 days | −0.88R | −0.60R | −0.51R | −0.42R |
| 5m / 1h | 1 year | −0.37R | −0.28R | −0.43R | −0.14R |
| **15m / 1h** | 2 years | −0.50R | +0.08R | +0.02R | +0.02R |
| 15m / 4h | 2 years | −0.36R | −0.16R | −0.14R | −0.15R |

With per-leg fees, 15m / 1h is a little worse on every coin, because about 70% of trades end at the
stop, which costs more than the flat rate:

| 15m / 1h, per-leg fees | BTC | ETH | SOL | BAT |
|---|---|---|---|---|
| 2R | −0.48R | −0.01R | −0.01R | −0.02R |
| 3R | −0.56R | +0.06R | −0.01R | +0.01R |
| 4R | −0.57R | +0.06R | −0.00R | −0.06R |

Higher charts help mostly because fees shrink relative to the stop (0.4–0.7R per trade on 1m entries,
about 0.2R on 15m). Before fees, A+ setups average close to 0R almost everywhere: as coded, the strategy
has no clear edge, and none of these results is large enough to trust over 170–200 trades.

### Wedge breakouts (second strategy, backtest only)

```bash
uv run futurecast backtest B-ETH_USDT --strategy wedge                 # 1h, 2 years, A+, 2R, 1% risk, at most 5x
uv run futurecast backtest B-SOL_USDT --strategy wedge --entry 4h --days 2190 --target measured --grade all
```

A second rule set on a single chart, not shown on the dashboard. A wedge is two converging lines, one
through the latest two swing highs and one through the latest two swing lows, with no close outside
them since the first pivot and an apex 10–30 candles ahead. A close beyond a line is a breakout (A+ when
its volume is 1.5× the 20-candle average and its body is bigger than ATR(14)). Price must then come back
to within 0.2% of the broken line within 5 candles without closing 0.2% back inside it, and that candle
or the next must close the trade's way in the top (bottom) half of its range. The entry is a market
order at that close, the stop 0.5 ATR beyond the retest, the target 2R, 2.5R or the wedge's height
(`--target measured`), and the stop moves to break-even plus fees at 1.5R (`--no-breakeven` keeps it).
Positions are capped at 5× the account (`--max-leverage`), so a trade with a stop too tight for that
risks less than 1%. Rising wedges, falling wedges and triangles are reported separately.

Results to October 2026, all grades (A+ alone leaves 1–45 trades a coin), 2R target, average result per
trade after fees, trades in brackets:

| Chart | History | BTC | ETH | SOL | BAT |
|---|---|---|---|---|---|
| 15m | 2 years | −0.50R (223) | −0.28R (233) | −0.37R (181) | −0.17R (167) |
| 1h | 2 years | −0.24R (55) | −0.23R (42) | −0.38R (23) | −0.04R (40) |
| 4h | 2 years | −0.17R (11) | +0.05R (8) | +0.36R (12) | +0.29R (9) |

The 4h results looked good over two years, but there were too few trades to judge. Over 6 years and ten coins (BTC, ETH,
SOL, BAT, BNB, XRP, DOGE, ADA, LINK, AVAX) they come to −0.02R a trade over 212 trades (90% bootstrap
interval −0.18R to +0.14R), with good and bad years alternating. Before fees, wedge trades average about
0R on every chart, the same as the fair-value-gap setups, and the market entry makes fees heavier (0.25–0.55R a
trade on 15m). Moving the stop to break-even does not help consistently. As coded, this strategy has no
edge either.

## Running the Dashboard

```bash
./run.sh
```

That starts the dashboard on port 8000 on every network interface: open **http://127.0.0.1:8000**
on the same machine, or the machine's IP from elsewhere. The dashboard has no login, so anyone who
can reach the port can add and remove coins; firewall the port or pass `--host 127.0.0.1`
to keep it local. `./run.sh` is a thin wrapper around the `futurecast` command, so these are
equivalent:

```bash
uv run futurecast                    # same as ./run.sh
uv run futurecast --port 8002        # or: PORT=8002 ./run.sh
uv run futurecast --host 127.0.0.1   # this machine only
uv run futurecast --offline          # simulated random-walk candles, no network
```

`futurecast web` is still accepted and does the same thing.

In the dashboard, add any CoinDCX futures pair (e.g. `B-DOGE_USDT`, `B-SOL_USDT`). A new coin's
history downloads in the background, one coin at a time (the card shows the progress); from then on
each candle is appended as it closes. All market data comes from the Binance perpetual
behind the pair (`B-SOL_USDT` → `SOLUSDT`, `B-1000PEPE_USDT` → `1000PEPEUSDT`, INR pairs → their
`USDT` perpetual, priced in USDT). The add-coin search lists every Binance crypto and commodity
perpetual under its CoinDCX name; CoinDCX itself is never called.

Everything the app keeps lives in `~/.futurecast` (set `FUTURECAST_HOME` to use another folder):

| Path | Contents |
|---|---|
| `tracked_symbols.json` | Coins on the dashboard |
| `cache/candles/` | Downloaded closed candles, so restarts and backtests only fetch new ones |
| `config/*.yaml` | Optional overrides (`assets.yaml`, `safety.yaml`); any file left out uses the defaults shown in this repo's `config/` |

Older versions also kept `models/` and `logs/forecasts/` there; nothing reads them any more, so they
can be deleted.

---

### Sector Benchmarks

Each coin's moves are compared with its sector's leader, after taking out beta. A coin is never its
own benchmark.

| Sector | Benchmark | Example coins |
|---|---|---|
| **Meme** | `DOGE` | `PEPE`, `SHIB`, `FLOKI`, `BONK`, `WIF` |
| **L1** | `SOL` | `AVAX`, `SUI`, `APT`, `ADA`, `NEAR` |
| **DeFi** | `ETH` | `UNI`, `AAVE`, `CRV`, `LDO`, `PENDLE` |
| **NFT / Gaming** | `BLUR` | `SAND`, `MANA`, `AXS`, `GALA`, `APE` |
| **AI** | `NEAR` | `RENDER`, `FET`, `TAO`, `AR` |
| **Infrastructure** | `LINK` | `PYTH`, `FIL` |
| **Leaders and others** | `BTC` | `DOGE`, `SOL`, `ETH`, `BLUR`, `LINK` and unlisted coins; `BTC` uses `ETH` |

---

### Run the Test Suite
```bash
uv run pytest -v
```
