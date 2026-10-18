# DeltaRisk

Formerly FutureCast: an old `~/.futurecast` folder is moved to `~/.deltarisk` on first start, and
`FUTURECAST_HOME` still works.

A local, read-only **web dashboard for CoinDCX futures** that answers one question for intraday
trading: *if I open a long or a short on this coin right now, how badly can it hurt me before I close it?*
For every coin you add, at eight holding windows (5m · 15m · 30m · 1h · 2h · 3h · 4h · 8h), it shows:

- **Long risk** and **short risk**, 0–100: the chance in % that a position opened now hits your stop
  within the window. The stop is a loss of margin (the PnL % CoinDCX shows, −5% to −10%), so it
  depends on the coin's leverage: megacaps (BTC, ETH, SOL, BNB, XRP) 20×, shitcoins (meme coins, or
  under $50M traded in 24h) 3×, other altcoins 10×, changeable per coin.
- the **bad case**: the 1-in-10 adverse move within the window, as a price move and as a share of margin,
  the **liquidation chance**, and the main reason behind the number;
- **momentum** (bullish / bearish / neutral, −100..+100) and the **market state** (trending / choppy /
  quiet / wild): rules describing what has happened, not forecasts;
- **volatility** (calm / normal / high / wild) and the **funding rate** (crowded longs or shorts).

The `deltarisk risk` command prints the models' walk-forward check and today's readings for a coin.
The `deltarisk backtest` command replays two rule-based setups (fair-value-gap setups and wedge
breakouts) on history. They are not on the dashboard.

> It only reads public market data: there are no exchange keys and no code that places orders.

---

## Architecture Overview

```
[ Binance USD-M Futures Public REST (Read-Only) ]
  (closed 5m / 15m / 1h candles of the coin and its benchmark, the coin's
   settled funding, live price; candles cached on disk)
               │
               ▼
   [ Alignment ]   coin and benchmark on one candle grid per chart
               │
               ▼
   [ regime ]    momentum, efficiency, volatility percentile, state      (per window)
   [ risk ]      HAR volatility forecast and walk-forward check;
                 trained on load and daily, scored on each close        (per window, side)
   [ funding ]   funding rate and its percentile
               │
               ▼
┌────────────────────────────────────────────────────────────────────────┐
│      READ-ONLY WEB DASHBOARD (Go, REST + WebSocket, port 8000)         │
│                                                                        │
│  • Strictly NO Private Keys / NO Trading APIs                          │
│  • Rescored whenever a 5m, 15m or 1h candle closes                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Risk readings

Windows 5m–1h use 5-minute candles (90 days of history), 2h–4h 15-minute candles (180 days), 8h hourly
candles (2 years). Everything is computed on closed candles, in `internal/risk` and
`internal/regime`.

- **Adverse move:** for a long, how far the lowest low of the next window fell below the entry close;
  for a short, how far the highest high rose above it. Wicks count.
- **Swing unit:** a HAR volatility forecast (a regression of the next window's high–low volatility on
  the last 1, 4, 24 and 96 windows' volatility). Adverse moves are modelled in units of this forecast,
  so thousands of past windows, calm or wild, say something about today's 0.25% or 3% move.
- **Model:** the forecast times the past distribution of adverse moves in swing units. (LightGBM
  quantile regressions on ~20 inputs ran next to it until October 2026; they were dropped because they
  beat it in only 9 of 96 window-sides, by a few percent.) Nothing predicts direction: earlier tests
  found the next move's direction indistinguishable from chance, while its size is predictable because
  volatility clusters.
- **Check:** walk-forward, training on the first 60% of history and testing on four later blocks
  with an 8h gap. The 1-in-10 bad case should be exceeded in about 10% of test windows; outside 7–13%
  the page shows "—" instead of a number. Models are retrained once a day.

```bash
go run ./cmd/deltarisk risk B-SOL_USDT                  # every window, −5% stop, the tier's leverage
go run ./cmd/deltarisk risk B-ETH_USDT --window 1h --stop 8 --leverage 10 --calibration
```

Results on 17 October 2026 (ETH, SOL, DASH, ZK, BTC; −5% stop at the tier's leverage): the baseline's
bad case was exceeded in 7–13% of test windows in 77 of 80 window-sides, most of them within 9–11%.
ZK's shorts at 2h–4h failed (13–15%: it pumped harder than its history said). The predicted stop-hit chance matched what happened within about 2 points.
At 20× with a −5% margin stop (a 0.25% price move), a BTC, ETH or SOL long or short hits the stop
within 1h 35–58% of the time.

---

## Setups

The rules run on a pair of charts: 15-minute entries in 1-hour context by default (`setups.Spec`).

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
go run ./cmd/deltarisk backtest B-ETH_USDT                             # 15m / 1h, 2 years, A+, 3R, 1% risk
go run ./cmd/deltarisk backtest B-BTC_USDT --entry 5m --context 1h --days 365 --target 4 --csv trades.csv
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
go run ./cmd/deltarisk backtest B-ETH_USDT --strategy wedge            # 1h, 2 years, A+, 2R, 1% risk, at most 5x
go run ./cmd/deltarisk backtest B-SOL_USDT --strategy wedge --entry 4h --days 2190 --target measured --grade all
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

The backend is a Go program and the page a Svelte + Tailwind app in `web/`. [mise](https://mise.jdx.dev)
installs both toolchains (Go 1.25 and Node 24, pinned in `mise.toml`) and runs the tasks:

```bash
mise start
```

That builds the page into `web/dist` and the binary into `bin/deltarisk`, then starts the dashboard on
port 8000 on every network interface: open **http://127.0.0.1:8000** on the same machine, or the
machine's IP from elsewhere. The dashboard has no login, so anyone who can reach the port can add and
remove coins; firewall the port or pass `--host 127.0.0.1` to keep it local. After a build, the binary
runs on its own:

```bash
bin/deltarisk                        # same as mise start, without the build
bin/deltarisk --port 8002            # or: PORT=8002 mise start
bin/deltarisk --host 127.0.0.1       # this machine only
bin/deltarisk --offline              # simulated random-walk candles, no network
bin/deltarisk --web-dist path/dist   # serve another build of the page
```

`deltarisk web` is still accepted and does the same thing; `go run ./cmd/deltarisk` works without a build.
While working on the page, run both halves with hot reload:

```bash
mise dev            # page on http://localhost:5000, backend on 127.0.0.1:5001
mise tasks          # build, check, test, dev:api, dev:web, start, install
```

In the dashboard, add any CoinDCX futures pair (e.g. `B-DOGE_USDT`, `B-SOL_USDT`). A new coin's
history downloads in the background and its models train (about a second a coin); the next coin
downloads while the previous one trains (the row shows the progress). After a restart, cached candles
make every coin ready in seconds; from then on each candle is appended as it closes and the readings are
rescored. Pick the window and your stop at the top; open a row for every window and to change the
coin's tier or leverage. All market data comes from the Binance perpetual
behind the pair (`B-SOL_USDT` → `SOLUSDT`, `B-1000PEPE_USDT` → `1000PEPEUSDT`, INR pairs → their
`USDT` perpetual, priced in USDT). The add-pair drawer lists every Binance crypto and commodity
perpetual under its CoinDCX name, like CoinDCX's futures list: last price, change over 5m, 15m, 1h, 4h or
24h and 24h volume, sortable, searchable and filtered by sector. CoinDCX itself is never called, so
prices are the Binance perpetual's (no INR column or CoinDCX leverage badge).

Everything the app keeps lives in `~/.deltarisk` (set `DELTARISK_HOME` to use another folder):

| Path | Contents |
|---|---|
| `tracked_symbols.json` | Coins on the dashboard, with any tier or leverage you set |
| `cache/candles/` | Downloaded closed candles, so restarts and backtests only fetch new ones |
| `cache/funding/` | Downloaded funding rates, likewise |
| `config/assets.yaml` | Optional overrides of the defaults shown in this repo's `config/` |

Older versions also kept `models/`, `logs/forecasts/` and `cache/risk/` there; nothing reads them any more, so
they can be deleted.

---

### Sector Benchmarks

Each coin is downloaded with its sector's leader, whose volatility is one of the risk models' inputs.
A coin is never its own benchmark.

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
mise run test       # go test ./...
mise run check      # svelte-check and go vet
```
