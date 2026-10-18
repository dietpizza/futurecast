# Plan: intraday risk scores, momentum and market state

> **2026-10-19:** LightGBM was removed with the Go rewrite (`GO-REWRITE-PLAN.md`): every window and side now uses the HAR
> baseline. The LightGBM parts below are history.

Agreed with the user on 2026-10-17 in chat and implemented the same day (all six steps). The user's answers: the stop is on margin; the tier rule as proposed; the risk score is the stop-hit chance (see below). This replaces the momentum statistics
dashboard (event study, variance ratio, "next after similar") with a risk view for intraday trading of
crypto futures. It brings a model back on purpose: the user asked for predictive risk analysis and for a
decision on XGBoost / LightGBM / CatBoost (answer below). Read the repo's `CLAUDE.md` first; this file only
says what changes.

## The user's goal

"If I open a long or a short on this coin right now, how badly can it hurt me before I close it?"
They trade intraday, one day at most, with stop losses, and want a few clear readings per coin instead of
a card full of statistics:

1. **Long risk** and **short risk**, 0–100 each.
2. **Momentum**: bullish / bearish / neutral, with a strength.
3. **Market state**: trending / choppy / quiet / wild.
4. Two extras: **volatility** (calm / normal / high / wild) and **funding** (crowding).

## Decisions made

| Topic | Decision |
|---|---|
| Windows | **5m, 15m, 30m, 1h, 2h, 3h, 4h, 8h**, picked with a selector (default 1h) |
| Leverage | By tier: **megacap 20×, altcoin 10×, shitcoin 3×**, shown on the row, changeable per coin |
| Stop | Default **−5%**, adjustable to −10%, one page-wide setting. See open question 1 for what it's measured on |
| "Bad case" | The 1-in-10 adverse move within the window (90th percentile of the move against the position) |
| Risk score scale | In the trader's money (depends on leverage and stop), not relative to the coin's own past |
| Old statistics | Removed: `stats/momentum.py` event study, variance ratio, IC, `momentum_signal`, taker-flow and order-book columns, and their page code. Taker flow and funding become model inputs |
| Backtest command | Unchanged (`deltarisk backtest`, fair-value gaps and wedges) |
| Watchlist size | Small (a handful to ~20 coins); one model per coin is fine |

## Open questions to settle first (ask the user)

1. **Is the stop measured on margin (the PnL % CoinDCX shows) or on price?** The user said "−5 to −10% of
   the profits", which reads as margin. **Assume margin** unless they say otherwise:

   | Tier | Leverage | Price move for a −5% margin stop | for −10% |
   |---|---|---|---|
   | Megacap | 20× | 0.25% | 0.5% |
   | Altcoin | 10× | 0.5% | 1% |
   | Shitcoin | 3× | 1.67% | 3.33% |

   A −5% *price* stop at 20× can't work: isolated liquidation comes first at about −4.5%.
2. **Tier rule** (proposed, not yet confirmed):
   - Megacap: fixed list BTC, ETH, SOL, BNB, XRP.
   - Shitcoin: sector `MEME` (`ingestion/factory.get_symbol_sector`), or Binance 24h quote volume under
     $50M (`/fapi/v1/ticker/24hr`, public).
   - Altcoin: everything else.
   - Per-coin override stored with the watchlist.

## What each number means

For a coin, window `h`, side (long/short), leverage `L` and stop `s` (as a margin fraction, so the stop is
a price move of `d = s / L`):

- **Adverse move** over the window: for a long, `(entry − lowest low within the next h) / entry`; for a
  short, `(highest high − entry) / entry`. It's measured with candle highs and lows, so the dip inside the
  window counts, not just the close. Entry is the close of the candle the reading is made on.
- **Bad case**: the predicted 90th percentile of the adverse move, shown as price % and margin %
  (× `L`).
- **Stop-hit chance**: predicted P(adverse move ≥ `d`) within the window.
- **Liquidation chance**: P(adverse move ≥ liquidation distance) with the isolated-margin approximation
  `liq ≈ 1/L − mmr`. Binance's maintenance-margin brackets need an authenticated endpoint, so use fixed
  rates: 0.4% megacap, 0.5% altcoin, 1% shitcoin. With margin stops this is tiny; it matters only when a
  stop slips in a fast move.
- **Risk score** = the stop-hit chance in percent (changed on 2026-10-17 after the walk-forward report: the
  first formula, `min(100, round(100 × bad case / d))`, read 100 for megacaps at 20× with a −5% stop in every
  window from 1h up). The tooltip gives the stop-hit chance, the bad case in price % and margin %, and the main
  reason (e.g. "volatility 2× normal, funding crowded long").

## Data

New chart: **5m candles, 90 days (~25,920)**, cached like the others (`ingestion/cache.py`). Existing:
15m (180 days) and 1h (2 years). In `publisher/multi_manager.py`, extend `CHART_MINUTES` and
`HISTORY_BARS`; funding already rides on 1h (`FUNDING_CHARTS`). Offline mode (`load_synthetic`) needs 5m
too.

| Windows | Chart | Past windows that don't overlap |
|---|---|---|
| 5m, 15m, 30m, 1h | 5m | 25,920 → ~2,160 |
| 2h, 3h, 4h | 15m | ~2,160 / 1,440 / 1,080 |
| 8h | 1h | ~2,190 |

The stream loop must also refresh on every 5m close (today it's 15m and 1h).

Check that the benchmark alignment (`ingestion/storage.align_cross_asset_data`) still makes sense. The
benchmark was only used by the beta-adjusted "versus" number, which goes away; drop the benchmark download
unless a model input uses it (e.g. BTC's own volatility as an input for alts, worth trying).

## Momentum and market state (rules, no model)

New module, e.g. `stats/regime.py`, per window, on that window's chart:

- **Momentum** −100..+100: a blend of the return over the last few windows in units of typical move
  (z, clipped) and the slope of an EMA matching the window. Bullish above +25, bearish below −25, else
  neutral. Tune the thresholds on real data so neutral isn't the answer 90% of the time. It describes what
  has happened; label it so on the page, not as a forecast.
- **Efficiency ratio** (Kaufman): |net move| / sum of |candle moves| over the last few windows.
- **Volatility level**: realized vol over the last few windows against the coin's own history for that
  window (percentile).
- **State**: wild if volatility percentile ≥ 90; else trending if efficiency ≥ ~0.35 and |momentum| ≥ 25;
  else quiet if volatility percentile ≤ 25; else choppy. Thresholds go in a spec dataclass and get
  checked on real coins.

## The risk model

Per coin, per window, per side. Train when history loads, retrain once a day, score on every candle close
of that window's chart.

**Inputs** (all known at the reading's close, no look-ahead):
- realized volatility at several lengths (HAR: last window, last 4, last 24, last ~96, in that chart's
  candles), using both close-to-close and high–low ranges (Parkinson);
- momentum, efficiency ratio and volatility percentile from the regime module;
- funding rate and its percentile (1h chart, forward-filled; `ingestion/binance.funding_known_at` already
  avoids look-ahead);
- taker-buy share over the last window(s);
- position in the recent range;
- time of day (UTC hour) and hours to the next funding settlement;
- optional: BTC's volatility for non-BTC coins.

**Targets**: the long and short adverse move over the next `h` (definition above), divided by the HAR
volatility forecast made at that time ("adverse move in swing units").

**Models, in order. Keep the simplest that passes the check:**
1. **Baseline**: HAR volatility forecast (linear regression on log realized vol) × the empirical
   distribution of past adverse moves in swing units. The stop-hit chance is the share of past windows
   where the adverse move in swing units exceeded `d / forecast`. Using swing units is what makes rare
   0.25–3% moves estimable: there are thousands of examples at every scale.
2. **LightGBM quantile regression** (objective `quantile`, alpha 0.5 / 0.9 / 0.97) on the same inputs,
   predicting the adverse move in swing units, separately for long and short. This captures long/short
   asymmetry (skew, crowded funding, trend). Stop-hit chance comes from interpolating the predicted
   quantiles at `d`.

**Why LightGBM and not XGBoost or CatBoost:** native quantile objective, fastest to retrain per coin,
handles ~20 numeric inputs well. CatBoost's advantage is categorical inputs, which we don't have; XGBoost
would be similar but slower. Use one. **Don't** build a direction classifier: earlier tests on real
BTC/ETH/SOL/BAT/INJ/DASH history (Oct 2026) found the next move's direction indistinguishable from chance.
Risk comes from the size of moves, which is predictable because volatility clusters.

**Dependency**: add `lightgbm` (and `scikit-learn` only if it's actually needed) with `uv add`.

## The honesty check (gate for showing a number)

Walk-forward: train on the first 60% of history, test on the next block, roll forward. Use an 8h-window
gap between train and test so overlapping targets don't leak. Report per coin, window and side:

- **Coverage**: the share of test windows where the real adverse move exceeded the predicted 90% bad case.
  It should be about 10%; flag outside 7–13%.
- **Calibration of stop-hit chance**: bucket predictions (0–10%, 10–20%, …) against what actually
  happened.
- **Pinball loss**: LightGBM against the HAR baseline. LightGBM ships for that coin and window only if it
  beats the baseline by a margin (e.g. ≥ 2% lower loss) on the test blocks; otherwise the baseline ships.

A coin and window that fails coverage shows "—" with a tooltip saying why, instead of a number.

CLI: `uv run deltarisk risk B-SOL_USDT [--window 1h] [--stop 5] [--leverage 20]` prints the check per
window and side, and today's readings. **Build this and show the user real numbers before any UI work.**

## Backend changes

- `stats/regime.py` (new): momentum, efficiency, volatility level, state.
- `stats/risk.py` (new): inputs, targets, HAR baseline, LightGBM, walk-forward check, scoring.
- `stats/signals.py`: keep only what the new page shows (funding, volatility); delete the rest.
- `stats/momentum.py`: delete (move `rolling_vol` or `percentile_of` if still needed). Delete the momentum
  tests with it.
- `publisher/multi_manager.py`: 5m chart; models trained in the existing background load (status shows
  "training" after "downloading"); daily retrain; per-window scoring on candle close. Snapshot fields: for
  each window, `momentum`, `state`, `volatility`, and for `long` / `short`: `bad_case_pct`,
  `stop_hit_chance`, `liquidation_chance`, `score`, `reason`, `model` ("har" or "lgbm"), `checked` (passed
  the check). Plus `tier`, `leverage`, `funding`. Scores depend on the stop: either send the quantiles and
  let the page compute the score for the chosen stop, or take the stop as a query parameter. Prefer the
  first, so the stop slider needs no server round-trip.
- Tier, leverage and override: stored with the watchlist (`tracked_symbols.json` grows from a list to
  objects; read the old format too). Endpoint `PATCH /api/coins/{symbol}` for tier and leverage.
- `cli.py`: `risk` command.
- Offline mode keeps working (random-walk candles; the check will mostly show "—", which is fine).

## Frontend changes (`web/`, Svelte + Tailwind)

- Top bar: window selector (5m … 8h), stop setting (−5% … −10% margin, saved in `localStorage`).
- One compact row per coin: pair · price · tier and leverage · momentum (label + bar −100..+100) · state ·
  **long risk** (score, bar, stop-hit %) · **short risk** · volatility · funding.
- Clicking a row opens a small table of all 8 windows for that coin (same columns), plus the tier and
  leverage override and Remove. No big card.
- Tooltips: the reason line, bad case in price % and margin %, liquidation chance, model used.
- Remove `CoinDetails` content, the event-study wording in `lib/momentum.ts`, and the old glossary
  entries; rewrite the glossary for the new readings in plain words.
- Keep the Cyberpunk look, tokens and both themes (`src/app.css`). Risk colors: low green, middle warn,
  high red, as new tokens defined in both themes.

## Build order

1. 5m chart and inputs (backend only) + tests.
2. Regime rules + tests; check the label distribution on real coins.
3. Risk model, walk-forward check, `deltarisk risk` CLI. **Stop here and show the user the report** on
   their coins (copy the user's candle cache into a temp `DELTARISK_HOME`, see `CLAUDE.md`).
4. Live scoring, snapshot and API, tiers and leverage.
5. New page.
6. Delete the old statistics, update `CLAUDE.md` and `README.md`, and update the memory note that said
   models were removed.

## Tests to add

- Adverse move uses highs and lows and never looks past the window; the last `h` candles have no target.
- No input uses data after the reading's close (shift test: changing future candles changes no input).
- Walk-forward split leaves a gap; coverage on a synthetic random walk with known volatility is ~10%.
- Score formula, stop as margin vs price, leverage by tier, liquidation distance.
- Regime labels on hand-made series (straight trend → trending, zigzag → choppy, flat → quiet, spike →
  wild).
- API: snapshot fields, tier override round-trip, old watchlist format still loads.
