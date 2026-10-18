// The risk readings in words and numbers: the stop-hit chance for the page's stop, the labels, and the
// loading states. The math mirrors engine/risk; keep the two in sync.

import { formatNumber, isNum, pct, pctile, sp } from "./format";
import type { Coin, Side, SideRisk, Tier, WindowName, WindowRisk } from "./types";

export const WINDOWS: WindowName[] = ["5m", "15m", "30m", "1h", "2h", "3h", "4h", "8h"];
export const STOPS = [5, 6, 7, 8, 9, 10]; // stop loss choices, % of margin
export const TIERS: { value: Tier; label: string; leverage: number }[] = [
  { value: "megacap", label: "Megacap", leverage: 20 },
  { value: "altcoin", label: "Altcoin", leverage: 10 },
  { value: "shitcoin", label: "Shitcoin", leverage: 3 },
];
export const SIDES: Side[] = ["long", "short"];

// Score bands: the stop-hit chance in %. Under 25 the stop is rarely reached within the window,
// from 50 it is reached more often than not.
const RISK_MID = 25;
const RISK_HIGH = 50;

export function tierLabel(tier?: Tier | null): string {
  return TIERS.find((t) => t.value === tier)?.label || "—";
}

export function windowOf(coin: Coin, name: WindowName): WindowRisk | null {
  return coin.risk?.windows?.[name] || null;
}

// P(adverse move ≥ x) from quantiles (ascending): linear from (0, 0) through them, an exponential
// tail through the last two beyond the last (risk.ExceedChance in engine/risk).
export function exceedChance(alphas: number[], moves: number[], x: number): number {
  const n = moves.length;
  if (!n) return NaN;
  if (x >= moves[n - 1]) {
    if (n < 2) return 1 - alphas[n - 1];
    const step = moves[n - 1] - moves[n - 2];
    const scale = step > 0 ? step / Math.log((1 - alphas[n - 2]) / (1 - alphas[n - 1])) : 1e-12;
    return (1 - alphas[n - 1]) * Math.exp(-(x - moves[n - 1]) / scale);
  }
  let lo = 0;
  let a0 = 0;
  for (let i = 0; i < n; i++) {
    if (x <= moves[i]) {
      const span = moves[i] - lo;
      const cdf = span > 0 ? a0 + ((alphas[i] - a0) * (x - lo)) / span : alphas[i];
      return 1 - cdf;
    }
    lo = moves[i];
    a0 = alphas[i];
  }
  return 1 - alphas[n - 1];
}

export interface Reading {
  side: Side;
  risk: SideRisk;
  stopMovePct: number; // the stop as a price move, %
  chance: number; // stop-hit chance for the page's stop
  score: number;
  checked: boolean;
}

// One side's reading for the page's stop. At the server's own stop its exact number is used.
export function reading(coin: Coin, w: WindowRisk, side: Side, stopPct: number): Reading {
  const risk = w[side];
  const leverage = coin.leverage || 1;
  const stopMovePct = stopPct / leverage;
  const chance = stopPct === coin.risk?.stop_pct
    ? risk.stop_hit_chance
    : exceedChance(risk.quantiles.alphas, risk.quantiles.moves_pct, stopMovePct);
  return { side, risk, stopMovePct, chance, score: Math.round(100 * Math.min(Math.max(chance, 0), 1)), checked: risk.checked };
}

export function riskTone(score: number): "low" | "mid" | "high" {
  return score >= RISK_HIGH ? "high" : score >= RISK_MID ? "mid" : "low";
}

export function riskText(coin: Coin, w: WindowRisk, r: Reading, name: WindowName): string {
  const lev = coin.leverage || 1;
  const side = r.side === "long" ? "A long" : "A short";
  const lines = [
    `${side} opened now hits a ${sp(-(r.stopMovePct * lev), 0)} margin stop (${formatNumber(r.stopMovePct, 2)}% price at ${lev}×) within ${name} ${pct(r.chance)} of the time.`,
    `Bad case (1 in 10): ${formatNumber(r.risk.bad_case_pct, 2)}% against you, ${formatNumber(r.risk.bad_case_margin_pct, 1)}% of margin.`,
    `Liquidation: ${chanceText(r.risk.liquidation_chance)}.`,
    `Why: ${r.risk.reason}.`,
  ];
  if (isNum(r.risk.coverage)) lines.push(`Check: bad case exceeded in ${pct(r.risk.coverage, 1)} of test windows.`);
  return lines.join("\n");
}

export function uncheckedText(r: Reading, name: WindowName): string {
  const cov = r.risk.coverage;
  return `The model failed its walk-forward check for ${name} ${r.side}s: its 1-in-10 bad case was exceeded in ${isNum(cov) ? pct(cov, 1) : "—"} of test windows instead of 7–13%, so no number is shown.`;
}

function chanceText(p: number): string {
  if (!isNum(p)) return "—";
  if (p < 0.0005) return "under 0.05%";
  return pct(p, p < 0.1 ? 1 : 0);
}

export const MOMENTUM_LABELS = { bullish: "Bullish", bearish: "Bearish", neutral: "Neutral" } as const;
export const STATE_LABELS = { trending: "Trending", choppy: "Choppy", quiet: "Quiet", wild: "Wild" } as const;
export const VOL_LABELS = { calm: "Calm", normal: "Normal", high: "High", wild: "Wild" } as const;

export function momentumText(name: WindowName): string {
  return `What price did over the last few ${name} windows, −100 to +100: the return and the slope of its moving average, in units of a typical move. Above +25 bullish, below −25 bearish. It describes the past; it is not a forecast.`;
}

export function stateText(w: WindowRisk, name: WindowName): string {
  const eff = isNum(w.efficiency) ? `${Math.round(w.efficiency * 100)}%` : "—";
  return `Wild when volatility is in the top 10% of the last 30 days; trending when price moved in a fairly straight line (efficiency ${eff}: net move over the distance travelled) with momentum; quiet when volatility is in the bottom quarter; else choppy. Over the last few ${name} windows.`;
}

export function volatilityText(w: WindowRisk, name: WindowName): string {
  return `Volatility of the last few ${name} windows against the last 30 days: ${pctile(w.vol_pct)} percentile. A typical ${name} swing now is about ${formatNumber(w.swing_pct, 2)}%.`;
}

// ───────── the call: Long / Short / Wait ─────────

// Long when momentum is bullish and the market is trending, and a long's risk at the page's stop is
// checked and under CALL_MAX_RISK; Short the mirror; else Wait. A rule, not a forecast: the tooltip
// shows how the next window went after past trending readings on this coin.
const CALL_MAX_RISK = 50;

export interface Call {
  call: "long" | "short" | "wait";
  label: string;
  why: string;
}

export function callOf(coin: Coin, w: WindowRisk, name: WindowName, stopPct: number): Call {
  const side: Side | null =
    w.state === "trending" && w.momentum_label === "bullish" ? "long"
    : w.state === "trending" && w.momentum_label === "bearish" ? "short"
    : null;
  if (!side) {
    const state = w.state ? STATE_LABELS[w.state].toLowerCase() : "unknown";
    const mom = w.momentum_label ? MOMENTUM_LABELS[w.momentum_label].toLowerCase() : "unknown";
    return { call: "wait", label: "Wait", why: `No trend to follow at ${name}: the market is ${state} and momentum ${mom}. A call needs a trending market with bullish or bearish momentum.` };
  }
  const r = reading(coin, w, side, stopPct);
  const title = side === "long" ? "Long" : "Short";
  const trend = `${w.momentum_label === "bullish" ? "Bullish" : "Bearish"} momentum in a trending market`;
  if (!r.checked) {
    return { call: "wait", label: "Wait", why: `${trend}, but the ${side} risk model failed its check at ${name}, so there is no risk number to trust.` };
  }
  if (r.score >= CALL_MAX_RISK) {
    return { call: "wait", label: "Wait", why: `${trend}, but a ${side} would hit your −${stopPct}% stop within ${name} ${r.score}% of the time (a call needs under ${CALL_MAX_RISK}%). A wider stop, lower leverage or a shorter window would bring it down.` };
  }
  return { call: side, label: title, why: `${trend}, and a ${side} hits your −${stopPct}% stop within ${name} ${r.score}% of the time.` };
}

// The call's track record on this coin, in words
export function callRecordText(w: WindowRisk, call: Call, name: WindowName): string {
  const rec = w.trend_record;
  const base = rec?.base;
  if (!rec || !base || !isNum(base.win_rate) || !isNum(base.mean_pct)) return "";
  const side: Side = call.call === "short" ? "short" : "long";
  const o = rec[side];
  const label = side === "long" ? "bullish" : "bearish";
  if (!o.n || !isNum(o.win_rate) || !isNum(o.mean_pct)) return `No trending ${label} readings at ${name} in this coin's history.`;
  const baseWay = side === "long" ? base.win_rate : 1 - base.win_rate;
  const baseMean = side === "long" ? base.mean_pct : -base.mean_pct;
  return `After past trending ${label} readings at ${name} on this coin (${o.n.toLocaleString()}, overlapping), the next ${name} went the ${side}'s way ${pct(o.win_rate)} of the time (any reading: ${pct(baseWay)}), ${sp(o.mean_pct, 3)} on average (any reading: ${sp(baseMean, 3)}) against ${formatNumber(rec.round_trip_pct, 3)}% round-trip fees. Measured on the same history, risk check left out: not a backtest.`;
}

// ───────── loading states ─────────

const BUSY_STATUSES = ["queued", "downloading", "training"];

export function isBusy(coin: Coin): boolean {
  return BUSY_STATUSES.includes(coin.status || "");
}

export function loadingStage(coin: Coin): string {
  if (coin.status === "queued") return "Queued";
  const p = coin.progress;
  const verb = coin.status === "training" ? "Training" : "Downloading";
  return p && p.fraction != null ? `${verb} ${Math.floor(p.fraction * 100)}%` : verb;
}

export function loadingDetail(coin: Coin): string {
  const p = coin.progress;
  if (coin.status === "queued") return "Queued (one coin at a time)";
  if (coin.status === "training") return p ? `${p.label} · ${p.step} of ${p.steps}` : "Training risk models…";
  if (!p) return "Downloading candles…";
  const counts = p.total ? ` · ${p.done.toLocaleString()} of ${p.total.toLocaleString()} new` : "";
  return `Downloading ${p.label}${counts} · ${p.step} of ${p.steps}`;
}

// What a cell shows instead of a number while there is none
export function waitingNote(coin: Coin): string {
  if (coin.status === "training") return "training…";
  if (isBusy(coin)) return "downloading…";
  if (coin.status === "failed") return "download failed";
  return coin.has_data ? "too little history" : "";
}
