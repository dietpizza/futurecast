// The server's momentum statistics (stats/momentum.py) put into words. Every number is shown as the
// server sent it: when the history is missing or too short the page says so instead of filling the gap.

import type { After, Coin, Horizon, HorizonStats } from "./types";
import { formatNumber, pct, pText, signed, sp } from "./format";

export const HORIZONS: Horizon[] = ["1h", "4h", "1d", "3d"];

const FIFTHS = ["bottom fifth", "second fifth", "middle fifth", "fourth fifth", "top fifth"];

export const CHARACTER = {
  trending: { label: "Trending", note: "moves tend to keep their direction within this horizon" },
  mean_reverting: { label: "Reverting", note: "moves tend to reverse within this horizon" },
  random: { label: "Random", note: "no significant trending or reversal: a random walk at 5%" },
};

const BUSY_STATUSES = ["queued", "downloading"];

export function isBusy(coin: Coin): boolean {
  return BUSY_STATUSES.includes(coin.status || "");
}

// "Downloading 34%": share of the download done, from the server's progress
export function loadingStage(coin: Coin): string {
  if (coin.status === "queued") return "Queued";
  const p = coin.progress;
  return p && p.fraction != null ? `Downloading ${Math.floor(p.fraction * 100)}%` : "Downloading";
}

export function loadingDetail(coin: Coin): string {
  const p = coin.progress;
  if (coin.status === "queued") return "Queued for download (one coin at a time)";
  if (!p) return "Downloading candles…";
  const counts = p.total ? ` · ${p.done.toLocaleString()} of ${p.total.toLocaleString()} new` : "";
  return `Downloading ${p.label}${counts} · ${p.step} of ${p.steps}`;
}

// What a cell shows instead of a number while there is none
export function waitingNote(coin: Coin): string {
  if (isBusy(coin)) return "downloading…";
  if (coin.status === "failed") return "download failed";
  return coin.has_data ? "too little history" : "";
}

export function horizonOf(coin: Coin, name: Horizon): HorizonStats | null {
  return coin.momentum?.horizons?.[name] || null;
}

export function hasMove(h: HorizonStats | null | undefined): h is HorizonStats & { z: number } {
  return Boolean(h && h.z != null && !isNaN(h.z));
}

export function significant(a: After | null | undefined): boolean {
  return Boolean(a && (a.ci_low_pct > 0 || a.ci_high_pct < 0));
}

export function rangeText(a: After): string {
  return `${sp(a.ci_low_pct)} to ${sp(a.ci_high_pct)}`;
}

function fifthText(a: After): string {
  return a.buckets === 5 ? FIFTHS[a.bucket - 1] : `quantile ${a.bucket} of ${a.buckets}`;
}

function zRange(a: After): string {
  if (a.z_low == null) return `z below ${signed(a.z_high, 1)}`;
  if (a.z_high == null) return `z above ${signed(a.z_low, 1)}`;
  return `z ${signed(a.z_low, 1)} to ${signed(a.z_high, 1)}`;
}

// The latest move in words, for tooltips
export function moveText(coin: Coin, h: HorizonStats, name: string): string {
  return `Last ${name}: ${sp(h.return_pct)} over ${h.candles} ${h.interval} candles. A typical ${name} move is ±${formatNumber(h.typical_move_pct, 2)}% (one standard deviation, from the last ${formatNumber(coin.momentum?.vol_days, 0)} days), so this is z ${signed(h.z, 2)}: above ${formatNumber(h.percentile, 0)}% of the ${(h.history_candles || 0).toLocaleString()} past ${name} moves on this chart.`;
}

// What followed moves like the latest one, in words
export function afterText(a: After, name: string): string {
  return `After past ${name} moves in the ${fifthText(a)} (${zRange(a)}), the next ${name} averaged ${sp(a.mean_pct, 3)} (median ${sp(a.median_pct, 3)}) and rose ${pct(a.hit_rate)} of the time; over every ${name} window: ${sp(a.all_mean_pct, 3)}, up ${pct(a.all_hit_rate)}. 95% range of the mean ${rangeText(a)}; ${a.n.toLocaleString()} windows, about ${Math.round(a.windows)} that don't overlap.`;
}

export function characterText(h: HorizonStats, name: string): string {
  const p = h.persistence || {};
  const c = p.character ? CHARACTER[p.character] : null;
  return [
    `Variance ratio ${formatNumber(p.vr, 2)} (z ${signed(p.vr_z, 1)}, ${pText(p.vr_p)}): ${c ? c.note : "not enough history"}. Above 1 is trending, below 1 reverting.`,
    `Last ${name} move against the next ${name}: correlation ${signed(p.ic, 3)} (t ${signed(p.ic_t, 1)}, ${pText(p.ic_p)}).`,
  ].join("\n");
}

export interface Verdict {
  state: "idle" | "go-long" | "go-short";
  label: string;
  pill: string;
  why: string;
}

// A horizon's verdict. Mirrors momentum_signal in stats/momentum.py: keep the reasons in sync.
export function signalVerdict(coin: Coin, name: Horizon): Verdict {
  const h = horizonOf(coin, name);
  const m = coin.momentum || {};
  const fees = `${formatNumber(m.round_trip_pct, 2)}%`;
  if (!hasMove(h)) {
    return { state: "idle", label: "No data", pill: "pill-neutral", why: `Not enough ${name} history on hand yet.` };
  }
  const a = h.after;
  if ((h.signal === "LONG" || h.signal === "SHORT") && a) {
    const up = h.signal === "LONG";
    const kind = h.kind === "reversal"
      ? `A reversal: after ${h.z > 0 ? "rises" : "falls"} like this, price tended to turn.`
      : "A continuation: moves like this tended to keep going.";
    return {
      state: up ? "go-long" : "go-short",
      label: up ? "Long" : "Short",
      pill: up ? "pill-bull" : "pill-bear",
      why: `After ${name} moves like this one, the next ${name} averaged ${sp(a.mean_pct)} (95% range ${rangeText(a)}) and rose ${pct(a.hit_rate)} of the time, against ${pct(a.all_hit_rate)} for every window. That clears the ${fees} round trip in fees, in both halves of the history (${sp(a.older_mean_pct)} older, ${sp(a.newer_mean_pct)} newer). ${kind}`,
    };
  }
  const reasons: Record<string, () => string> = {
    flat: () => `The last ${name} move (z ${signed(h.z, 1)}) is in the middle fifth of past ${name} moves: nothing to read from it.`,
    few: () => a
      ? `Only about ${Math.round(a.windows)} non-overlapping ${name} windows behind moves like this (needs ${m.min_windows}).`
      : `Too little ${name} history to compare with.`,
    noise: () => `After moves like this the next ${name} averaged ${sp(a!.mean_pct)}, but its 95% range ${rangeText(a!)} includes zero: no edge over chance.`,
    fees: () => `After moves like this the next ${name} averaged ${sp(a!.mean_pct)} (95% range ${rangeText(a!)}): real, but smaller than the ${fees} round trip in fees.`,
    unstable: () => `After moves like this the next ${name} averaged ${sp(a!.mean_pct)}, but ${sp(a!.older_mean_pct)} in the older half of the history and ${sp(a!.newer_mean_pct)} in the newer: it didn't hold.`,
  };
  const why = h.reason ? reasons[h.reason] : undefined;
  const label = ({ flat: "Flat", few: "Thin data" } as Record<string, string>)[h.reason || ""] || "No edge";
  return { state: "idle", label, pill: "pill-neutral", why: why ? why() : "" };
}
