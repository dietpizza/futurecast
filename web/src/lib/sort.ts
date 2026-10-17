// Sorting the watchlist by a column. A click on a sortable head cycles its first direction, the other
// direction, then off (the order the coins were added in). Coins without a value sort last either way.

import { callOf, reading, windowOf } from "./risk";
import type { Coin, WindowName } from "./types";

export type SortKey = "pair" | "call" | "tier" | "long" | "short" | "momentum" | "volatility" | "funding";
export type SortDir = "asc" | "desc";
export interface Sort {
  key: SortKey;
  dir: SortDir;
}

const CALL_ORDER = { long: 0, short: 1, wait: 2 } as const;

// What each column sorts by, and which direction its first click goes: text A→Z, numbers highest first
const KEYS: Record<SortKey, { first: SortDir; value: (coin: Coin, name: WindowName, stopPct: number) => number | string | null }> = {
  pair: { first: "asc", value: (c) => c.symbol.replace(/^[BI]-/, "") },
  call: {
    first: "asc",
    value: (c, name, stop) => {
      const w = windowOf(c, name);
      return w ? CALL_ORDER[callOf(c, w, name, stop).call] : null;
    },
  },
  tier: { first: "desc", value: (c) => c.leverage ?? null },
  long: { first: "desc", value: (c, name, stop) => riskValue(c, name, stop, "long") },
  short: { first: "desc", value: (c, name, stop) => riskValue(c, name, stop, "short") },
  momentum: { first: "desc", value: (c, name) => windowOf(c, name)?.momentum ?? null },
  volatility: { first: "desc", value: (c, name) => windowOf(c, name)?.vol_pct ?? null },
  funding: { first: "desc", value: (c) => c.funding?.rate_pct ?? null },
};

export const SORT_KEYS = Object.keys(KEYS) as SortKey[];

function riskValue(coin: Coin, name: WindowName, stopPct: number, side: "long" | "short"): number | null {
  const w = windowOf(coin, name);
  if (!w) return null;
  const r = reading(coin, w, side, stopPct);
  return r.checked ? r.chance : null; // a reading that failed its check has no number to sort by
}

// The next sort after a click on `key`
export function nextSort(current: Sort | null, key: SortKey): Sort | null {
  const first = KEYS[key].first;
  if (!current || current.key !== key) return { key, dir: first };
  if (current.dir === first) return { key, dir: first === "asc" ? "desc" : "asc" };
  return null;
}

export function sortCoins(coins: [string, Coin][], sort: Sort | null, name: WindowName, stopPct: number): [string, Coin][] {
  if (!sort) return coins;
  const value = KEYS[sort.key].value;
  const sign = sort.dir === "asc" ? 1 : -1;
  const keyed = coins.map((entry, i) => ({ entry, i, v: value(entry[1], name, stopPct) }));
  keyed.sort((a, b) => {
    const missingA = a.v === null || (typeof a.v === "number" && !Number.isFinite(a.v));
    const missingB = b.v === null || (typeof b.v === "number" && !Number.isFinite(b.v));
    if (missingA || missingB) return missingA === missingB ? a.i - b.i : missingA ? 1 : -1;
    const cmp = typeof a.v === "string" ? a.v.localeCompare(b.v as string) : (a.v as number) - (b.v as number);
    return cmp ? sign * cmp : a.i - b.i;
  });
  return keyed.map((k) => k.entry);
}

export function isSort(v: unknown): v is Sort {
  const s = v as Sort;
  return !!s && SORT_KEYS.includes(s.key) && (s.dir === "asc" || s.dir === "desc");
}
