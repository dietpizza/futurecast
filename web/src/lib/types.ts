// The parts of the server's coin snapshot the page reads (manager.Snapshot in engine/manager/worker.go).
// Readings are null or missing when the history can't support them.

export type WindowName = "5m" | "15m" | "30m" | "1h" | "2h" | "3h" | "4h" | "8h";
export type Side = "long" | "short";
export type Tier = "megacap" | "altcoin" | "shitcoin";

export interface SideRisk {
  checked: boolean; // passed the walk-forward check (bad case exceeded in 7–13% of test windows)
  coverage: number | null; // share of test windows worse than the bad case
  bad_case_pct: number; // 1-in-10 adverse move within the window, price %
  bad_case_margin_pct: number;
  stop_hit_chance: number; // at the server's stop (risk.stop_pct), 0–1
  liquidation_chance: number;
  score: number | null; // stop-hit chance in %
  reason: string;
  quantiles: { alphas: number[]; moves_pct: number[] }; // adverse move quantiles, price %
}

export interface TrendOutcome {
  n: number; // past readings with a known next window (they overlap)
  win_rate: number | null; // share where the next window closed the call's way
  mean_pct: number | null; // mean move the call's way over the next window, before fees
}

// What followed past trending readings on this coin (regime.ComputeTrendRecord), behind the Long / Short call
export interface TrendRecord {
  long: TrendOutcome;
  short: TrendOutcome;
  base: TrendOutcome; // every reading: the coin's plain drift
  round_trip_pct: number;
}

export interface WindowRisk {
  at: string;
  swing_pct: number;
  momentum: number | null; // −100..+100
  momentum_label: "bullish" | "bearish" | "neutral" | null;
  efficiency: number | null;
  vol_pct: number | null;
  volatility: "calm" | "normal" | "high" | "wild" | null;
  state: "trending" | "choppy" | "quiet" | "wild" | null;
  trend_record: TrendRecord | null;
  long: SideRisk;
  short: SideRisk;
}

export interface Risk {
  stop_pct: number;
  mmr: number;
  trained_at: string | null;
  windows: Partial<Record<WindowName, WindowRisk>>;
}

export interface Funding {
  rate_pct: number;
  mean_pct: number;
  percentile: number | null;
  days: number;
}

export interface Progress {
  label: string;
  step: number;
  steps: number;
  done: number;
  total: number;
  fraction: number | null;
}

export interface Coin {
  symbol: string;
  sector?: string;
  has_data?: boolean;
  status?: "pending" | "queued" | "downloading" | "training" | "ready" | "failed";
  progress?: Progress | null;
  error?: string | null;
  history?: Record<string, { candles: number; from: string | null }>;
  current_price?: number | null;
  tier?: Tier;
  tier_auto?: Tier;
  leverage?: number;
  overrides?: { tier: Tier | null; leverage: number | null };
  risk?: Risk | null;
  funding?: Funding | null;
}

export interface CatalogItem {
  symbol: string;
  base: string;
  name: string;
  sector: string;
  description: string;
  coindcx_url: string;
}

// The add-pair drawer's market board (GET /api/market, engine/board)
export type BoardWindow = "5m" | "15m" | "1h" | "4h" | "24h";

export interface MarketCoin {
  price: number; // in the pair's units
  volume_24h: number; // USDT
  change: Record<BoardWindow, number | null>; // %
}

export interface Market {
  windows: BoardWindow[];
  updated_at: number | null; // ms
  markets: number;
  backfilled: number; // markets whose 5m–4h history has been filled in
  coins: Record<string, MarketCoin>;
  offline?: boolean;
}
