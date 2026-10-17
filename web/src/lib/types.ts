// The parts of the server's coin snapshot the page reads (SymbolWorker.snapshot in multi_manager.py).
// Statistics are null or missing when the history can't support them.

export type Horizon = "1h" | "4h" | "1d" | "3d";

export interface After {
  bucket: number;
  buckets: number;
  z_low: number | null;
  z_high: number | null;
  n: number;
  windows: number;
  mean_pct: number;
  median_pct: number;
  hit_rate: number;
  ci_low_pct: number;
  ci_high_pct: number;
  older_mean_pct: number;
  newer_mean_pct: number;
  all_mean_pct: number;
  all_hit_rate: number;
}

export interface Persistence {
  character?: "trending" | "mean_reverting" | "random" | null;
  vr?: number;
  vr_z?: number;
  vr_p?: number;
  ic?: number;
  ic_t?: number;
  ic_p?: number;
}

export interface HorizonStats {
  candles: number;
  interval: string;
  as_of?: string;
  return_pct: number;
  z: number | null;
  percentile: number;
  typical_move_pct: number;
  history_candles?: number;
  after?: After | null;
  persistence?: Persistence | null;
  signal?: "LONG" | "SHORT" | "NEUTRAL";
  reason?: "flat" | "few" | "noise" | "fees" | "unstable" | "edge";
  kind?: "continuation" | "reversal" | null;
  flow?: { buy_pct: number; percentile: number } | null;
  versus?: { symbol: string; beta: number; return_pct: number; excess_pct: number } | null;
}

export interface Momentum {
  vol_days?: number;
  round_trip_pct?: number;
  min_windows?: number;
  horizons?: Partial<Record<Horizon, HorizonStats | null>>;
}

export interface Signals {
  rsi?: Record<string, number | null>;
  ema?: { interval: string; stack: "up" | "down" | "mixed"; vs_slowest_pct: number; ema: Record<string, number> } | null;
  ranges?: Record<string, { position: number; low: number; high: number } | null>;
  volatility?: { daily_pct: number; percentile: number } | null;
  volume?: { last_24h_usd: number; typical_24h_usd: number; ratio: number } | null;
  funding?: { rate_pct: number; percentile: number; days: number; mean_pct: number } | null;
}

export interface OrderBook {
  available?: boolean;
  bid_pct?: number;
  ask_pct?: number;
  bid_depth_usd?: number;
  ask_depth_usd?: number;
  book_imbalance_pct?: number;
  near_imbalance_pct?: number;
  spread_bps?: number;
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
  status?: "pending" | "queued" | "downloading" | "ready" | "failed";
  progress?: Progress | null;
  error?: string | null;
  history?: Record<string, { candles: number; from: string | null }>;
  current_price?: number | null;
  momentum?: Momentum | null;
  signals?: Signals | null;
  orderbook_depth?: OrderBook | null;
}

export interface CatalogItem {
  symbol: string;
  base: string;
  name: string;
  sector: string;
  description: string;
  coindcx_url: string;
}
