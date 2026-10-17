// Page state, the REST calls and the /ws stream. Components read `app` and re-render on their own;
// state that must survive a stream update (open cards, the horizon) lives here, not in a component.

import type { CatalogItem, Coin, Horizon } from "./types";

type Connection = "connecting" | "live" | "reconnecting";

export const app = $state({
  coins: {} as Record<string, Coin>,
  timeframe: "4h" as Horizon,
  expanded: {} as Record<string, boolean>,
  connection: "connecting" as Connection,
  catalog: [] as CatalogItem[],
  sidebarOpen: false,
});

let ws: WebSocket | null = null;
let live = true;

function connect(): void {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
  ws.onopen = () => (app.connection = "live");
  ws.onmessage = (event) => {
    try {
      handleMessage(JSON.parse(event.data));
    } catch (err) {
      console.error("WS parse error:", err);
    }
  };
  ws.onclose = () => {
    if (!live) return;
    app.connection = "reconnecting";
    setTimeout(connect, 2000);
  };
  ws.onerror = (err) => console.error("WS error:", err);
}

function handleMessage(msg: { type: string; symbol?: string; coins?: Coin[] } & Partial<Coin>): void {
  if (msg.type === "INIT_STATE") {
    for (const coin of msg.coins || []) app.coins[coin.symbol] = coin;
  } else if (msg.type === "FORECAST_UPDATE" && msg.symbol) {
    const { type: _, ...coin } = msg;
    app.coins[msg.symbol] = Object.assign(app.coins[msg.symbol] || {}, coin) as Coin;
  } else if (msg.type === "COIN_REMOVED" && msg.symbol) {
    forget(msg.symbol);
  }
}

function forget(symbol: string): void {
  delete app.coins[symbol];
  delete app.expanded[symbol];
}

export async function start(): Promise<void> {
  loadCatalog();
  try {
    const resp = await fetch("/api/coins");
    if (resp.ok) for (const c of (await resp.json()) as Coin[]) app.coins[c.symbol] = c;
  } catch (err) {
    console.error("Initial load failed:", err);
  }
  connect();
}

// Stops the stream so state edited from the console stays put (for previewing signal states)
export function stopLive(): void {
  live = false;
  ws?.close();
  app.connection = "reconnecting";
}

async function loadCatalog(): Promise<void> {
  try {
    const resp = await fetch("/api/coindcx/futures");
    if (resp.ok) app.catalog = await resp.json();
  } catch (err) {
    console.error("Failed to load CoinDCX futures catalog:", err);
  }
}

export async function addCoin(symbol: string): Promise<void> {
  try {
    const resp = await fetch("/api/coins", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol }),
    });
    if (resp.ok) {
      const coin: Coin = await resp.json();
      app.coins[coin.symbol] = coin;
    } else {
      const err = await resp.json();
      alert(err.detail || "Failed to add coin");
    }
  } catch (err) {
    console.error("Failed to add coin:", err);
  }
}

export async function reloadCoin(symbol: string): Promise<void> {
  const coin = app.coins[symbol];
  if (!coin) return;
  coin.status = "queued";
  try {
    await fetch(`/api/coins/${encodeURIComponent(symbol)}/reload`, { method: "POST" });
  } catch (err) {
    console.error("Failed to reload coin:", err);
  }
}

export async function removeCoin(symbol: string): Promise<void> {
  try {
    await fetch(`/api/coins/${encodeURIComponent(symbol)}`, { method: "DELETE" });
    forget(symbol);
  } catch (err) {
    console.error("Failed to remove coin:", err);
  }
}

export function toggleExpanded(symbol: string): void {
  // A click that ends a text selection shouldn't fold the card away
  if (window.getSelection?.()?.toString()) return;
  app.expanded[symbol] = !app.expanded[symbol];
}
