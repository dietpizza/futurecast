// Page state, the REST calls and the /ws stream. Components read `app` and re-render on their own;
// state that must survive a stream update (open rows, the window, the stop) lives here, not in a component.

import { STOPS, WINDOWS } from "./risk";
import { isSort, nextSort, type Sort, type SortKey } from "./sort";
import type { CatalogItem, Coin, Tier, WindowName } from "./types";

type Connection = "connecting" | "live" | "reconnecting";

// The window and the stop are remembered per browser; storage may be unavailable (private windows)
function stored<T>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    const value = raw === null ? fallback : (JSON.parse(raw) as T);
    return allowed.includes(value) ? value : fallback;
  } catch {
    return fallback;
  }
}

export function remember(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // not saved; the page still works
  }
}

function storedSort(): Sort | null {
  try {
    const raw = localStorage.getItem("futurecast.sort");
    const value = raw === null ? null : JSON.parse(raw);
    return isSort(value) ? value : null;
  } catch {
    return null;
  }
}

export const app = $state({
  coins: {} as Record<string, Coin>,
  window: stored<WindowName>("futurecast.window", WINDOWS, "1h"),
  stopPct: stored<number>("futurecast.stop", STOPS, 5),
  sort: storedSort(),
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

// Stops the stream so state edited from the console stays put (for previewing risk states)
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

// Tier and leverage overrides; null goes back to the automatic one
export async function updateCoin(symbol: string, settings: { tier?: Tier | null; leverage?: number | null }): Promise<void> {
  try {
    const resp = await fetch(`/api/coins/${encodeURIComponent(symbol)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    });
    if (resp.ok) {
      const coin: Coin = await resp.json();
      app.coins[coin.symbol] = Object.assign(app.coins[coin.symbol] || {}, coin);
    } else {
      const err = await resp.json();
      alert(err.detail || "Could not change the coin's settings");
    }
  } catch (err) {
    console.error("Failed to update coin:", err);
  }
}

export function selectWindow(name: WindowName): void {
  app.window = name;
  remember("futurecast.window", name);
}

// A click on a sortable column head: its first direction, the other, then the added order
export function sortBy(key: SortKey): void {
  app.sort = nextSort(app.sort, key);
  remember("futurecast.sort", app.sort);
}

export function selectStop(stop: number): void {
  app.stopPct = stop;
  remember("futurecast.stop", stop);
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
