/**
 * Main Web Terminal Application Logic.
 * Handles WebSocket streaming, state management, CoinDCX integration, and DOM rendering.
 *
 * Every number shown is the server's momentum statistics as-is (stats/momentum.py,
 * stats/signals.py): when the history is missing or too short the page says so instead of
 * filling the gap.
 */

// Global State
window.coinsData = {};
const coinsData = window.coinsData;
window.expandedCards = window.expandedCards || {};
const lastPrices = {};
let ws = null;

const SECTOR_LABELS = {
  MEME: "Meme",
  L1: "L1",
  DEFI: "DeFi",
  AI: "AI",
  NFT_GAMING: "NFT / Gaming",
  MAJOR: "Major",
  INFRA: "Infra",
  GENERAL_ALT: "Alt",
};
const HORIZONS = ["1h", "4h", "1d", "3d"];
const FIFTHS = ["bottom fifth", "second fifth", "middle fifth", "fourth fifth", "top fifth"];
const CHARACTER = {
  trending: { label: "Trending", note: "moves tend to keep their direction within this horizon" },
  mean_reverting: { label: "Reverting", note: "moves tend to reverse within this horizon" },
  random: { label: "Random", note: "no significant trending or reversal: a random walk at 5%" },
};
const ICON_CHEVRON = `<svg class="icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 6l4 4 4-4" /></svg>`;
const ICON_CLOSE = `<svg class="icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7" /></svg>`;
const ICON_EXTERNAL = `<svg class="icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3H3.5v9.5H13V10M9 3h4v4M13 3L7.5 8.5" /></svg>`;

window.toggleCardDetails = function(symbol) {
  const selection = window.getSelection && window.getSelection();
  if (selection && selection.toString()) return;
  window.expandedCards[symbol] = !window.expandedCards[symbol];
  renderCoinCard(symbol, false);
};

function normalizeCoinDCXSymbol(raw) {
  let s = (raw || "").trim();
  if (!s) return "";
  if (s.toLowerCase().includes("coindcx.com/")) {
    const parts = s.split("/");
    s = parts[parts.length - 1].split("?")[0].split("#")[0];
  }
  s = s.toUpperCase();
  if (s.startsWith("B-") || s.startsWith("I-")) {
    return s;
  }
  if (s.includes("_")) {
    return `B-${s.replace(/^B-|^I-/, "")}`;
  }
  if (s.endsWith("USDT") && s.length > 4) {
    return `B-${s.slice(0, -4)}_USDT`;
  }
  if (s.endsWith("INR") && s.length > 3) {
    return `I-${s.slice(0, -3)}_INR`;
  }
  return `B-${s}_USDT`;
}

// Universal Clock (UTC)
function updateClock() {
  const clockEl = document.getElementById("live-clock");
  if (clockEl) {
    const now = new Date();
    clockEl.textContent = now.toISOString().slice(11, 19) + " UTC";
  }
}
setInterval(updateClock, 1000);
updateClock();

// WebSocket Streaming Manager
function setConnectionState(state, label) {
  const statusEl = document.getElementById("ws-status");
  if (!statusEl) return;
  statusEl.dataset.state = state;
  const textEl = statusEl.querySelector(".status-text");
  if (textEl) textEl.textContent = label;
}

function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => setConnectionState("live", "Live");

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleWebSocketMessage(msg);
    } catch (err) {
      console.error("WS Parse error:", err);
    }
  };

  ws.onclose = () => {
    setConnectionState("reconnecting", "Reconnecting");
    setTimeout(initWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error("WS error:", err);
  };
}

function handleWebSocketMessage(msg) {
  if (msg.type === "INIT_STATE") {
    (msg.coins || []).forEach((coin) => {
      coinsData[coin.symbol] = coin;
    });
    renderAll();
  } else if (msg.type === "FORECAST_UPDATE") {
    const symbol = msg.symbol;
    if (!coinsData[symbol]) {
      coinsData[symbol] = { symbol: symbol };
    }
    Object.assign(coinsData[symbol], msg);
    renderCoinCard(symbol, true);
  } else if (msg.type === "COIN_REMOVED") {
    forgetCoin(msg.symbol);
  }
}

// REST APIs
async function quickAdd(symbol) {
  try {
    const resp = await fetch("/api/coins", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol: symbol }),
    });
    if (resp.ok) {
      const data = await resp.json();
      coinsData[data.symbol] = data;
      renderAll();
    } else {
      const err = await resp.json();
      alert(err.detail || "Failed to add coin");
    }
  } catch (err) {
    console.error("Failed to add coin:", err);
  }
}

async function reloadCoin(symbol) {
  if (!coinsData[symbol]) return;
  coinsData[symbol].status = "queued";
  renderCoinCard(symbol, false);
  try {
    await fetch(`/api/coins/${encodeURIComponent(symbol)}/reload`, { method: "POST" });
  } catch (err) {
    console.error("Failed to reload coin:", err);
  }
}

async function removeCoin(symbol) {
  try {
    await fetch(`/api/coins/${encodeURIComponent(symbol)}`, { method: "DELETE" });
    forgetCoin(symbol);
    renderAll();
  } catch (err) {
    console.error("Failed to remove coin:", err);
  }
}

function forgetCoin(symbol) {
  delete coinsData[symbol];
  delete lastPrices[symbol];
  delete window.expandedCards[symbol];
  const card = document.getElementById(cardIdFor(symbol));
  if (card) card.remove();
  updateSummary();
}

// Formatting Helpers
function esc(value) {
  return String(value === undefined || value === null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function cardIdFor(symbol) {
  return `card-${symbol.replace(/[^a-zA-Z0-9]/g, "_")}`;
}

function splitSymbol(symbol) {
  const core = (symbol || "").replace(/^[BI]-/, "");
  const [base, quote] = core.split("_");
  return { base: base || core, quote: quote || "" };
}

function sectorLabel(sector) {
  if (!sector) return "";
  return SECTOR_LABELS[sector] || sector.charAt(0) + sector.slice(1).toLowerCase().replace(/_/g, " ");
}

function isNum(v) {
  return v !== undefined && v !== null && !isNaN(v);
}

function formatNumber(num, decimals = 4) {
  if (!isNum(num)) return "—";
  return Number(num).toFixed(decimals);
}

function signed(num, decimals = 2) {
  if (!isNum(num)) return "—";
  const v = Number(num);
  const fixed = Math.abs(v).toFixed(decimals);
  if (Number(fixed) === 0) return fixed;
  return (v > 0 ? "+" : "−") + fixed;
}

// A signed percentage that is already in percent: 1.234 -> "+1.23%"
function sp(v, decimals = 2) {
  return isNum(v) ? `${signed(v, decimals)}%` : "—";
}

function pct(p, decimals = 0) {
  return isNum(p) ? `${(Number(p) * 100).toFixed(decimals)}%` : "—";
}

function ordinal(n) {
  if (!isNum(n)) return "—";
  const v = Math.round(n);
  const tail = v % 100 >= 11 && v % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[v % 10] || "th");
  return `${v}${tail}`;
}

// A percentile as an ordinal, with the ends said as "<1st" and ">99th" rather than "0th" and "100th"
function pctile(p) {
  if (!isNum(p)) return "—";
  if (p < 0.5) return "<1st";
  if (p >= 99.5) return ">99th";
  return ordinal(p);
}

function pText(p) {
  if (!isNum(p)) return "p —";
  return p < 0.001 ? "p < 0.001" : `p ${p < 0.01 ? p.toFixed(3) : p.toFixed(2)}`;
}

function formatPrice(p) {
  if (!p) return "—";
  const n = Number(p);
  if (n >= 100) return "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (n >= 1) return "$" + n.toFixed(4);
  return "$" + n.toLocaleString("en-US", { minimumSignificantDigits: 4, maximumSignificantDigits: 4 });
}

function formatUsdCompact(v) {
  if (!isNum(v) || !v) return "—";
  if (v >= 1e9) return "$" + (v / 1e9).toFixed(1) + "B";
  if (v >= 1e6) return "$" + (v / 1e6).toFixed(1) + "M";
  if (v >= 1e3) return "$" + Math.round(v / 1e3) + "k";
  return "$" + Math.round(v);
}

// Colour by sign, treating values that display as zero at `decimals` as flat
function toneClass(v, decimals = 2) {
  if (!isNum(v)) return "val-muted";
  if (Number(Math.abs(v).toFixed(decimals)) === 0) return "val-flat";
  return v > 0 ? "val-pos" : "val-neg";
}

function coindcxUrl(symbol) {
  return symbol.startsWith("B-") ? "https://coindcx.com/futures/" + symbol : "https://coindcx.com/trade/" + symbol;
}

function clockTime(iso) {
  return iso ? `${iso.slice(11, 16)} UTC` : "—";
}

// Clock time, with the date when it is not today (UTC)
function whenText(iso) {
  if (!iso) return "—";
  const today = new Date().toISOString().slice(0, 10);
  if (iso.slice(0, 10) === today) return clockTime(iso);
  return `${new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" })} ${clockTime(iso)}`;
}

function spanText(fromIso) {
  if (!fromIso) return "";
  const days = (Date.now() - new Date(fromIso).getTime()) / 86400000;
  return days >= 365 ? `${formatNumber(days / 365, 1)} years` : `${Math.round(days)} days`;
}

// ───────── History download ─────────

const BUSY_STATUSES = ["queued", "downloading"];

function isBusy(coin) {
  return BUSY_STATUSES.includes(coin.status);
}

// "Downloading 34%": share of the download done, from the server's progress
function loadingStage(coin) {
  if (coin.status === "queued") return "Queued";
  const p = coin.progress;
  return p && isNum(p.fraction) ? `Downloading ${Math.floor(p.fraction * 100)}%` : "Downloading";
}

function loadingDetail(coin) {
  const p = coin.progress;
  if (coin.status === "queued") return "Queued for download (one coin at a time)";
  if (!p) return "Downloading candles…";
  const counts = p.total ? ` · ${p.done.toLocaleString()} of ${p.total.toLocaleString()} new` : "";
  return `Downloading ${p.label}${counts} · ${p.step} of ${p.steps}`;
}

// ───────── Momentum helpers ─────────

function momentumOf(coin) {
  return coin.momentum || {};
}

function horizonOf(coin, name = currentTimeframe) {
  return (momentumOf(coin).horizons || {})[name] || null;
}

function hasMove(h) {
  return Boolean(h && isNum(h.z));
}

function fifthText(a) {
  return a.buckets === 5 ? FIFTHS[a.bucket - 1] : `quantile ${a.bucket} of ${a.buckets}`;
}

function zRange(a) {
  if (!isNum(a.z_low)) return `z below ${signed(a.z_high, 1)}`;
  if (!isNum(a.z_high)) return `z above ${signed(a.z_low, 1)}`;
  return `z ${signed(a.z_low, 1)} to ${signed(a.z_high, 1)}`;
}

function significant(a) {
  return Boolean(a && (a.ci_low_pct > 0 || a.ci_high_pct < 0));
}

function rangeText(a) {
  return `${sp(a.ci_low_pct)} to ${sp(a.ci_high_pct)}`;
}

// The latest move in words, for tooltips
function moveText(coin, h, name) {
  return `Last ${name}: ${sp(h.return_pct)} over ${h.candles} ${h.interval} candles. A typical ${name} move is ±${formatNumber(h.typical_move_pct, 2)}% (one standard deviation, from the last ${formatNumber(momentumOf(coin).vol_days, 0)} days), so this is z ${signed(h.z, 2)}: above ${formatNumber(h.percentile, 0)}% of the ${(h.history_candles || 0).toLocaleString()} past ${name} moves on this chart.`;
}

// What followed moves like the latest one, in words
function afterText(a, name) {
  return `After past ${name} moves in the ${fifthText(a)} (${zRange(a)}), the next ${name} averaged ${sp(a.mean_pct, 3)} (median ${sp(a.median_pct, 3)}) and rose ${pct(a.hit_rate)} of the time; over every ${name} window: ${sp(a.all_mean_pct, 3)}, up ${pct(a.all_hit_rate)}. 95% range of the mean ${rangeText(a)}; ${a.n.toLocaleString()} windows, about ${Math.round(a.windows)} that don't overlap.`;
}

// The selected horizon's verdict. Mirrors momentum_signal in stats/momentum.py.
function signalVerdict(coin, name = currentTimeframe) {
  const h = horizonOf(coin, name);
  const m = momentumOf(coin);
  const fees = `${formatNumber(m.round_trip_pct, 2)}%`;
  if (!hasMove(h)) {
    return { state: "idle", label: "No data", pill: "pill-neutral", why: `Not enough ${name} history on hand yet.` };
  }
  const a = h.after;
  if (h.signal === "LONG" || h.signal === "SHORT") {
    const up = h.signal === "LONG";
    const kind = h.kind === "reversal"
      ? `A reversal: after ${h.z > 0 ? "rises" : "falls"} like this, price tended to turn.`
      : "A continuation: moves like this tended to keep going.";
    return {
      state: up ? "go-long" : "go-short", label: up ? "Long" : "Short", pill: up ? "pill-bull" : "pill-bear",
      why: `After ${name} moves like this one, the next ${name} averaged ${sp(a.mean_pct)} (95% range ${rangeText(a)}) and rose ${pct(a.hit_rate)} of the time, against ${pct(a.all_hit_rate)} for every window. That clears the ${fees} round trip in fees, in both halves of the history (${sp(a.older_mean_pct)} older, ${sp(a.newer_mean_pct)} newer). ${kind}`,
    };
  }
  const why = {
    flat: () => `The last ${name} move (z ${signed(h.z, 1)}) is in the middle fifth of past ${name} moves: nothing to read from it.`,
    few: () => a
      ? `Only about ${Math.round(a.windows)} non-overlapping ${name} windows behind moves like this (needs ${m.min_windows}).`
      : `Too little ${name} history to compare with.`,
    noise: () => `After moves like this the next ${name} averaged ${sp(a.mean_pct)}, but its 95% range ${rangeText(a)} includes zero: no edge over chance.`,
    fees: () => `After moves like this the next ${name} averaged ${sp(a.mean_pct)} (95% range ${rangeText(a)}): real, but smaller than the ${fees} round trip in fees.`,
    unstable: () => `After moves like this the next ${name} averaged ${sp(a.mean_pct)}, but ${sp(a.older_mean_pct)} in the older half of the history and ${sp(a.newer_mean_pct)} in the newer: it didn't hold.`,
  }[h.reason];
  const label = { flat: "Flat", few: "Thin data" }[h.reason] || "No edge";
  return { state: "idle", label, pill: "pill-neutral", why: why ? why() : "" };
}

function signalPill(coin) {
  if (!coin.has_data) {
    if (isBusy(coin)) return `<span class="pill pill-warn">${esc(loadingStage(coin))}</span>`;
    if (coin.status === "failed") {
      return `<span class="pill pill-neutral" data-tooltip-title="Download failed" data-tooltip="${esc(coin.error || "")}">Failed</span>`;
    }
    return `<span class="pill pill-neutral">Loading</span>`;
  }
  const v = signalVerdict(coin);
  return `<span class="pill ${v.pill}" data-tooltip-title="${esc(`${currentTimeframe} momentum · ${v.label}`)}" data-tooltip="${esc(v.why)}">${esc(v.label)}</span>`;
}

function emptyCell(cls, sub = "") {
  return `<div class="${cls}"><span class="val val-muted">—</span>${sub ? `<div class="cell-sub">${esc(sub)}</div>` : ""}</div>`;
}

function waitingNote(coin) {
  if (isBusy(coin)) return "downloading…";
  if (coin.status === "failed") return "download failed";
  return coin.has_data ? "too little history" : "";
}

function moveCell(coin) {
  const cls = "c-move num";
  const h = horizonOf(coin);
  if (!hasMove(h)) return emptyCell(cls, waitingNote(coin));
  const tone = Math.abs(h.z) >= 1 ? toneClass(h.return_pct) : "val-flat";
  return `
    <div class="${cls}" data-tooltip-title="${esc(`Move · last ${currentTimeframe}`)}" data-tooltip="${esc(moveText(coin, h, currentTimeframe))}">
      <span class="val val-lg ${tone}">${sp(h.return_pct)}</span>
      <div class="cell-sub">z ${signed(h.z, 1)} · ${pctile(h.percentile)} pct</div>
    </div>`;
}

function nextCell(coin) {
  const cls = "c-next num";
  const h = horizonOf(coin);
  const a = h && h.after;
  if (!a) return emptyCell(cls, waitingNote(coin));
  const tone = significant(a) ? toneClass(a.mean_pct, 2) : "val-flat";
  return `
    <div class="${cls}" data-tooltip-title="${esc(`Next ${currentTimeframe} after similar moves`)}" data-tooltip="${esc(afterText(a, currentTimeframe))}">
      <span class="val val-lg ${tone}">${sp(a.mean_pct)}</span>
      <div class="cell-sub">${pct(a.hit_rate)} up · ${significant(a) ? "significant" : "noise"}</div>
    </div>`;
}

function characterText(h, name) {
  const p = h.persistence || {};
  const c = CHARACTER[p.character];
  return [
    `Variance ratio ${formatNumber(p.vr, 2)} (z ${signed(p.vr_z, 1)}, ${pText(p.vr_p)}): ${c ? c.note : "not enough history"}. Above 1 is trending, below 1 reverting.`,
    `Last ${name} move against the next ${name}: correlation ${signed(p.ic, 3)} (t ${signed(p.ic_t, 1)}, ${pText(p.ic_p)}).`,
  ].join("\n");
}

function characterCell(coin) {
  const cls = "c-char";
  const h = horizonOf(coin);
  const p = h && h.persistence;
  if (!p || !p.character) return emptyCell(cls, waitingNote(coin));
  const c = CHARACTER[p.character];
  return `
    <div class="${cls}" data-tooltip-title="${esc(`Character · ${currentTimeframe}`)}" data-tooltip="${esc(characterText(h, currentTimeframe))}">
      <span class="val ${p.character === "random" ? "val-flat" : ""}">${esc(c.label)}</span>
      <div class="cell-sub">VR ${formatNumber(p.vr, 2)} · ${pText(p.vr_p)}</div>
    </div>`;
}

function flowCell(coin) {
  const h = horizonOf(coin);
  const f = h && h.flow;
  if (!f) return `<div class="c-flow"><div class="book-bar is-empty"></div><div class="cell-sub">${esc(waitingNote(coin) || "No volume data")}</div></div>`;
  const lean = f.percentile >= 80 ? " val-pos" : (f.percentile <= 20 ? " val-neg" : "");
  const width = Math.max(5, Math.min(95, f.buy_pct));
  return `
    <div class="c-flow" data-tooltip-title="${esc(`Taker flow · last ${currentTimeframe}`)}" data-tooltip="${esc(`Market buys took ${formatNumber(f.buy_pct, 1)}% of the traded value over the last ${currentTimeframe} (the rest were market sells): more buying than ${formatNumber(f.percentile, 0)}% of past ${currentTimeframe} windows.`)}">
      <div class="book-bar"><span class="book-bid" style="width:${width}%"></span><span class="book-ask"></span></div>
      <div class="cell-sub${lean}">${formatNumber(f.buy_pct, 0)}% buys · ${pctile(f.percentile)} pct</div>
    </div>`;
}

function bookCell(depth) {
  if (!isNum(depth.bid_pct) || depth.available === false) {
    return `<div class="c-book"><div class="book-bar is-empty"></div><div class="cell-sub">No book data</div></div>`;
  }
  const imb = depth.book_imbalance_pct || 0;
  const lean = imb >= 12 ? " val-pos" : (imb <= -12 ? " val-neg" : "");
  return `
    <div class="c-book" data-tooltip-title="Order book · top 20 levels" data-tooltip="${esc(`Bids ${formatUsdCompact(depth.bid_depth_usd)} · ${depth.bid_pct}%\nAsks ${formatUsdCompact(depth.ask_depth_usd)} · ${depth.ask_pct}%\nSpread ${formatNumber(depth.spread_bps, 1)} bps`)}">
      <div class="book-bar"><span class="book-bid" style="width:${depth.bid_pct}%"></span><span class="book-ask"></span></div>
      <div class="cell-sub${lean}">${depth.bid_pct}% bids</div>
    </div>`;
}

// ───────── Expanded details ─────────
//
// The selected horizon's verdict and its numbers first, then every horizon side by side,
// then the supporting signals.

function stat(label, value, sub = "", tip = null, valueClass = "") {
  const attrs = tip ? ` data-tooltip-title="${esc(tip.title)}" data-tooltip="${esc(tip.body)}"` : "";
  return `
    <div class="trade-stat"${attrs}>
      <span class="trade-stat-label">${label}</span>
      <span class="trade-stat-value ${valueClass}">${value}</span>
      ${sub ? `<span class="trade-stat-sub">${sub}</span>` : ""}
    </div>`;
}

function verdictBlock(coin) {
  const name = currentTimeframe;
  const h = horizonOf(coin);
  if (!coin.has_data || !h) return "";
  const v = signalVerdict(coin);
  const a = h.after;
  const vs = h.versus;
  const f = h.flow;
  const base = vs ? splitSymbol(vs.symbol).base : "";
  const stats = hasMove(h) ? `
      <div class="trade-stats">
        ${stat(`Move · last ${name}`, sp(h.return_pct), `z ${signed(h.z, 1)} · ${pctile(h.percentile)} pct`,
          { title: "Move", body: moveText(coin, h, name) }, Math.abs(h.z) >= 1 ? toneClass(h.return_pct) : "")}
        ${stat(`Next ${name} after similar`, a ? sp(a.mean_pct) : "—", a ? `95%: ${rangeText(a)}` : "",
          a ? { title: "After similar moves", body: afterText(a, name) } : null, significant(a) ? toneClass(a.mean_pct) : "")}
        ${stat("Rose afterwards", a ? pct(a.hit_rate) : "—", a ? `all windows ${pct(a.all_hit_rate)}` : "",
          { title: "Share up", body: `How often the next ${name} closed higher after moves like this, against every ${name} window.` })}
        ${stat(vs ? `vs ${esc(base)} · beta-adjusted` : "vs benchmark", vs ? sp(vs.excess_pct) : "—",
          vs ? `${esc(base)} ${sp(vs.return_pct)} · β ${formatNumber(vs.beta, 2)}` : "",
          vs ? { title: `Against ${base}`, body: `The coin's last ${name} return minus ${formatNumber(vs.beta, 2)} × ${base}'s (${sp(vs.return_pct)}): how much it moved on its own. Beta is from ${h.interval} returns over the last ${formatNumber(momentumOf(coin).vol_days, 0)} days.` } : null,
          vs ? toneClass(vs.excess_pct) : "")}
        ${stat(`Taker buys · last ${name}`, f ? `${formatNumber(f.buy_pct, 1)}%` : "—", f ? `${pctile(f.percentile)} pct of past ${name}` : "",
          { title: "Taker flow", body: `Share of the traded value bought with market orders. 50% is balanced.` })}
      </div>` : "";
  return `
    <section class="detail-block detail-wide trade-block">
      <h3>Momentum · ${esc(name)} <span class="h3-note">as of ${whenText(h.as_of)} · ${esc(h.candles)} × ${esc(h.interval)} candles · fees ${formatNumber(momentumOf(coin).round_trip_pct, 2)}% round trip</span></h3>
      <div class="trade-side is-${v.state}">
        <div class="trade-head"><span>${esc(name)} momentum</span><span class="pill ${v.pill}">${esc(v.label)}</span></div>
        <p class="trade-why">${esc(v.why)}</p>
        ${stats}
      </div>
    </section>`;
}

function horizonRow(coin, name) {
  const h = horizonOf(coin, name);
  const current = name === currentTimeframe ? " is-current" : "";
  const pick = `onclick="event.stopPropagation(); setTimeframe('${name}')"`;
  if (!hasMove(h)) {
    return `<tr class="hz-row${current}" ${pick}><td><strong>${name}</strong></td><td colspan="7" class="val-muted">Not enough history</td></tr>`;
  }
  const a = h.after;
  const p = h.persistence || {};
  const v = signalVerdict(coin, name);
  const c = CHARACTER[p.character];
  return `
    <tr class="hz-row${current}" ${pick}>
      <td><strong>${name}</strong> <span class="val-muted">${esc(h.interval)}</span></td>
      <td class="num"><span class="${Math.abs(h.z) >= 1 ? toneClass(h.return_pct) : ""}">${sp(h.return_pct)}</span> <span class="val-muted">z ${signed(h.z, 1)} · ${pctile(h.percentile)}</span></td>
      <td class="num" data-tooltip-title="After similar moves" data-tooltip="${a ? esc(afterText(a, name)) : ""}">${a ? `<strong class="${significant(a) ? toneClass(a.mean_pct) : ""}">${sp(a.mean_pct)}</strong> <span class="val-muted">${rangeText(a)}</span>` : "—"}</td>
      <td class="num">${a ? `${pct(a.hit_rate)} <span class="val-muted">/ ${pct(a.all_hit_rate)}</span>` : "—"}</td>
      <td class="num">${a ? `${sp(a.older_mean_pct)} <span class="val-muted">/</span> ${sp(a.newer_mean_pct)}` : "—"}</td>
      <td class="num">${signed(p.ic, 3)} <span class="val-muted">${pText(p.ic_p)}</span></td>
      <td class="num" data-tooltip-title="Character" data-tooltip="${esc(characterText(h, name))}">${formatNumber(p.vr, 2)} <span class="val-muted">${c ? c.label : ""}</span></td>
      <td><span class="pill pill-sm ${v.pill}" data-tooltip-title="${esc(`${name} · ${v.label}`)}" data-tooltip="${esc(v.why)}">${esc(v.label)}</span></td>
    </tr>`;
}

function horizonsBlock(coin) {
  const horizons = momentumOf(coin).horizons || {};
  if (!coin.has_data || !Object.keys(horizons).length) return "";
  return `
    <section class="detail-block detail-wide detail-table">
      <h3>Every horizon <span class="h3-note">click a row to select it</span></h3>
      <table class="data-table hz-table">
        <thead>
          <tr>
            <th>Horizon</th>
            <th class="num" data-tooltip-title="Move" data-tooltip="Return over the last horizon, its z (the move in units of a typical move of that length) and its percentile among past moves.">Last move</th>
            <th class="num" data-tooltip-title="Next, after similar moves" data-tooltip="Mean return over the next horizon after past moves in the same fifth as the latest one, with the 95% range of that mean.">Next, after similar</th>
            <th class="num" data-tooltip-title="Rose / all" data-tooltip="How often the next horizon closed higher after similar moves, and after every window.">Rose / all</th>
            <th class="num" data-tooltip-title="Older / newer" data-tooltip="The same mean in the older and the newer half of the history. A real effect should show in both.">Older / newer</th>
            <th class="num" data-tooltip-title="Past → next correlation" data-tooltip="Correlation of each move with the move that followed it, over the whole history (information coefficient). Positive: momentum. Negative: reversal.">Past → next</th>
            <th class="num" data-tooltip-title="Variance ratio" data-tooltip="Lo–MacKinlay variance ratio: variance of moves this long over the variance a random walk would give. Above 1 trending, below 1 reverting.">Var. ratio</th>
            <th>Signal</th>
          </tr>
        </thead>
        <tbody>${HORIZONS.map((name) => horizonRow(coin, name)).join("")}</tbody>
      </table>
    </section>`;
}

function rsiText(v) {
  if (!isNum(v)) return `<span class="val-muted">—</span>`;
  const cls = v >= 70 || v <= 30 ? "val-warn" : "";
  return `<strong class="${cls}">${formatNumber(v, 0)}</strong>`;
}

// The supporting signals: RSI, moving-average stack, ranges, volatility, volume, funding, order book
function signalsBlock(coin) {
  const s = coin.signals || {};
  if (!coin.has_data || !Object.keys(s).length) return "";
  const rows = [];
  const rsi = s.rsi || {};
  rows.push(["RSI 14", Object.keys(rsi).map((iv) => `${esc(iv)} ${rsiText(rsi[iv])}`).join(" &nbsp; "),
    "Wilder's relative strength index on closed candles. Above 70 is stretched up, below 30 stretched down."]);
  if (s.ema) {
    const e = s.ema.ema || {};
    const stack = { up: ["Up stack", "val-pos"], down: ["Down stack", "val-neg"], mixed: ["Mixed", ""] }[s.ema.stack] || ["—", ""];
    rows.push([`Trend ${esc(s.ema.interval)}`,
      `<strong class="${stack[1]}">${stack[0]}</strong> <span class="val-muted">· price ${sp(s.ema.vs_slowest_pct, 1)} vs EMA 200</span>`,
      `Price against its 20, 50 and 200-candle exponential moving averages (${formatPrice(e["20"])}, ${formatPrice(e["50"])}, ${formatPrice(e["200"])}). Up stack: price > EMA20 > EMA50 > EMA200.`]);
  }
  const ranges = s.ranges || {};
  const rangeParts = Object.keys(ranges).filter((k) => ranges[k]).map((k) => `${esc(k)} <strong>${pct(ranges[k].position)}</strong>`);
  if (rangeParts.length) {
    const tip = Object.keys(ranges).filter((k) => ranges[k]).map((k) => `${k}: ${formatPrice(ranges[k].low)} – ${formatPrice(ranges[k].high)}`).join("\n");
    rows.push(["Range", rangeParts.join(" &nbsp; "), `Where price sits between the low (0%) and high (100%) of each window.\n${tip}`]);
  }
  if (s.volatility) {
    rows.push(["Volatility", `<strong>${formatNumber(s.volatility.daily_pct, 2)}%</strong> a day <span class="val-muted">· ${pctile(s.volatility.percentile)} pct</span>`,
      "Realized volatility of the last 24 hours of 15m candles, scaled to a day, and its percentile among every earlier day on hand."]);
  }
  if (s.volume) {
    rows.push(["Volume", `<strong>${formatUsdCompact(s.volume.last_24h_usd)}</strong> in 24h <span class="val-muted">· ${formatNumber(s.volume.ratio, 2)}× the median day</span>`,
      `Traded value of the last 24 hours against the median of the 30 days before (${formatUsdCompact(s.volume.typical_24h_usd)}).`]);
  }
  if (s.funding) {
    rows.push(["Funding", `<strong class="${toneClass(s.funding.rate_pct, 4)}">${sp(s.funding.rate_pct, 4)}</strong> <span class="val-muted">· ${pctile(s.funding.percentile)} pct of ${s.funding.days} days</span>`,
      `The latest settled funding rate (per settlement; longs pay shorts when positive). Binance's baseline is +0.0100%. Mean over ${s.funding.days} days ${sp(s.funding.mean_pct, 4)}.`]);
  }
  const d = coin.orderbook_depth || {};
  if (d.available) {
    rows.push(["Order book", `<strong>${d.bid_pct}%</strong> bids <span class="val-muted">· top 5 ${signed(d.near_imbalance_pct, 0)}% · spread ${formatNumber(d.spread_bps, 2)} bps</span>`,
      "Resting limit orders on the bid side across the top 20 levels, the bid/ask imbalance of the top 5, and the spread. Display only."]);
  }
  return `
    <section class="detail-block detail-wide detail-quiet">
      <h3>Signals</h3>
      <ul class="levels">
        ${rows.map(([label, value, tip]) => `<li><span class="lvl-chart" data-tooltip-title="${esc(label.replace(/<[^>]+>/g, ""))}" data-tooltip="${esc(tip)}">${label}</span><span>${value}</span></li>`).join("")}
      </ul>
    </section>`;
}

function historyNote(coin) {
  const hist = coin.history || {};
  const parts = Object.keys(hist).filter((iv) => hist[iv] && hist[iv].candles)
    .map((iv) => `${spanText(hist[iv].from)} of ${iv}`);
  return parts.length ? `Statistics from ${parts.join(" and ")} candles` : "";
}

function detailsFooter(coin, symbol) {
  let left;
  if (isBusy(coin)) {
    const p = coin.progress;
    const bar = p && isNum(p.fraction) && coin.status !== "queued"
      ? `<span class="progress-track" aria-hidden="true"><span class="progress-fill" style="width:${Math.round(p.fraction * 100)}%"></span></span>`
      : "";
    left = `<span class="loading-tag">${esc(loadingDetail(coin))}</span>${bar}`;
  } else {
    const error = coin.status === "failed" && coin.error ? `<span class="val-neg">${esc(coin.error)}</span>` : "";
    const note = historyNote(coin);
    left = `<button class="btn btn-sm" onclick="reloadCoin('${symbol}')">${coin.status === "failed" ? "Retry download" : "Reload history"}</button>${error}${note ? `<span class="val-muted">${esc(note)}</span>` : ""}`;
  }
  return `
    <div class="details-footer">
      ${left}
      <span class="spacer"></span>
      <a class="text-link" href="${coindcxUrl(symbol)}" target="_blank" rel="noopener noreferrer">Open on CoinDCX ${ICON_EXTERNAL}</a>
      <button class="btn btn-sm btn-ghost-danger" onclick="removeCoin('${symbol}')">Remove</button>
    </div>`;
}

// Row Renderer
function renderCoinCard(symbol, flash = false) {
  const coin = coinsData[symbol];
  if (!coin) return;

  const container = document.getElementById("coins-container");
  const cardId = cardIdFor(symbol);
  let card = document.getElementById(cardId);

  if (!card) {
    const empty = container.querySelector(".empty-state");
    if (empty) empty.remove();
    card = document.createElement("article");
    card.className = "coin-row";
    card.id = cardId;
    container.appendChild(card);
  }

  const { base, quote } = splitSymbol(symbol);
  const isExpanded = Boolean(window.expandedCards[symbol]);
  const detailsId = `${cardId}-details`;

  let flashClass = "";
  const price = Number(coin.current_price);
  if (flash && isNum(lastPrices[symbol]) && price !== lastPrices[symbol]) {
    flashClass = price > lastPrices[symbol] ? "flash-green" : "flash-red";
  }
  if (isNum(price) && price) lastPrices[symbol] = price;

  const subParts = [];
  if (coin.sector) subParts.push(esc(sectorLabel(coin.sector)));
  if (isBusy(coin)) subParts.push(`<span class="loading-tag">${esc(loadingStage(coin))}</span>`);

  card.classList.toggle("is-open", isExpanded);
  card.setAttribute("aria-label", base);
  card.innerHTML = `
    <div class="row-main" onclick="toggleCardDetails('${symbol}')">
      <div class="c-pair">
        <div class="pair-line">
          <a class="pair-base symbol-link" href="${coindcxUrl(symbol)}" target="_blank" rel="noopener noreferrer"
             onclick="event.stopPropagation()" title="Open ${esc(symbol)} on CoinDCX">${esc(base)}</a>
          ${quote ? `<span class="pair-quote">${esc(quote)}</span>` : ""}
        </div>
        <div class="cell-sub">${subParts.join(" · ")}</div>
      </div>

      <div class="c-price num"><span class="price-val ${flashClass}">${formatPrice(coin.current_price)}</span></div>

      <div class="c-signal">${signalPill(coin)}</div>

      ${moveCell(coin)}
      ${nextCell(coin)}
      ${characterCell(coin)}
      ${flowCell(coin)}
      ${bookCell(coin.orderbook_depth || {})}

      <div class="c-actions">
        <button class="icon-btn btn-remove" onclick="event.stopPropagation(); removeCoin('${symbol}')" aria-label="Remove ${esc(base)}" title="Remove">${ICON_CLOSE}</button>
        <button class="icon-btn btn-expand" onclick="event.stopPropagation(); toggleCardDetails('${symbol}')"
                aria-expanded="${isExpanded}" aria-controls="${detailsId}" aria-label="${isExpanded ? "Hide" : "Show"} details for ${esc(base)}">${ICON_CHEVRON}</button>
      </div>
    </div>

    <div class="row-details" id="${detailsId}" ${isExpanded ? "" : "hidden"}>
      ${isExpanded ? `
        <div class="details-grid">
          ${verdictBlock(coin)}
          ${horizonsBlock(coin)}
          ${signalsBlock(coin)}
        </div>
        ${detailsFooter(coin, symbol)}
      ` : ""}
    </div>
  `;

  updateSummary();
}

function updateSummary() {
  const el = document.getElementById("watchlist-summary");
  if (!el) return;
  const coins = Object.values(coinsData);
  if (coins.length === 0) {
    el.textContent = "No pairs tracked";
    return;
  }
  const counts = { LONG: 0, SHORT: 0 };
  let downloading = 0;
  coins.forEach((c) => {
    const h = horizonOf(c);
    if (h && h.signal in counts) counts[h.signal]++;
    if (isBusy(c)) downloading++;
  });
  el.innerHTML = `${coins.length} ${coins.length === 1 ? "pair" : "pairs"} · `
    + `<span class="${counts.LONG ? "sum-pos" : ""}">${counts.LONG} long</span> · `
    + `<span class="${counts.SHORT ? "sum-neg" : ""}">${counts.SHORT} short</span> at ${esc(currentTimeframe)}`
    + (downloading ? ` · ${downloading} downloading` : "");
}

function renderAll() {
  const container = document.getElementById("coins-container");
  const symbols = Object.keys(coinsData);

  document.querySelectorAll("[data-horizon-label]").forEach((el) => {
    el.textContent = el.dataset.horizonLabel.replace("{h}", currentTimeframe);
  });

  if (symbols.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <h3>Your watchlist is empty</h3>
        <p>Add a CoinDCX futures pair to start watching its momentum statistics.</p>
        <button class="btn btn-primary" onclick="openSidebar()">Add pair <kbd class="hotkey-hint">/</kbd></button>
      </div>
    `;
    updateSummary();
    if (typeof renderSidebarList === "function") {
      renderSidebarList();
    }
    return;
  }

  // Remove any stale DOM rows
  const activeCardIds = new Set(symbols.map(cardIdFor));
  Array.from(container.children).forEach((child) => {
    if (!child.id || !activeCardIds.has(child.id)) {
      child.remove();
    }
  });

  symbols.forEach((sym) => renderCoinCard(sym, false));
  if (typeof renderSidebarList === "function") {
    renderSidebarList();
  }
}

// Initial Bootstrapping
async function loadInitialState() {
  try {
    const resp = await fetch("/api/coins");
    if (resp.ok) {
      const list = await resp.json();
      list.forEach((c) => {
        coinsData[c.symbol] = c;
      });
      renderAll();
    }
  } catch (err) {
    console.error("Initial load failed:", err);
  }
  initWebSocket();
}

window.addEventListener("DOMContentLoaded", loadInitialState);
