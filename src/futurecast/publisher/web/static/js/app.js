/**
 * Main Web Terminal Application Logic.
 * Handles WebSocket streaming, state management, CoinDCX integration, and DOM rendering.
 *
 * Everything shown comes from the server as-is: when a model is missing, untrained or has
 * no proven edge the page says so instead of filling the gap.
 */

// Global State
window.coinsData = {};
const coinsData = window.coinsData;
window.expandedCards = window.expandedCards || {};
const lastPrices = {};
let ws = null;

const HORIZON_ORDER = ["5m", "15m", "1h", "4h", "8h"];
const MODEL_ORDER = ["LightGBM", "XGBoost", "CatBoost"];
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
const BIAS_LABELS = {
  LONG_RISKIER: { label: "Longs riskier", tag: "tag-warn" },
  SHORT_RISKIER: { label: "Shorts riskier", tag: "tag-warn" },
  BALANCED: { label: "Balanced", tag: "" },
};
const RISK_STATUS = {
  edge: { label: "Beats baseline", tag: "tag-pos", note: "The models beat a volatility-only baseline on unseen data (90% confidence)." },
  no_edge: { label: "Volatility baseline", tag: "", note: "The models did not beat a volatility-only baseline on unseen data, so the baseline is shown." },
  insufficient_data: { label: "Too little data", tag: "tag-warn", note: "Too few independent test windows to judge, so the volatility baseline is shown." },
};
const STATUS_INFO = {
  edge: { label: "Proven edge", tag: "tag-pos", note: "Beat the volatility-only odds on unseen data (90% confidence), and the long / short split itself predicted outcomes." },
  no_edge: { label: "No edge", tag: "", note: "Did not beat the volatility-only odds on unseen data, so those are shown." },
  insufficient_data: { label: "Too little data", tag: "tag-warn", note: "Too few independent test trades to judge, so the volatility-only odds are shown." },
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
      body: JSON.stringify({ symbol: symbol, auto_train: true }),
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

async function trainCoin(symbol) {
  if (!coinsData[symbol]) return;
  coinsData[symbol].training_status = "queued";
  renderCoinCard(symbol, false);
  try {
    await fetch(`/api/coins/${encodeURIComponent(symbol)}/train`, { method: "POST" });
  } catch (err) {
    console.error("Failed to train coin:", err);
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

function pct(p, decimals = 0) {
  return isNum(p) ? `${(Number(p) * 100).toFixed(decimals)}%` : "—";
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

function titleCase(s) {
  return (s || "").toLowerCase().replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function riskLevelClass(level) {
  const lvl = (level || "").toUpperCase();
  if (lvl === "EXTREME") return "lvl-extreme";
  if (lvl === "HIGH") return "lvl-high";
  if (lvl === "MODERATE") return "lvl-moderate";
  return "lvl-low";
}

function coindcxUrl(symbol) {
  return symbol.startsWith("B-") ? "https://coindcx.com/futures/" + symbol : "https://coindcx.com/trade/" + symbol;
}

// ───────── Trade probability helpers ─────────

function horizonData(coin, horizon = currentTimeframe) {
  return (coin.trade_probabilities || {})[horizon] || null;
}

function signalKey(signal) {
  if (signal === "LONG") return "bull";
  if (signal === "SHORT") return "bear";
  return "neutral";
}

const BUSY_STATUSES = ["queued", "downloading", "training"];

function isBusy(coin) {
  return BUSY_STATUSES.includes(coin.training_status);
}

function trainingVerb(coin) {
  return { queued: "Queued", downloading: "Downloading", training: "Training" }[coin.training_status] || "";
}

// "Downloading 34%": share of this stage done, from the server's training_progress
function trainingStage(coin) {
  const p = coin.training_progress;
  const verb = trainingVerb(coin);
  return p && isNum(p.fraction) && coin.training_status !== "queued" ? `${verb} ${Math.floor(p.fraction * 100)}%` : verb;
}

function trainingDetail(coin) {
  const p = coin.training_progress;
  if (coin.training_status === "queued") return "Queued for training (one coin at a time)";
  if (!p) return coin.training_status === "downloading" ? "Downloading candles…" : "Training on real candles…";
  if (p.stage === "download") {
    const counts = p.total ? ` · ${p.done.toLocaleString()} of ${p.total.toLocaleString()} new` : "";
    return `Downloading ${p.label}${counts} · ${p.step} of ${p.steps}`;
  }
  return `Training ${p.label} · step ${p.step} of ${p.steps}`;
}

function trainingNote(coin) {
  const status = coin.training_status;
  if (status === "downloading") return "downloading…";
  if (status === "training") return "training…";
  if (status === "queued") return "queued";
  if (status === "failed") return "training failed";
  if (status === "untrained") return "not trained";
  return "no model";
}

// Odds clear break-even with proven skill, but replaying such signals on unseen data
// did not make money after fees on enough trades, so no signal is raised.
function unprovenSide(h) {
  if (!h || h.signal !== "NEUTRAL") return null;
  return ["long", "short"].find((side) => {
    const d = h[side];
    return d && d.status === "edge" && !d.signal_validated && d.probability >= d.break_even;
  }) || null;
}

const UNPROVEN_TIP = "The odds clear break-even, but taking signals like this on history the models never saw did not make money after fees on at least 10 trades. No signal until that changes.";

function signalPill(coin) {
  if (!coin.has_data) return `<span class="pill pill-neutral">Loading</span>`;
  const h = horizonData(coin);
  if (!h) {
    const busy = isBusy(coin);
    return `<span class="pill ${busy ? "pill-warn" : "pill-neutral"}">${busy ? trainingVerb(coin) : "No model"}</span>`;
  }
  if (h.signal === "LONG") return `<span class="pill pill-bull">Long</span>`;
  if (h.signal === "SHORT") return `<span class="pill pill-bear">Short</span>`;
  const side = unprovenSide(h);
  if (side) {
    return `<span class="pill pill-warn" data-tooltip-title="Unproven ${side}" data-tooltip="${esc(UNPROVEN_TIP)}">Unproven</span>`;
  }
  return `<span class="pill pill-neutral">Neutral</span>`;
}

// Reward to risk of a [take-profit, stop] level in volatility units, e.g. "2:1"
function rewardRisk(level) {
  const [tp, sl] = level || [1, 1];
  const ratio = (a, b) => String(+(a / b).toFixed(1));
  return tp >= sl ? `${ratio(tp, sl)}:1` : `1:${ratio(sl, tp)}`;
}

// What a probability without a proven edge is: volatility-only odds, or the base rate when even those are missing
function probabilitySource(d) {
  return d.probability_source === "base_rate" ? "the recent base rate" : "what volatility alone predicts";
}

function levelText(side, d) {
  const up = side === "long";
  return `${up ? "+" : "−"}${formatNumber(d.tp_pct, 2)}% / ${up ? "−" : "+"}${formatNumber(d.sl_pct, 2)}%`;
}

// How the take-profit / stop pair was chosen, with every pair the models scored
function levelTip(side, d) {
  const levels = d.levels || [];
  const how = d.level_pick === "model"
    ? `Picked from ${levels.length} take-profit / stop pairs as the one with the highest expected profit after fees at the models' odds: ${signed(d.expected_return_pct, 3)}% per trade.`
    : `No pair has a proven edge, so this is the pair whose volatility-only odds lose least after fees (${signed(d.expected_return_pct, 3)}% per trade).`;
  const rows = levels.map((o) => {
    const mark = o.level[0] === d.level[0] && o.level[1] === d.level[1] ? "▸ " : "";
    const odds = `${pct(o.probability)} win${o.status === "edge" ? "" : o.probability_source === "base_rate" ? " (base rate)" : " (volatility only)"}`;
    return `${mark}${levelText(side, o)} · ${odds} · ${signed(o.expected_return_pct, 3)}%`;
  });
  return {
    title: `Take-profit and stop · ${rewardRisk(d.level)} reward to risk`,
    body: [`${formatNumber(d.level[0], 1)}× / ${formatNumber(d.level[1], 1)}× this horizon's volatility, never closer than 2× the round-trip fees.`, how, ...rows].join("\n"),
  };
}

function sideTooltip(side, d, h, horizon) {
  const name = side === "long" ? "Long" : "Short";
  const m = d.metrics || {};
  const lines = [
    `Take-profit ${side === "long" ? "+" : "−"}${formatNumber(d.tp_pct, 2)}% before stop-loss ${side === "long" ? "−" : "+"}${formatNumber(d.sl_pct, 2)}% within ${horizon} (${rewardRisk(d.level)}, picked by the models)`,
    `Needs ${pct(d.break_even)} to break even after ${formatNumber(h.cost_pct, 2)}% fees`,
    (STATUS_INFO[d.status] || {}).note || "",
  ];
  if (isNum(m.n_independent)) lines.push(`Tested on ${m.n_independent.toLocaleString()} independent past trades`);
  return { title: `${name} wins ${pct(d.probability)} of the time`, body: lines.filter(Boolean).join("\n") };
}

function winCell(coin, side) {
  const cls = `c-${side} num`;
  if (!coin.has_data) return `<div class="${cls}"><span class="val val-muted">—</span></div>`;
  const h = horizonData(coin);
  const d = h && h[side];
  if (!d) {
    return `<div class="${cls}"><span class="val val-muted">—</span><div class="cell-sub">${esc(trainingNote(coin))}</div></div>`;
  }
  const isSignal = h.signal === side.toUpperCase();
  const tone = isSignal ? (side === "long" ? "val-pos" : "val-neg") : (d.status === "edge" ? "val-flat" : "val-muted");
  const tip = sideTooltip(side, d, h, currentTimeframe);
  return `
    <div class="${cls}" data-tooltip-title="${esc(tip.title)}" data-tooltip="${esc(tip.body)}">
      <span class="val val-lg ${tone}">${pct(d.probability)}</span>
      <div class="cell-sub">needs ${pct(d.break_even)}</div>
    </div>`;
}

function modelSides(h) {
  if (!h || !h.long || !h.short) return [];
  const names = MODEL_ORDER.filter((m) => m in (h.long.per_model || {}) && m in (h.short.per_model || {}));
  return names.map((name) => {
    const pl = h.long.per_model[name];
    const ps = h.short.per_model[name];
    const ml = pl - h.long.break_even;
    const ms = ps - h.short.break_even;
    const key = Math.max(ml, ms) >= 0 ? (ml >= ms ? "bull" : "bear") : "neutral";
    return { name, key, pl, ps };
  });
}

function modelsCell(coin) {
  const h = horizonData(coin);
  const sides = modelSides(h);
  if (!sides.length) return `<div class="c-models"><span class="val-muted">—</span></div>`;
  const want = signalKey(h.signal);
  const agree = sides.filter((s) => s.key === want).length;
  const dots = sides.map((s) => `<i class="dot dot-${s.key}"></i>`).join("");
  const tip = sides.map((s) => `${s.name} · long ${pct(s.pl)} · short ${pct(s.ps)}`).join("\n");
  return `
    <div class="c-models" data-tooltip-title="${agree} of ${sides.length} models agree" data-tooltip="${esc(tip)}">
      <span class="dots">${dots}</span>
      <div class="cell-sub">${agree}/${sides.length} agree</div>
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

function riskData(coin, horizon = currentTimeframe) {
  return (coin.trade_risk || {})[horizon] || null;
}

// Volatility regime from the risk payload: name, how sure, and the chance of a stormy stretch ahead
function regimeLines(r, horizon = currentTimeframe) {
  if (!r || !r.regime) return [];
  const probs = r.regime_probs || {};
  const others = Object.keys(probs).filter((k) => k !== r.regime).map((k) => `${k} ${pct(probs[k])}`).join(", ");
  return [
    `Volatility regime: ${r.regime} (${pct(probs[r.regime])}${others ? `; ${others}` : ""}), from a hidden Markov model of recent returns.`,
    `Chance of a stormy stretch within ${horizon}: ${pct(r.storm_chance)}.`,
  ];
}

function regimeTag(coin) {
  const r = riskData(coin);
  if (!r || !r.regime) return "";
  const body = [...regimeLines(r), "Worst-case bounds are calibrated separately for calm, normal and stormy stretches."].join("\n");
  return `<span class="tag${r.regime === "stormy" ? " tag-warn" : ""}" data-tooltip-title="Volatility regime" data-tooltip="${esc(body)}">${esc(titleCase(r.regime))} volatility</span>`;
}

function worstMoveText(side, d) {
  return `${side === "long" ? "−" : "+"}${formatNumber(d.worst_move_pct, 2)}%`;
}

function riskCell(coin) {
  if (!coin.has_data) return `<div class="c-risk"><span class="val-muted">—</span></div>`;
  const r = riskData(coin);
  if (!r || !r.long || !r.short) {
    return `<div class="c-risk"><span class="val-muted">—</span><div class="cell-sub">${esc(trainingNote(coin))}</div></div>`;
  }
  const bias = (BIAS_LABELS[r.bias] || BIAS_LABELS.BALANCED).label;
  const line = (side, name) => {
    const d = r[side];
    return `${name} ${worstMoveText(side, d)} worst case · riskier than ${d.score}% of recent windows · max ${d.max_leverage}x`;
  };
  const notes = [...new Set(["long", "short"].map((side) => (RISK_STATUS[r[side].status] || {}).note).filter(Boolean))];
  const tip = [
    line("long", "Long"),
    line("short", "Short"),
    ...regimeLines(r),
    ...notes,
  ].join("\n");
  return `
    <div class="c-risk" data-tooltip-title="Worst 5% of moves within ${esc(currentTimeframe)} · ${esc(bias.toLowerCase())}" data-tooltip="${esc(tip)}">
      <span class="risk-pair">
        <span class="risk-num ${riskLevelClass(r.long.level)}"><small>L</small>${r.long.score}</span>
        <span class="risk-num ${riskLevelClass(r.short.level)}"><small>S</small>${r.short.score}</span>
      </span>
      <div class="cell-sub">Max ${r.long.max_leverage}x / ${r.short.max_leverage}x</div>
    </div>`;
}

// ───────── Expanded details ─────────

function replayText(r) {
  return r.trades
    ? `${r.trades} ${r.trades === 1 ? "trade" : "trades"} · ${pct(r.win_rate)} won · ${signed(r.avg_net_return_pct, 2)}% avg after fees`
    : "never cleared break-even";
}

// Why a side is or isn't a trade right now. Mirrors the signal rules in models/trade_probability.py.
function sideVerdict(h, side) {
  const d = h[side];
  const r = d.replay || {};
  const odds = `${pct(d.probability)} win odds`;
  const gap = Math.round(d.probability * 100) - Math.round(d.break_even * 100);
  const vsNeeded = gap === 0
    ? `level with the ${pct(d.break_even)} needed to break even`
    : `${Math.abs(gap)} pts ${gap > 0 ? "above" : "below"} the ${pct(d.break_even)} needed to break even`;
  if (h.signal === side.toUpperCase()) {
    return { state: `go-${side}`, label: `${titleCase(side)} signal`, pill: side === "long" ? "pill-bull" : "pill-bear", why: `${odds}, ${vsNeeded}` };
  }
  const idle = { state: "idle", label: "No trade", pill: "pill-neutral" };
  if (d.status !== "edge") {
    const what = d.status === "insufficient_data" ? "Too little test data to prove an edge" : "No proven edge";
    return { ...idle, why: `${what}: ${pct(d.probability)} is ${probabilitySource(d)}` };
  }
  if (d.probability < d.break_even) return { ...idle, why: `${odds}, ${vsNeeded}` };
  if (!d.signal_validated) {
    const n = r.trades || 0;
    const record = n < 10
      ? `only ${n} past ${n === 1 ? "signal" : "signals"} to judge by (needs 10)`
      : `past signals averaged ${signed(r.avg_net_return_pct, 2)}% after fees`;
    return { state: "unproven", label: "Unproven", pill: "pill-warn", why: `${odds} clear break-even, but ${record}` };
  }
  return { ...idle, why: `${odds} clear break-even, but the ${side === "long" ? "short" : "long"} side is stronger` };
}

function tradeStat(label, value, sub = "", tip = null, valueClass = "") {
  const attrs = tip ? ` data-tooltip-title="${esc(tip.title)}" data-tooltip="${esc(tip.body)}"` : "";
  return `
    <div class="trade-stat"${attrs}>
      <span class="trade-stat-label">${label}</span>
      <span class="trade-stat-value ${valueClass}">${value}</span>
      ${sub ? `<span class="trade-stat-sub">${sub}</span>` : ""}
    </div>`;
}

function tradeSide(coin, h, side) {
  const d = h[side];
  if (!d) return "";
  const v = sideVerdict(h, side);
  const up = side === "long";
  const price = Number(coin.current_price);
  const level = (dir, movePct) => (isNum(price) && price ? formatPrice(price * (1 + dir * movePct / 100)) : "—");
  const risk = (riskData(coin) || {})[side];

  let riskStats = tradeStat("Worst case", "—", trainingNote(coin)) + tradeStat("Max leverage", "—");
  if (risk) {
    const status = RISK_STATUS[risk.status] || {};
    const m = risk.metrics || {};
    const stormy = (m.coverage_by_regime || {}).stormy || {};
    const held = isNum(m.coverage) && isNum(m.n_independent)
      ? `Held ${pct(m.coverage, 1)} of the time on ${m.n_independent.toLocaleString()} unseen windows`
        + (isNum(stormy.model) && stormy.n_independent ? `, ${pct(stormy.model, 1)} in stormy stretches.` : ".")
      : "";
    const worstTip = {
      title: `Worst 5% of moves within ${currentTimeframe}`,
      body: [
        `In 95% of cases price ${up ? "falls" : "rises"} no more than ${formatNumber(risk.worst_move_pct, 2)}% against a ${side}.`,
        `Risk ${risk.score}/100: riskier than ${risk.score}% of this coin's recent windows.`,
        status.note,
        held,
      ].filter(Boolean).join("\n"),
    };
    const levTip = {
      title: "Max leverage",
      body: "Highest of 1, 2, 3, 5, 10 or 20x whose liquidation price, after maintenance margin and fees, stays beyond the worst case.",
    };
    riskStats = tradeStat("Worst case", worstMoveText(side, risk), `<span class="risk-num ${riskLevelClass(risk.level)}">${risk.score} · ${esc(titleCase(risk.level))}</span>`, worstTip, `is-${riskLevelClass(risk.level)}`)
      + tradeStat("Max leverage", `${risk.max_leverage}x`, "", levTip);
  }

  const replay = d.replay;
  const record = d.status === "edge" && replay && isNum(replay.trades)
    ? `<p class="trade-record"><span data-tooltip-title="Replay on unseen data" data-tooltip="On every candle of history the models never saw, take the proven take-profit / stop pair with the best expected profit, and trade whenever its odds cleared break-even, one position at a time. Signals only fire for sides whose replay made money after fees on 10+ trades.">When it said go</span> <strong>${esc(replayText(replay))}</strong></p>`
    : "";
  const tpTip = levelTip(side, d);

  return `
    <div class="trade-side is-${v.state}">
      <div class="trade-head">
        <span>${up ? "Long" : "Short"}</span>
        <span class="pill ${v.pill}">${v.label}</span>
      </div>
      <p class="trade-why">${esc(v.why)}</p>
      <div class="trade-stats">
        ${tradeStat("Take-profit", level(up ? 1 : -1, d.tp_pct), `${up ? "+" : "−"}${formatNumber(d.tp_pct, 2)}% · ${rewardRisk(d.level)}`, tpTip, "is-tp")}
        ${tradeStat("Stop", level(up ? -1 : 1, d.sl_pct), `${up ? "−" : "+"}${formatNumber(d.sl_pct, 2)}%`, tpTip, "is-sl")}
        ${riskStats}
      </div>
      ${record}
    </div>`;
}

// The few numbers you act on for the selected horizon, highlighted only when a side is a trade
function tradeBlock(coin) {
  const h = horizonData(coin);
  if (!h) return "";
  const elsewhere = HORIZON_ORDER.filter((k) => k !== currentTimeframe)
    .map((k) => [k, horizonData(coin, k)])
    .filter(([, o]) => o && (o.signal === "LONG" || o.signal === "SHORT"))
    .map(([k, o]) => `<button class="pill pill-sm pill-link ${o.signal === "LONG" ? "pill-bull" : "pill-bear"}" onclick="setTimeframe('${k}')">${titleCase(o.signal)} at ${k}</button>`)
    .join("");
  return `
    <section class="detail-block detail-wide trade-block">
      <h3>Trade · ${esc(currentTimeframe)} <span class="h3-note">levels from the live price ${formatPrice(coin.current_price)}</span> ${regimeTag(coin)}${elsewhere ? `<span class="trade-elsewhere">Other horizons ${elsewhere}</span>` : ""}</h3>
      <div class="sides-grid">
        ${tradeSide(coin, h, "long")}
        ${tradeSide(coin, h, "short")}
      </div>
    </section>`;
}

function statusText(d) {
  const info = STATUS_INFO[d.status] || { label: titleCase(d.status) };
  return `<span class="status-text${d.status === "edge" ? " is-edge" : ""}">${info.label}</span>`;
}

function horizonsBlock(coin) {
  const rows = HORIZON_ORDER.filter((key) => horizonData(coin, key)).map((key) => {
    const h = horizonData(coin, key);
    const unproven = unprovenSide(h);
    const sig = h.signal === "LONG" ? `<span class="pill pill-sm pill-bull">Long</span>`
      : h.signal === "SHORT" ? `<span class="pill pill-sm pill-bear">Short</span>`
      : unproven ? `<span class="pill pill-sm pill-warn" data-tooltip-title="Unproven ${unproven}" data-tooltip="${esc(UNPROVEN_TIP)}">Unproven ${unproven}</span>`
      : `<span class="status-text">Neutral</span>`;
    const side = (s) => {
      if (!h[s]) return `<td class="num">—</td><td></td>`;
      const odds = h.signal === s.toUpperCase()
        ? `<strong class="${s === "long" ? "val-pos" : "val-neg"}">${pct(h[s].probability)}</strong>`
        : `<span class="${h[s].status === "edge" ? "val-flat" : "val-muted"}">${pct(h[s].probability)}</span>`;
      return `<td class="num">${odds} <span class="val-muted">/ ${pct(h[s].break_even)}</span></td><td class="num val-muted">${levelText(s, h[s])}</td><td>${statusText(h[s])}</td>`;
    };
    return `
      <tr class="${key === currentTimeframe ? "is-current" : ""}" onclick="setTimeframe('${key}')" title="Show ${key}">
        <td><strong>${key}</strong> <span class="val-muted">· ${esc(h.candle_interval)} candles</span></td>
        ${side("long")}
        ${side("short")}
        <td>${sig}</td>
      </tr>`;
  }).join("");

  if (!rows) {
    return `
      <section class="detail-block detail-wide">
        <h3>Win probability</h3>
        <p class="detail-note">No trained models yet (${esc(trainingNote(coin))}). Probabilities appear once training on real candles finishes.${coin.training_error ? ` Error: ${esc(coin.training_error)}` : ""}</p>
      </section>`;
  }
  return `
    <section class="detail-block detail-wide detail-table detail-quiet">
      <h3>Win probability by horizon <span class="h3-note">probability / needed to break even</span></h3>
      <table class="data-table">
        <thead>
          <tr>
            <th>Horizon</th>
            <th class="num">Long win</th><th class="num" data-tooltip-title="Take-profit / stop" data-tooltip="Per side, the models pick the volatility-scaled pair with the highest expected profit after fees. Never closer than 2× the round-trip fees.">TP / SL</th><th>Long model</th>
            <th class="num">Short win</th><th class="num">TP / SL</th><th>Short model</th>
            <th>Signal</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </section>`;
}

// Reliability bars stay folded behind a one-line summary; remember which are open across re-renders
window.openReliability = window.openReliability || {};

function reliabilityBlock(symbol, side, rel) {
  if (!rel || !rel.length) return "";
  const gaps = rel.map((b) => Math.abs(b.observed - b.predicted));
  const avg = gaps.reduce((a, b) => a + b, 0) / gaps.length;
  const worst = rel[gaps.indexOf(Math.max(...gaps))];
  const key = `${symbol}:${side}`;
  return `
    <details class="reliability" ${window.openReliability[key] ? "open" : ""} ontoggle="window.openReliability['${key}'] = this.open">
      <summary data-tooltip-title="Said vs won" data-tooltip="Test predictions split into five equal groups, comparing the average odds the models gave with how often those trades won. Open to see each group: the tick is what the models said, the bar is what happened.">
        <span>Said vs won: off by ${(avg * 100).toFixed(0)} pts on average <span class="val-muted">(worst: said ${pct(worst.predicted)}, won ${pct(worst.observed)})</span></span>
        ${ICON_CHEVRON}
      </summary>
      <div class="rel-rows" aria-label="Predicted versus observed win rate on unseen data">
        ${rel.map((b) => `
          <div class="rel-row">
            <span class="rel-label">said ${pct(b.predicted)}</span>
            <span class="rel-track"><span class="rel-fill" style="width:${Math.min(100, b.observed * 100)}%"></span><span class="rel-mark" style="left:${Math.min(100, b.predicted * 100)}%"></span></span>
            <span class="rel-value">won ${pct(b.observed)}</span>
          </div>`).join("")}
      </div>
    </details>`;
}

function evidenceSide(coin, side, h) {
  const d = h[side];
  if (!d) return "";
  const m = d.metrics || {};
  const head = `<div class="risk-side-head"><span>${side === "long" ? "Long" : "Short"}</span></div>`;

  if (d.status !== "edge") {
    const reason = m.reason ? ` ${esc(m.reason.charAt(0).toUpperCase() + m.reason.slice(1))}.` : "";
    const verdict = d.status === "insufficient_data" ? "Too little test data to judge." : "Did not beat the volatility-only odds on unseen data.";
    return `
      <div class="evidence-side">
        ${head}
        <p class="side-note">${verdict}${reason}</p>
      </div>`;
  }

  return `
    <div class="evidence-side">
      ${head}
      <dl class="kv">
        <dt data-tooltip-title="${esc(levelTip(side, d).title)}" data-tooltip="${esc(levelTip(side, d).body)}">Take-profit / stop</dt><dd>${rewardRisk(d.level)} <span class="val-muted">(best of ${(d.levels || []).length} tried)</span></dd>
        <dt>Recent base rate</dt><dd>${pct(d.base_rate)}</dd>
        ${isNum(m.brier_skill) ? `<dt data-tooltip-title="Skill vs volatility only" data-tooltip="How much better the probabilities scored than the odds volatility alone gives: a hidden Markov model forecasts how volatile the coin will be, and an absorbing Markov chain turns that into the chance of touching the take-profit first. Knowing price will move a lot is not a direction call, so it earns nothing here. Above 0 is better; the lower bound is the 90% worst case.">Skill vs volatility only</dt><dd>${signed(m.brier_skill, 3)} <span class="val-muted">(≥ ${signed(m.brier_skill_lower, 3)})</span></dd>` : ""}
        ${isNum(m.direction_skill) ? `<dt data-tooltip-title="Direction check" data-tooltip="Scored against the models' own long and short odds averaged (shifted by the usual long / short gap). A model that only knows how much price will move gives both sides the same odds, so only a real direction call scores above 0.">Direction check</dt><dd>${signed(m.direction_skill, 3)} <span class="val-muted">(≥ ${signed(m.direction_skill_lower, 3)})</span></dd>` : ""}
        ${isNum(m.shrink) && m.shrink < 0.995 ? `<dt data-tooltip-title="Overconfidence correction" data-tooltip="On unseen data the raw odds ran too far from the volatility-only odds, so they are pulled back toward them by this much before being shown.">Odds tempered</dt><dd>${pct(1 - m.shrink)} toward volatility only</dd>` : ""}
        ${isNum(m.auc) ? `<dt data-tooltip-title="AUC" data-tooltip="How well the model ranks winners above losers. 0.5 is a coin flip, 1.0 is perfect.">Ranking (AUC)</dt><dd>${formatNumber(m.auc, 2)}</dd>` : ""}
      </dl>
      ${reliabilityBlock(coin.symbol, side, d.reliability)}
    </div>`;
}

// Supporting statistics behind the edge verdicts; skipped when neither side has an edge to back up
function evidenceBlock(coin) {
  const h = horizonData(coin);
  if (!h || !["long", "short"].some((s) => h[s] && h[s].status === "edge")) return "";
  const n = ["long", "short"].map((s) => h[s] && (h[s].metrics || {}).n_independent).find((v) => isNum(v));
  const tested = isNum(n) ? `tested on ${n.toLocaleString()} independent trades the models never saw` : "tested on history the models never saw";
  return `
    <section class="detail-block detail-wide detail-quiet">
      <h3>Evidence · ${currentTimeframe} <span class="h3-note">${esc(h.candle_interval)} candles × ${h.horizon_bars} · ${tested}</span></h3>
      <div class="sides-grid">
        ${evidenceSide(coin, "long", h)}
        ${evidenceSide(coin, "short", h)}
      </div>
    </section>`;
}

function bookBlock(depth) {
  if (!isNum(depth.bid_pct) || depth.available === false) {
    return `<section class="detail-block detail-wide detail-inline detail-quiet"><h3>Order book</h3><span class="inline-stats val-muted">No live order book data for this pair yet.</span></section>`;
  }
  const imb = depth.book_imbalance_pct || 0;
  const lean = imb >= 12 ? "Bid-heavy" : (imb <= -12 ? "Ask-heavy" : "Balanced");
  const tip = [
    `Bids ${formatUsdCompact(depth.bid_depth_usd)} · ${depth.bid_pct}%`,
    `Asks ${formatUsdCompact(depth.ask_depth_usd)} · ${depth.ask_pct}%`,
    `Microprice skew ${signed(depth.microprice_skew_bps, 1)} bps`,
    "Top 20 levels of the Binance perpetual. Shown for context; no model uses it.",
  ].join("\n");
  return `
    <section class="detail-block detail-wide detail-inline detail-quiet" data-tooltip-title="Order book · top 20 levels" data-tooltip="${esc(tip)}">
      <h3>Order book <span class="tag">${lean}</span></h3>
      <span class="inline-stats">
        <span><span class="val-muted">Imbalance</span> <strong>${signed(imb, 1)}%</strong></span>
        <span><span class="val-muted">Spread</span> <strong>${formatNumber(depth.spread_bps, 1)} bps</strong></span>
      </span>
    </section>`;
}

function detailsFooter(coin, symbol) {
  const benches = (coin.benchmarks || []).map((b) => splitSymbol(b).base).join(", ");
  let training;
  if (isBusy(coin)) {
    const p = coin.training_progress;
    const bar = p && isNum(p.fraction) && coin.training_status !== "queued"
      ? `<span class="progress-track" aria-hidden="true"><span class="progress-fill" style="width:${Math.round(p.fraction * 100)}%"></span></span>`
      : "";
    training = `<span class="training-tag">${esc(trainingDetail(coin))}</span>${bar}`;
  } else {
    const label = coin.is_custom_trained ? "Retrain" : "Train models";
    training = `<button class="btn btn-sm" onclick="trainCoin('${symbol}')">${label}</button>`;
  }
  const source = coin.data_source ? `Candles: <strong>${esc(coin.data_source)}</strong>` : "";
  return `
    <div class="details-footer">
      ${source ? `<span>${source}</span>` : ""}
      ${benches ? `<span>Benchmarks <strong>${esc(benches)}</strong></span>` : ""}
      ${training}
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
  if (isBusy(coin)) subParts.push(`<span class="training-tag">${esc(trainingStage(coin))}</span>`);

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

      ${winCell(coin, "long")}
      ${winCell(coin, "short")}
      ${modelsCell(coin)}
      ${bookCell(coin.orderbook_depth || {})}
      ${riskCell(coin)}

      <div class="c-actions">
        <button class="icon-btn btn-remove" onclick="event.stopPropagation(); removeCoin('${symbol}')" aria-label="Remove ${esc(base)}" title="Remove">${ICON_CLOSE}</button>
        <button class="icon-btn btn-expand" onclick="event.stopPropagation(); toggleCardDetails('${symbol}')"
                aria-expanded="${isExpanded}" aria-controls="${detailsId}" aria-label="${isExpanded ? "Hide" : "Show"} details for ${esc(base)}">${ICON_CHEVRON}</button>
      </div>
    </div>

    <div class="row-details" id="${detailsId}" ${isExpanded ? "" : "hidden"}>
      ${isExpanded ? `
        <div class="details-grid">
          ${tradeBlock(coin)}
          ${horizonsBlock(coin)}
          ${evidenceBlock(coin)}
          ${bookBlock(coin.orderbook_depth || {})}
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
  const counts = { LONG: 0, SHORT: 0, NEUTRAL: 0 };  // "Unproven" counts as neutral
  let training = 0;
  let downloading = 0;
  coins.forEach((c) => {
    const h = horizonData(c);
    if (h) counts[h.signal] = (counts[h.signal] || 0) + 1;
    if (c.training_status === "downloading") downloading++;
    else if (isBusy(c)) training++;
  });
  el.innerHTML = `${coins.length} ${coins.length === 1 ? "pair" : "pairs"} · `
    + `<span class="${counts.LONG ? "sum-pos" : ""}">${counts.LONG} long</span> · `
    + `<span class="${counts.SHORT ? "sum-neg" : ""}">${counts.SHORT} short</span> · `
    + `${counts.NEUTRAL} neutral at ${currentTimeframe}`
    + (downloading ? ` · ${downloading} downloading` : "")
    + (training ? ` · ${training} training` : "");
}

function renderAll() {
  const container = document.getElementById("coins-container");
  const symbols = Object.keys(coinsData);

  document.querySelectorAll("[data-horizon-label]").forEach((el) => {
    el.textContent = `${el.dataset.horizonLabel} · ${currentTimeframe}`;
  });

  if (symbols.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <h3>Your watchlist is empty</h3>
        <p>Add a CoinDCX futures pair to start streaming calibrated win probabilities.</p>
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
