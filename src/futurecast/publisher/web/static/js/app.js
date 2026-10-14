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
  FAVORS_LONG: { label: "Favors long", tag: "tag-pos" },
  FAVORS_SHORT: { label: "Favors short", tag: "tag-neg" },
  AVOID_BOTH_HIGH_RISK: { label: "Risky both ways", tag: "tag-warn" },
  BALANCED: { label: "Balanced", tag: "" },
};
const STATUS_INFO = {
  edge: { label: "Proven edge", tag: "tag-pos", note: "Beat the base rate on unseen data (90% confidence)." },
  no_edge: { label: "No edge", tag: "", note: "Did not beat the base rate on unseen data, so the recent base rate is shown." },
  insufficient_data: { label: "Too little data", tag: "tag-warn", note: "Too few independent test trades to judge, so the recent base rate is shown." },
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

function trainingNote(coin) {
  const status = coin.training_status;
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
    const busy = coin.training_status === "training" || coin.training_status === "queued";
    return `<span class="pill ${busy ? "pill-warn" : "pill-neutral"}">${busy ? "Training" : "No model"}</span>`;
  }
  if (h.signal === "LONG") return `<span class="pill pill-bull">Long</span>`;
  if (h.signal === "SHORT") return `<span class="pill pill-bear">Short</span>`;
  const side = unprovenSide(h);
  if (side) {
    return `<span class="pill pill-warn" data-tooltip-title="Unproven ${side}" data-tooltip="${esc(UNPROVEN_TIP)}">Unproven</span>`;
  }
  return `<span class="pill pill-neutral">Neutral</span>`;
}

function sideTooltip(side, d, h, horizon) {
  const name = side === "long" ? "Long" : "Short";
  const m = d.metrics || {};
  const lines = [
    `Take-profit ${side === "long" ? "+" : "−"}${formatNumber(h.tp_pct, 2)}% before stop-loss ${side === "long" ? "−" : "+"}${formatNumber(h.sl_pct, 2)}% within ${horizon}`,
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
      <div class="book-bar"><span style="width:${depth.bid_pct}%"></span></div>
      <div class="cell-sub${lean}">${depth.bid_pct}% bids</div>
    </div>`;
}

function riskCell(risk) {
  if (!isNum(risk.long_risk_score) || !isNum(risk.short_risk_score)) {
    return `<div class="c-risk"><span class="val-muted">—</span></div>`;
  }
  const lev = (risk.max_recommended_leverage || "").replace(/\s*\(.*\)\s*/, "").replace("-", "–");
  const bias = (BIAS_LABELS[risk.asymmetric_bias] || BIAS_LABELS.BALANCED).label;
  const tip = [
    `Long ${risk.long_risk_score}/100 · ${titleCase(risk.long_risk_level || "Low")}`,
    `Short ${risk.short_risk_score}/100 · ${titleCase(risk.short_risk_level || "Low")}`,
    risk.max_recommended_leverage ? `Max leverage ${risk.max_recommended_leverage}` : "",
    "Rule-based hazard scores, not probabilities",
  ].filter(Boolean).join("\n");
  return `
    <div class="c-risk" data-tooltip-title="Trade risk · ${esc(bias.toLowerCase())}" data-tooltip="${esc(tip)}">
      <span class="risk-pair">
        <span class="risk-num ${riskLevelClass(risk.long_risk_level)}"><small>L</small>${risk.long_risk_score}</span>
        <span class="risk-num ${riskLevelClass(risk.short_risk_level)}"><small>S</small>${risk.short_risk_score}</span>
      </span>
      ${lev ? `<div class="cell-sub">Max ${esc(lev)}</div>` : ""}
    </div>`;
}

// ───────── Expanded details ─────────

function statusTag(d) {
  const info = STATUS_INFO[d.status] || { label: titleCase(d.status), tag: "" };
  return `<span class="tag ${info.tag}">${info.label}</span>`;
}

function horizonsBlock(coin) {
  const rows = HORIZON_ORDER.filter((key) => horizonData(coin, key)).map((key) => {
    const h = horizonData(coin, key);
    const unproven = unprovenSide(h);
    const sig = h.signal === "LONG" ? `<span class="pill pill-sm pill-bull">Long</span>`
      : h.signal === "SHORT" ? `<span class="pill pill-sm pill-bear">Short</span>`
      : unproven ? `<span class="pill pill-sm pill-warn" data-tooltip-title="Unproven ${unproven}" data-tooltip="${esc(UNPROVEN_TIP)}">Unproven ${unproven}</span>`
      : `<span class="pill pill-sm pill-neutral">Neutral</span>`;
    const side = (s) => h[s]
      ? `<td class="num"><strong class="${h.signal === s.toUpperCase() ? (s === "long" ? "val-pos" : "val-neg") : (h[s].status === "edge" ? "" : "val-muted")}">${pct(h[s].probability)}</strong> <span class="val-muted">/ ${pct(h[s].break_even)}</span></td><td>${statusTag(h[s])}</td>`
      : `<td class="num">—</td><td></td>`;
    return `
      <tr class="${key === currentTimeframe ? "is-current" : ""}" onclick="setTimeframe('${key}')" title="Show ${key}">
        <td><strong>${key}</strong> <span class="val-muted">· ${esc(h.candle_interval)} candles</span></td>
        <td class="num">±${formatNumber(h.tp_pct, 2)}%</td>
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
    <section class="detail-block detail-wide detail-table">
      <h3>Win probability by horizon <span class="h3-note">probability / needed to break even</span></h3>
      <table class="data-table">
        <thead>
          <tr>
            <th>Horizon</th>
            <th class="num" data-tooltip-title="Take-profit and stop-loss" data-tooltip="Distance from entry, scaled to recent volatility and never closer than 2× the round-trip fees.">TP / SL</th>
            <th class="num">Long win</th><th>Long model</th>
            <th class="num">Short win</th><th>Short model</th>
            <th>Signal</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </section>`;
}

function reliabilityBars(rel) {
  if (!rel || !rel.length) return "";
  return `
    <div class="reliability" aria-label="Predicted versus observed win rate on unseen data">
      ${rel.map((b) => `
        <div class="rel-row">
          <span class="rel-label">said ${pct(b.predicted)}</span>
          <span class="rel-track"><span class="rel-fill" style="width:${Math.min(100, b.observed * 100)}%"></span><span class="rel-mark" style="left:${Math.min(100, b.predicted * 100)}%"></span></span>
          <span class="rel-value">won ${pct(b.observed)}</span>
        </div>`).join("")}
    </div>`;
}

function evidenceSide(coin, side, h) {
  const d = h[side];
  if (!d) return "";
  const m = d.metrics || {};
  const r = m.replay || {};
  const price = Number(coin.current_price);
  const up = side === "long";
  const tpPrice = isNum(price) && price ? formatPrice(price * (1 + (up ? 1 : -1) * h.tp_pct / 100)) : null;
  const slPrice = isNum(price) && price ? formatPrice(price * (1 + (up ? -1 : 1) * h.sl_pct / 100)) : null;
  const replay = r.trades
    ? `${r.trades} trades · ${pct(r.win_rate)} won · ${signed(r.avg_net_return_pct, 2)}% avg after fees`
    : "never cleared break-even";
  return `
    <div class="evidence-side">
      <div class="risk-side-head"><span>${up ? "Long" : "Short"} · ${pct(d.probability)}</span>${statusTag(d)}</div>
      <dl class="kv">
        ${tpPrice ? `<dt>Take-profit / stop</dt><dd>${tpPrice} / ${slPrice}</dd>` : ""}
        <dt>Needed to break even</dt><dd>${pct(d.break_even)}</dd>
        ${d.model_probability !== null && d.status !== "edge" ? `<dt>Model said (not trusted)</dt><dd class="val-muted">${pct(d.model_probability)}</dd>` : ""}
        <dt>Recent base rate</dt><dd>${pct(d.base_rate)}</dd>
        ${isNum(m.n_independent) ? `<dt>Independent test trades</dt><dd>${m.n_independent.toLocaleString()}</dd>` : ""}
        ${isNum(m.brier_skill) ? `<dt data-tooltip-title="Brier skill" data-tooltip="How much better the probabilities scored than always quoting the base rate. Above 0 is better; the lower bound is the 90% worst case.">Skill vs base rate</dt><dd>${signed(m.brier_skill, 3)} <span class="val-muted">(≥ ${signed(m.brier_skill_lower, 3)})</span></dd>` : ""}
        ${isNum(m.shrink) && m.shrink < 0.995 ? `<dt data-tooltip-title="Overconfidence correction" data-tooltip="On unseen data the raw odds ran too far from the usual win rate, so they are pulled back toward it by this much before being shown.">Odds tempered</dt><dd>${pct(1 - m.shrink)} toward base rate</dd>` : ""}
        ${isNum(m.auc) ? `<dt data-tooltip-title="AUC" data-tooltip="How well the model ranks winners above losers. 0.5 is a coin flip, 1.0 is perfect.">Ranking (AUC)</dt><dd>${formatNumber(m.auc, 2)}</dd>` : ""}
        ${m.replay ? `<dt data-tooltip-title="Replay on unseen data" data-tooltip="Take the trade every time the odds cleared break-even, one position at a time, on history the model never saw. Signals only fire for sides whose replay made money after fees on 10+ trades.">When it said go</dt><dd>${esc(replay)}</dd>` : ""}
      </dl>
      ${m.reason ? `<p class="detail-note">${esc(m.reason)}</p>` : ""}
      ${reliabilityBars(d.reliability)}
    </div>`;
}

function evidenceBlock(coin) {
  const h = horizonData(coin);
  if (!h) return "";
  return `
    <section class="detail-block detail-wide">
      <h3>Evidence · ${currentTimeframe} <span class="h3-note">${esc(h.candle_interval)} candles × ${h.horizon_bars} · tested on history the models never saw</span></h3>
      <div class="evidence-grid">
        ${evidenceSide(coin, "long", h)}
        ${evidenceSide(coin, "short", h)}
      </div>
    </section>`;
}

function bookBlock(depth) {
  if (!isNum(depth.bid_pct) || depth.available === false) {
    return `<section class="detail-block"><h3>Order book</h3><p class="detail-note">No live order book data for this pair yet.</p></section>`;
  }
  const imb = depth.book_imbalance_pct || 0;
  const tag = imb >= 12 ? `<span class="tag tag-pos">Bid-heavy</span>` : (imb <= -12 ? `<span class="tag tag-neg">Ask-heavy</span>` : `<span class="tag">Balanced</span>`);
  return `
    <section class="detail-block">
      <h3>Order book ${tag}</h3>
      <dl class="kv">
        <dt>Bid depth</dt><dd>${formatUsdCompact(depth.bid_depth_usd)} · ${depth.bid_pct}%</dd>
        <dt>Ask depth</dt><dd>${formatUsdCompact(depth.ask_depth_usd)} · ${depth.ask_pct}%</dd>
        <dt>Imbalance</dt><dd class="${toneClass(imb, 1)}">${signed(imb, 1)}%</dd>
        <dt>Spread</dt><dd>${formatNumber(depth.spread_bps, 1)} bps</dd>
        <dt>Microprice skew</dt><dd>${signed(depth.microprice_skew_bps, 1)} bps</dd>
      </dl>
      ${depth.summary ? `<p class="detail-note">${esc(depth.summary)}</p>` : ""}
    </section>`;
}

function riskBlock(risk) {
  if (!isNum(risk.long_risk_score) || !isNum(risk.short_risk_score)) {
    return `<section class="detail-block"><h3>Trade risk</h3><p class="detail-note">Risk scores are not available yet.</p></section>`;
  }
  const bias = BIAS_LABELS[risk.asymmetric_bias] || BIAS_LABELS.BALANCED;
  const side = (label, score, level, summary) => `
    <div class="risk-side">
      <div class="risk-side-head"><span>${label}</span><span class="risk-num ${riskLevelClass(level)}">${score} · ${esc(titleCase(level || "Low"))}</span></div>
      <div class="meter"><span class="${riskLevelClass(level)}" style="width:${score}%"></span></div>
      ${summary ? `<p>${esc(summary.replace(/^[A-Z]+ Risk:\s*/, ""))}</p>` : ""}
    </div>`;
  return `
    <section class="detail-block">
      <h3>Trade risk <span class="tag ${bias.tag}">${bias.label}</span></h3>
      ${side("Long", risk.long_risk_score, risk.long_risk_level, risk.long_risk_summary)}
      ${side("Short", risk.short_risk_score, risk.short_risk_level, risk.short_risk_summary)}
      <p class="detail-note">${risk.max_recommended_leverage ? `Max leverage <strong>${esc(risk.max_recommended_leverage)}</strong>. ` : ""}Rule-based hazard scores (0–100), not probabilities.</p>
    </section>`;
}

function residualBlock(coin) {
  const models = coin.model_predictions || {};
  const names = MODEL_ORDER.filter((m) => models[m]);
  if (!names.length) {
    return `<section class="detail-block"><h3>15m residual alpha</h3><p class="detail-note">Not trained on 5m candles yet.</p></section>`;
  }
  const rows = names.map((m) => {
    const d = models[m];
    return `<dt>${m}</dt><dd><span class="${toneClass(d.expected_residual_alpha_bps, 1)}">${signed(d.expected_residual_alpha_bps, 1)} bps</span>${isNum(d.hit_rate) ? ` <span class="val-muted">· ${pct(d.hit_rate)} hit</span>` : ""}</dd>`;
  }).join("");
  return `
    <section class="detail-block">
      <h3 data-tooltip-title="15m residual alpha" data-tooltip="Expected move versus a BTC-hedged benchmark over 15 minutes. Hit rates are from an unseen test slice. Secondary to the win probabilities.">15m residual alpha</h3>
      <dl class="kv">
        <dt>Ensemble</dt><dd class="${toneClass(coin.expected_residual_alpha_bps, 1)}">${signed(coin.expected_residual_alpha_bps, 1)} bps</dd>
        ${rows}
      </dl>
      <p class="detail-note">BTC-hedged expected move. Secondary signal; hit rates from unseen test data.</p>
    </section>`;
}

function detailsFooter(coin, symbol) {
  const benches = (coin.benchmarks || []).map((b) => splitSymbol(b).base).join(", ");
  let training;
  if (coin.training_status === "training" || coin.training_status === "queued") {
    training = `<span class="training-tag">${coin.training_status === "queued" ? "Queued for training…" : "Training on real candles…"}</span>`;
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
  if (coin.training_status === "training") subParts.push(`<span class="training-tag">Training</span>`);

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
      ${riskCell(coin.trade_risk || {})}

      <div class="c-actions">
        <button class="icon-btn btn-remove" onclick="event.stopPropagation(); removeCoin('${symbol}')" aria-label="Remove ${esc(base)}" title="Remove">${ICON_CLOSE}</button>
        <button class="icon-btn btn-expand" onclick="event.stopPropagation(); toggleCardDetails('${symbol}')"
                aria-expanded="${isExpanded}" aria-controls="${detailsId}" aria-label="${isExpanded ? "Hide" : "Show"} details for ${esc(base)}">${ICON_CHEVRON}</button>
      </div>
    </div>

    <div class="row-details" id="${detailsId}" ${isExpanded ? "" : "hidden"}>
      ${isExpanded ? `
        <div class="details-grid">
          ${horizonsBlock(coin)}
          ${evidenceBlock(coin)}
          ${riskBlock(coin.trade_risk || {})}
          ${bookBlock(coin.orderbook_depth || {})}
          ${residualBlock(coin)}
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
  coins.forEach((c) => {
    const h = horizonData(c);
    if (h) counts[h.signal] = (counts[h.signal] || 0) + 1;
    if (c.training_status === "training" || c.training_status === "queued") training++;
  });
  el.innerHTML = `${coins.length} ${coins.length === 1 ? "pair" : "pairs"} · `
    + `<span class="${counts.LONG ? "sum-pos" : ""}">${counts.LONG} long</span> · `
    + `<span class="${counts.SHORT ? "sum-neg" : ""}">${counts.SHORT} short</span> · `
    + `${counts.NEUTRAL} neutral at ${currentTimeframe}`
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
