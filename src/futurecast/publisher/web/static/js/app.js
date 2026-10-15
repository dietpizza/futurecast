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

// ───────── Training helpers ─────────

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

// ───────── Setup helpers ─────────

// The four checks of an A+ setup, in the strategy's order: [key, name (side, charts), what it means (charts)]
const CHECKLIST = [
  ["htf_trend", (up, c) => `${c.context} trend ${up ? "up" : "down"}`,
    (c) => `The ${c.context} chart's latest structure break was in the trade's direction.`],
  ["htf_zone", (up, c) => `Pulled into a ${c.context} ${up ? "bullish" : "bearish"} gap`,
    (c) => `The ${c.entry} pullback reached into a ${c.context} fair value gap that existed before price got there, and no ${c.context} candle has closed through it since.`],
  ["ltf_choch", (up, c) => `${c.entry} CHoCH ${up ? "up" : "down"}`,
    (c) => `A ${c.entry} close beyond the last swing point against the ${c.entry} trend: the short-term trend changed character.`],
  ["ltf_gap", (up, c) => `${c.entry} gap left behind`,
    (c) => `The move that broke structure left a ${c.entry} fair value gap. The limit order sits at its midpoint.`],
];
const STATE_LABELS = {
  waiting: "Waiting for fill", in_trade: "Filled", won: "Hit target", lost: "Stopped out",
  timeout: "Timed out", expired: "Expired", cancelled: "Cancelled",
};
const STATE_SHORT = {
  in_trade: "filled", won: "hit target", lost: "stopped", timeout: "timed out", expired: "expired", cancelled: "cancelled",
};
const MIN_SIGNAL_TRADES = 10;  // replayed trades a signal needs; mirrors min_signal_trades in models/setup_model.py

function setupsOf(coin) {
  return coin.setups || {};
}

function setupOf(coin, side) {
  return setupsOf(coin)[side] || null;
}

// The setup charts, e.g. { entry: "15m", context: "4h" }
function chartsOf(coin) {
  const st = setupsOf(coin);
  return { entry: st.entry_interval || "1m", context: st.context_interval || "15m" };
}

function isWaitingAPlus(s) {
  return Boolean(s && s.state === "waiting" && s.grade === "A+");
}

// How long ago, from a count of candles of `minutes` each
function agoText(candles, minutes = 1) {
  if (!isNum(candles)) return "";
  const mins = candles * minutes;
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  if (mins < 2880) return `${Math.round(mins / 60)} h ago`;
  return `${Math.round(mins / 1440)} d ago`;
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

function trendText(trend) {
  return trend === "up" ? "Up" : trend === "down" ? "Down" : "None";
}

function trendClass(trend) {
  return trend === "up" ? "val-pos" : trend === "down" ? "val-neg" : "val-muted";
}

function intervalMinutes(interval) {
  const unit = String(interval || "").slice(-1);
  return (parseInt(interval, 10) || 1) * (unit === "d" ? 1440 : unit === "h" ? 60 : 1);
}

// The test evidence behind a setup's target: the models' replay when they have an edge, else the rules'
function targetEvidence(coin, targetR) {
  const ev = coin.setup_model;
  return ev && ev.targets ? ev.targets[String(Number(targetR))] || null : null;
}

function servedReplay(t) {
  return t ? (t.status === "edge" ? t.model_replay : t.rules_replay) || {} : {};
}

function recordText(r) {
  const n = (r && r.trades) || 0;
  return n < MIN_SIGNAL_TRADES
    ? `only ${n} past A+ ${n === 1 ? "trade" : "trades"} on unseen history to judge by (needs ${MIN_SIGNAL_TRADES})`
    : `past A+ trades on unseen history averaged ${signed(r.avg_net_r, 2)}R after fees`;
}

// SetupSpec.costs() for a heading note: "after fees", with what a trade pays per leg in the tooltip
// (nothing from a server that predates per-leg fees)
function feesNote(c) {
  if (!c) return "";
  const p = (v) => `${+Number(v).toFixed(4)}%`;
  const tip = `A win pays ${p(c.won_pct)} round trip, a stop ${p(c.lost_pct)}. The limit entry and a limit take-profit pay the ${p(c.maker_pct)} maker fee; the stop is a market order: the ${p(c.taker_pct)} taker fee plus ${p(c.slippage_pct)} slippage. A timeout exit pays the taker fee, ${p(c.timeout_pct)} round trip.`;
  return ` · <span data-tooltip-title="Fees per leg" data-tooltip="${esc(tip)}">after fees</span>`;
}

// What a side's latest setup means now. Mirrors the signal rule in engine/setups.py: an A+ setup whose
// limit order is still waiting, whose served win chance clears break-even and whose replay made money.
function setupVerdict(coin, side) {
  const s = setupOf(coin, side);
  const up = side === "long";
  const charts = chartsOf(coin);
  const ago = s ? agoText(s.candles_ago, intervalMinutes(charts.entry)) : "";
  if (!s) {
    return { state: "idle", label: "No setup", pill: "pill-neutral", why: `No ${charts.entry} change of character ${up ? "up" : "down"} with a gap behind it in the candles on hand.` };
  }
  const chance = isNum(s.probability)
    ? `${pct(s.probability)} win chance at ${s.target_r}R, ${pct(s.break_even)} needed after fees`
    : null;
  if (s.state !== "waiting") {
    const what = {
      in_trade: `the limit filled at ${whenText(s.filled_at)}; the trade runs until the stop or ${s.target_r}R.`,
      won: `filled, then reached ${s.target_r}R.`,
      lost: "filled, then hit the stop.",
      timeout: "filled, but reached neither level in time.",
      expired: "price never came back to the limit.",
      cancelled: `price reached ${s.target_r}R before coming back to the limit.`,
    }[s.state] || "";
    return { state: "idle", label: STATE_LABELS[s.state] || titleCase(s.state), pill: "pill-neutral", why: `${s.grade} setup ${ago}: ${what}` };
  }
  if (setupsOf(coin).signal === side.toUpperCase()) {
    return { state: `go-${side}`, label: `${titleCase(side)} signal`, pill: up ? "pill-bull" : "pill-bear", why: `A+ setup ${ago}: ${chance}. The limit waits until ${whenText(s.order_expires)}.` };
  }
  if (s.grade !== "A+") {
    const missing = CHECKLIST.filter(([key]) => !s.checklist[key]).map(([, name]) => name(up, charts).toLowerCase()).join(" and ");
    return { state: "idle", label: "B setup", pill: "pill-neutral", why: `Limit waiting until ${whenText(s.order_expires)}, but missing ${missing}. Only A+ setups raise a signal.` };
  }
  let why;
  if (!chance) why = "The rules found it. Train the models to see its win chance and its track record.";
  else if (s.probability < s.break_even) why = `${chance}: below break-even.`;
  else if (!s.validated) why = `${chance}, but ${recordText(servedReplay(targetEvidence(coin, s.target_r)))}.`;
  else why = `${chance}, but the ${up ? "short" : "long"} setup is stronger.`;
  return { state: "unproven", label: "A+ setup", pill: "pill-warn", why };
}

function signalPill(coin) {
  if (!coin.has_data) return `<span class="pill pill-neutral">Loading</span>`;
  const signal = setupsOf(coin).signal;
  if (signal === "LONG") return `<span class="pill pill-bull">Long</span>`;
  if (signal === "SHORT") return `<span class="pill pill-bear">Short</span>`;
  const side = ["long", "short"].find((s) => isWaitingAPlus(setupOf(coin, s)));
  if (side) {
    const v = setupVerdict(coin, side);
    return `<span class="pill pill-warn" data-tooltip-title="A+ ${side} setup, no signal" data-tooltip="${esc(v.why)}">A+ setup</span>`;
  }
  if (!coin.setup_model) {
    const busy = isBusy(coin);
    return `<span class="pill ${busy ? "pill-warn" : "pill-neutral"}">${busy ? trainingVerb(coin) : "No model"}</span>`;
  }
  return `<span class="pill pill-neutral">Neutral</span>`;
}

function levelsLine(s) {
  return `Limit ${formatPrice(s.entry)} · stop ${formatPrice(s.stop)} · ${s.target_r}R ${formatPrice(s.take_profit)}`;
}

function setupCell(coin, side) {
  const cls = `c-${side} num`;
  if (!coin.has_data) return `<div class="${cls}"><span class="val val-muted">—</span></div>`;
  const s = setupOf(coin, side);
  if (!s) return `<div class="${cls}"><span class="val val-muted">—</span><div class="cell-sub">no setup</div></div>`;
  const v = setupVerdict(coin, side);
  const live = s.state === "waiting";
  const tone = v.state.startsWith("go-") ? (side === "long" ? "val-pos" : "val-neg")
    : (live && s.grade === "A+" ? "val-flat" : "val-muted");
  const main = live && isNum(s.probability) ? pct(s.probability) : s.grade;
  const sub = live
    ? (isNum(s.probability) ? `${s.grade} · needs ${pct(s.break_even)}` : "waiting")
    : `${STATE_SHORT[s.state] || s.state} · ${agoText(s.candles_ago, intervalMinutes(chartsOf(coin).entry)).replace(" ago", "")}`;
  return `
    <div class="${cls}" data-tooltip-title="${esc(`${titleCase(side)} ${s.grade} setup · ${STATE_LABELS[s.state] || s.state}`)}" data-tooltip="${esc(`${v.why}\n${levelsLine(s)}`)}">
      <span class="val val-lg ${tone}">${esc(main)}</span>
      <div class="cell-sub">${esc(sub)}</div>
    </div>`;
}

function structureLines(v, label) {
  const lb = v.last_break;
  const minutes = intervalMinutes(v.interval);
  return [
    `${label}: trend ${trendText(v.trend).toLowerCase()}`
      + (lb ? `, last ${lb.kind} ${lb.direction} through ${formatPrice(lb.level)} ${agoText(lb.candles_ago, minutes)}` : ", no break yet"),
    `  swing high ${formatPrice(v.swing_high)} · swing low ${formatPrice(v.swing_low)} · ${(v.gaps || []).length} open gaps nearby`,
  ];
}

function trendCell(coin) {
  const st = coin.structure || {};
  if (!st.htf || !st.ltf) return `<div class="c-trend"><span class="val-muted">—</span></div>`;
  const inside = (st.htf.gaps || []).find((g) => g.contains_price);
  const sub = inside ? `in ${st.htf.interval} ${inside.direction} gap` : `${(st.htf.gaps || []).length} ${st.htf.interval} gaps near`;
  const tip = [...structureLines(st.htf, st.htf.interval), ...structureLines(st.ltf, st.ltf.interval)].join("\n");
  return `
    <div class="c-trend" data-tooltip-title="Market structure" data-tooltip="${esc(tip)}">
      <span class="trend-pair">
        <span><small>${esc(st.htf.interval)}</small> <span class="${trendClass(st.htf.trend)}">${trendText(st.htf.trend)}</span></span>
        <span><small>${esc(st.ltf.interval)}</small> <span class="${trendClass(st.ltf.trend)}">${trendText(st.ltf.trend)}</span></span>
      </span>
      <div class="cell-sub">${esc(sub)}</div>
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

function worstMoveText(side, d) {
  return `${side === "long" ? "−" : "+"}${formatNumber(d.worst_move_pct, 2)}%`;
}

function riskTip(r) {
  const line = (side, name) => {
    const d = r[side];
    return `${name} ${worstMoveText(side, d)} worst case · riskier than ${d.score}% of recent windows · max ${d.max_leverage}x`;
  };
  const notes = [...new Set(["long", "short"].map((side) => (RISK_STATUS[r[side].status] || {}).note).filter(Boolean))];
  return [line("long", "Long"), line("short", "Short"), ...regimeLines(r), ...notes].join("\n");
}

function riskCell(coin) {
  if (!coin.has_data) return `<div class="c-risk"><span class="val-muted">—</span></div>`;
  const r = riskData(coin);
  if (!r || !r.long || !r.short) {
    return `<div class="c-risk"><span class="val-muted">—</span><div class="cell-sub">${esc(trainingNote(coin))}</div></div>`;
  }
  const bias = (BIAS_LABELS[r.bias] || BIAS_LABELS.BALANCED).label;
  return `
    <div class="c-risk" data-tooltip-title="Worst 5% of moves within ${esc(currentTimeframe)} · ${esc(bias.toLowerCase())}" data-tooltip="${esc(riskTip(r))}">
      <span class="risk-pair">
        <span class="risk-num ${riskLevelClass(r.long.level)}"><small>L</small>${r.long.score}</span>
        <span class="risk-num ${riskLevelClass(r.short.level)}"><small>S</small>${r.short.score}</span>
      </span>
      <div class="cell-sub">Max ${r.long.max_leverage}x / ${r.short.max_leverage}x</div>
    </div>`;
}

// ───────── Expanded details ─────────
//
// Only what decides a trade, in this order: the live A+ setup's orders (or what the next one needs),
// whether trading every A+ setup made money on this coin, and the levels the next setup comes from.

function tradeStat(label, value, sub = "", tip = null, valueClass = "") {
  const attrs = tip ? ` data-tooltip-title="${esc(tip.title)}" data-tooltip="${esc(tip.body)}"` : "";
  return `
    <div class="trade-stat"${attrs}>
      <span class="trade-stat-label">${label}</span>
      <span class="trade-stat-value ${valueClass}">${value}</span>
      ${sub ? `<span class="trade-stat-sub">${sub}</span>` : ""}
    </div>`;
}

function targetsTip(s, up) {
  const rows = (s.targets || []).map((t) => {
    const mark = t.target_r === s.target_r ? "▸ " : "";
    const odds = isNum(t.probability)
      ? ` · ${pct(t.probability)} win (${t.probability_source === "model" ? "models" : "rules' win rate"}), needs ${pct(t.break_even)} · ${signed(t.expected_r, 2)}R expected`
      : "";
    return `${mark}${t.target_r}R at ${formatPrice(t.price)} (${up ? "+" : "−"}${formatNumber(s.risk_pct * t.target_r, 2)}%)${odds}`;
  });
  const how = isNum(s.probability)
    ? "The target with the highest expected result after fees is picked."
    : "Train the models to see each target's win chance.";
  return { title: "Take-profit · multiples of the risk (R)", body: [how, ...rows].join("\n") };
}

// A+ setups the strategy would act on now: a limit still waiting, or a filled trade still running
function liveSides(coin) {
  return ["long", "short"].filter((side) => {
    const s = setupOf(coin, side);
    return s && s.grade === "A+" && (s.state === "waiting" || s.state === "in_trade");
  });
}

function liveTrade(coin, side) {
  const s = setupOf(coin, side);
  const up = side === "long";
  const charts = chartsOf(coin);
  const filled = s.state === "in_trade";
  const v = filled
    ? { state: "idle", label: "In trade", pill: "pill-neutral", why: `Filled ${whenText(s.filled_at)}. Leave it: the stop or the ${s.target_r}R target closes it.` }
    : setupVerdict(coin, side);
  const checks = CHECKLIST.map(([, name]) => `✓ ${name(up, charts)}`).join("\n");
  const price = Number(coin.current_price);
  const away = isNum(price) && price ? `${signed((s.entry / price - 1) * 100, 2)}% from price` : "";
  const entrySub = filled ? `filled ${whenText(s.filled_at)}` : away;
  const entryTip = {
    title: "Entry · limit order",
    body: `Midpoint of the ${charts.entry} fair value gap ${formatPrice(s.gap.bottom)} – ${formatPrice(s.gap.top)}. The order waits until ${whenText(s.order_expires)} and is cancelled if price reaches the first target before filling.`,
  };
  const stopTip = {
    title: "Stop",
    body: `Just ${up ? "below the low" : "above the high"} of the pullback before the ${charts.entry} change of character through ${formatPrice(s.choch_level)}. Its distance from the entry is the risk, 1R.`,
  };
  const chanceTip = {
    title: "Win chance once filled",
    body: isNum(s.probability)
      ? [`Chance the ${s.target_r}R target is reached before the stop.`,
        s.probability_source === "model" ? "From the models, which beat the rules on setups they never saw." : "The rules' own win rate for A+ setups: the models did not beat it.",
        `Needs ${pct(s.break_even)}: the win rate at which a trade makes nothing after fees.`].join("\n")
      : "Appears once the models are trained on this coin's history.",
  };
  const sizeTip = {
    title: "Position size for 1% account risk",
    body: `1% of the account divided by the ${formatNumber(s.risk_pct, 2)}% stop distance: a position worth ${formatNumber(s.size_for_1pct_risk, 2)}× the account (a ${formatUsdCompact(1000 * s.size_for_1pct_risk)} position risks $10 of a $1,000 account).\nMax ${s.max_leverage}x: the highest of 1, 2, 3, 5, 10 or 20x whose liquidation price stays beyond the stop.`,
  };
  const clears = isNum(s.probability) && s.probability >= s.break_even;
  return `
    <div class="trade-side is-${v.state}">
      <div class="trade-head">
        <span>${up ? "Long" : "Short"}</span>
        <span class="pill ${v.pill}" data-tooltip-title="A+ setup · all four checks pass" data-tooltip="${esc(checks)}">${esc(v.label)}</span>
      </div>
      <p class="trade-why">${esc(v.why)}</p>
      <div class="trade-stats">
        ${tradeStat("Entry · limit", formatPrice(s.entry), esc(entrySub), entryTip)}
        ${tradeStat("Stop", formatPrice(s.stop), `${up ? "−" : "+"}${formatNumber(s.risk_pct, 2)}% · 1R`, stopTip, "is-sl")}
        ${tradeStat("Target", formatPrice(s.take_profit), `${up ? "+" : "−"}${formatNumber(s.risk_pct * s.target_r, 2)}% · ${s.target_r}R`, targetsTip(s, up), "is-tp")}
        ${tradeStat("Win chance", isNum(s.probability) ? pct(s.probability) : "—", isNum(s.probability) ? `needs ${pct(s.break_even)}` : esc(trainingNote(coin)), chanceTip, clears ? "is-tp" : "")}
        ${tradeStat("Size · × account", `${formatNumber(s.size_for_1pct_risk, 1)}×`, `1% risk · max ${s.max_leverage}x`, sizeTip)}
      </div>
    </div>`;
}

// No live A+ setup: what the next one needs, from the context chart's trend and its nearest gap
function waitingFor(coin) {
  const ctx = (coin.structure || {}).htf;
  const charts = chartsOf(coin);
  const price = Number(coin.current_price);
  if (!ctx) return "Waiting for candles.";
  if (ctx.trend !== "up" && ctx.trend !== "down") return `No ${ctx.interval} trend yet: waiting for a ${ctx.interval} break of structure.`;
  const up = ctx.trend === "up";
  const dir = up ? "bullish" : "bearish";
  const flip = `a ${charts.entry} change of character ${up ? "up" : "down"}`;
  const zone = (ctx.gaps || [])
    .filter((g) => g.direction === dir && (g.contains_price || !isNum(price) || (up ? g.top <= price : g.bottom >= price)))
    .sort((a, b) => Math.abs(a.mid - price) - Math.abs(b.mid - price))[0];
  const trend = `${ctx.interval} trend ${up ? "up" : "down"}`;
  if (!zone) return `${trend}, but no open ${ctx.interval} ${dir} gap ${up ? "below" : "above"} price to pull back into.`;
  const range = `${formatPrice(zone.bottom)} – ${formatPrice(zone.top)}`;
  if (zone.contains_price) return `${trend}, price is inside the ${ctx.interval} ${dir} gap ${range}: waiting for ${flip}.`;
  const edge = up ? zone.top : zone.bottom;
  const away = isNum(price) && price ? ` (${formatNumber(Math.abs(edge / price - 1) * 100, 1)}% away)` : "";
  return `${trend}: waiting for a pullback into the ${ctx.interval} ${dir} gap ${range}${away}, then ${flip}.`;
}

// The latest setup of each side the trade panel is not showing, in a few words
function lastSetups(coin, shown) {
  const minutes = intervalMinutes(chartsOf(coin).entry);
  return ["long", "short"].filter((side) => !shown.includes(side)).map((side) => {
    const s = setupOf(coin, side);
    if (!s) return null;
    const what = s.state === "waiting" ? "limit waiting" : (STATE_SHORT[s.state] || s.state);
    return `${side} ${s.grade}, ${what} ${agoText(s.candles_ago, minutes)}`;
  }).filter(Boolean).join(" · ");
}

function tradeBlock(coin) {
  const st = setupsOf(coin);
  if (!coin.has_data || !st.as_of) return "";
  const charts = chartsOf(coin);
  const live = liveSides(coin);
  const body = live.length
    ? `<div class="sides-grid${live.length === 1 ? " is-single" : ""}">${live.map((side) => liveTrade(coin, side)).join("")}</div>`
    : `<div class="trade-side is-idle">
        <div class="trade-head"><span>No trade</span></div>
        <p class="trade-why">${esc(waitingFor(coin))}</p>
      </div>`;
  const last = lastSetups(coin, live);
  return `
    <section class="detail-block detail-wide trade-block">
      <h3>Trade · ${esc(charts.context)} context, ${esc(charts.entry)} entry <span class="h3-note">as of ${whenText(st.as_of)}</span></h3>
      ${body}
      ${last ? `<p class="trade-foot">Last setups: ${esc(last)}</p>` : ""}
    </section>`;
}

// Account equity after each backtested trade, from 100; the dashed line is where it started
function equitySvg(points) {
  if (!points || points.length < 2) return "";
  const w = 160, h = 28, pad = 2;
  const eq = points.map((p) => p.equity);
  const lo = Math.min(100, ...eq), hi = Math.max(100, ...eq);
  const y = (v) => (pad + (hi - v) / ((hi - lo) || 1) * (h - 2 * pad)).toFixed(1);
  const line = eq.map((v, i) => `${(i / (eq.length - 1) * w).toFixed(1)},${y(v)}`).join(" ");
  const last = eq[eq.length - 1];
  return `
    <svg class="equity" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img"
         aria-label="Account from 100 to ${formatNumber(last, 1)}">
      <line class="equity-base" x1="0" x2="${w}" y1="${y(100)}" y2="${y(100)}" />
      <polyline class="equity-line ${last >= 100 ? "is-up" : "is-down"}" points="${line}" />
    </svg>`;
}

function backtestRow(bt) {
  const s = bt.summary || {};
  if (!s.trades) return `<tr><td><strong>${s.target_r}R</strong></td><td colspan="6" class="val-muted">No filled A+ trades</td></tr>`;
  return `
    <tr>
      <td><strong>${s.target_r}R</strong></td>
      <td class="num">${s.trades}</td>
      <td class="num">${pct(s.win_rate, 1)} <span class="val-muted">/ ${pct(s.break_even_win_rate, 1)}</span></td>
      <td class="num"><strong class="bt-avg ${toneClass(s.avg_net_r)}">${signed(s.avg_net_r, 2)}R</strong></td>
      <td class="num ${toneClass(s.total_return_pct, 1)}">${signed(s.total_return_pct, 1)}%</td>
      <td class="num">${formatNumber(s.max_drawdown_pct, 1)}%</td>
      <td class="spark">${equitySvg(bt.equity)}</td>
    </tr>`;
}

// Every A+ setup on the training history, taken one at a time: does the strategy make money on this coin?
function backtestBlock(coin) {
  const ev = coin.setup_model;
  const bts = ev && ev.backtests ? Object.values(ev.backtests) : [];
  if (!bts.length) return "";
  const s = bts[0].summary || {};
  const best = Math.max(...bts.map((bt) => (bt.summary && bt.summary.trades ? bt.summary.avg_net_r : -Infinity)));
  const verdict = !isFinite(best) ? ["No trades", ""]
    : best > 0.05 ? ["Made money", "tag-pos"] : best < -0.05 ? ["Lost money", "tag-neg"] : ["Break-even", "tag-warn"];
  const years = s.days >= 365 ? `${formatNumber(s.days / 365, 1)} years` : `${Math.round(s.days || 0)} days`;
  const few = bts.some((bt) => bt.summary && bt.summary.trades && bt.summary.trades < 100)
    ? `<p class="trade-foot">Under 100 trades: treat these rates as rough.</p>` : "";
  return `
    <section class="detail-block detail-wide detail-table">
      <h3>Backtest · every A+ setup <span class="h3-note">${years} of ${esc(s.entry_interval)} candles · 1 at a time · ${s.risk_pct}% risk each${feesNote(s.costs)}</span> <span class="tag ${verdict[1]}">${verdict[0]}</span></h3>
      <table class="data-table">
        <thead>
          <tr>
            <th>Target</th><th class="num">Trades</th>
            <th class="num" data-tooltip-title="Won / needed" data-tooltip="Share of filled trades that reached the target first, and the win rate needed to break even after fees.">Won / needed</th>
            <th class="num" data-tooltip-title="Average trade" data-tooltip="Average result per trade in multiples of the risk (R), after fees.">Avg trade</th>
            <th class="num" data-tooltip-title="Account" data-tooltip="The account's change over the whole history, compounding.">Account</th>
            <th class="num" data-tooltip-title="Max drawdown" data-tooltip="Largest fall of the account from a high.">Max drawdown</th>
            <th>Equity</th>
          </tr>
        </thead>
        <tbody>${bts.map(backtestRow).join("")}</tbody>
      </table>
      ${few}
    </section>`;
}

function breakText(v) {
  const lb = v.last_break;
  return lb ? `${lb.kind} ${lb.direction} through ${formatPrice(lb.level)}, ${agoText(lb.candles_ago, intervalMinutes(v.interval))}` : "no break yet";
}

// The levels the next setup comes from: both charts' trend and last break, the context chart's nearest gaps
function levelsBlock(coin) {
  const st = coin.structure || {};
  if (!st.htf || !st.ltf) return "";
  const price = Number(coin.current_price);
  const nearest = (dir) => (st.htf.gaps || []).filter((g) => g.direction === dir)
    .sort((a, b) => Math.abs(a.mid - price) - Math.abs(b.mid - price))[0];
  const gaps = [["bullish", "▲", "val-pos"], ["bearish", "▼", "val-neg"]].map(([dir, mark, cls]) => {
    const g = nearest(dir);
    return g ? `<span class="${cls}">${mark} ${formatPrice(g.bottom)} – ${formatPrice(g.top)}</span>${g.contains_price ? " <span class=\"val-muted\">price inside</span>" : ""}` : "";
  }).filter(Boolean).join(" &nbsp; ");
  const chart = (v, role) => `
      <li><span class="lvl-chart">${esc(v.interval)} <small>${role}</small></span>
        <span><strong class="${trendClass(v.trend)}">${trendText(v.trend)}</strong> <span class="val-muted">· ${esc(breakText(v))}</span></span></li>`;
  return `
    <section class="detail-block detail-wide detail-quiet">
      <h3>Levels</h3>
      <ul class="levels">
        ${chart(st.htf, "context")}
        ${chart(st.ltf, "entry")}
        ${gaps ? `<li><span class="lvl-chart" data-tooltip-title="Fair value gaps" data-tooltip="The nearest open ${esc(st.htf.interval)} gaps: where a pullback has to reach for an A+ setup. A gap stays open until a candle closes through its far side.">${esc(st.htf.interval)} <small>gaps</small></span><span>${gaps}</span></li>` : ""}
      </ul>
    </section>`;
}

function detailsFooter(coin, symbol) {
  let training;
  if (isBusy(coin)) {
    const p = coin.training_progress;
    const bar = p && isNum(p.fraction) && coin.training_status !== "queued"
      ? `<span class="progress-track" aria-hidden="true"><span class="progress-fill" style="width:${Math.round(p.fraction * 100)}%"></span></span>`
      : "";
    training = `<span class="training-tag">${esc(trainingDetail(coin))}</span>${bar}`;
  } else {
    const label = coin.is_custom_trained ? "Retrain" : "Train models";
    const error = coin.training_error ? `<span class="val-neg">${esc(coin.training_error)}</span>` : "";
    training = `<button class="btn btn-sm" onclick="trainCoin('${symbol}')">${label}</button>${error}`;
  }
  return `
    <div class="details-footer">
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

      ${setupCell(coin, "long")}
      ${setupCell(coin, "short")}
      ${trendCell(coin)}
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
          ${backtestBlock(coin)}
          ${levelsBlock(coin)}
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
  let waiting = 0;  // A+ setups whose limit is still waiting, signal or not
  let training = 0;
  let downloading = 0;
  coins.forEach((c) => {
    const signal = setupsOf(c).signal;
    if (signal in counts) counts[signal]++;
    waiting += ["long", "short"].filter((side) => isWaitingAPlus(setupOf(c, side))).length;
    if (c.training_status === "downloading") downloading++;
    else if (isBusy(c)) training++;
  });
  el.innerHTML = `${coins.length} ${coins.length === 1 ? "pair" : "pairs"} · `
    + `<span class="${counts.LONG ? "sum-pos" : ""}">${counts.LONG} long</span> · `
    + `<span class="${counts.SHORT ? "sum-neg" : ""}">${counts.SHORT} short</span> · `
    + `${waiting} A+ ${waiting === 1 ? "setup" : "setups"} waiting`
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
        <p>Add a CoinDCX futures pair to start watching its market structure, fair value gaps and setups.</p>
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
