// Number and text formatting shared by every component.

const SECTOR_LABELS: Record<string, string> = {
  MEME: "Meme",
  L1: "L1",
  DEFI: "DeFi",
  AI: "AI",
  NFT_GAMING: "NFT / Gaming",
  MAJOR: "Major",
  INFRA: "Infra",
  GENERAL_ALT: "Alt",
};

export function isNum(v: unknown): v is number {
  return v !== undefined && v !== null && !isNaN(v as number);
}

export function normalizeCoinDCXSymbol(raw: string): string {
  let s = (raw || "").trim();
  if (!s) return "";
  if (s.toLowerCase().includes("coindcx.com/")) {
    const parts = s.split("/");
    s = parts[parts.length - 1].split("?")[0].split("#")[0];
  }
  s = s.toUpperCase();
  if (s.startsWith("B-") || s.startsWith("I-")) return s;
  if (s.includes("_")) return `B-${s.replace(/^B-|^I-/, "")}`;
  if (s.endsWith("USDT") && s.length > 4) return `B-${s.slice(0, -4)}_USDT`;
  if (s.endsWith("INR") && s.length > 3) return `I-${s.slice(0, -3)}_INR`;
  return `B-${s}_USDT`;
}

export function splitSymbol(symbol: string): { base: string; quote: string } {
  const core = (symbol || "").replace(/^[BI]-/, "");
  const [base, quote] = core.split("_");
  return { base: base || core, quote: quote || "" };
}

export function sectorLabel(sector?: string | null): string {
  if (!sector) return "";
  return SECTOR_LABELS[sector] || sector.charAt(0) + sector.slice(1).toLowerCase().replace(/_/g, " ");
}

export function coindcxUrl(symbol: string): string {
  return symbol.startsWith("B-") ? "https://coindcx.com/futures/" + symbol : "https://coindcx.com/trade/" + symbol;
}

export function formatNumber(num: unknown, decimals = 4): string {
  if (!isNum(num)) return "—";
  return Number(num).toFixed(decimals);
}

export function signed(num: unknown, decimals = 2): string {
  if (!isNum(num)) return "—";
  const v = Number(num);
  const fixed = Math.abs(v).toFixed(decimals);
  if (Number(fixed) === 0) return fixed;
  return (v > 0 ? "+" : "−") + fixed;
}

// A signed percentage that is already in percent: 1.234 -> "+1.23%"
export function sp(v: unknown, decimals = 2): string {
  return isNum(v) ? `${signed(v, decimals)}%` : "—";
}

// A share 0–1 as a percentage: 0.5 -> "50%"
export function pct(p: unknown, decimals = 0): string {
  return isNum(p) ? `${(Number(p) * 100).toFixed(decimals)}%` : "—";
}

export function ordinal(n: unknown): string {
  if (!isNum(n)) return "—";
  const v = Math.round(n);
  const tail = v % 100 >= 11 && v % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[v % 10] || "th";
  return `${v}${tail}`;
}

// A percentile as an ordinal, with the ends said as "<1st" and ">99th" rather than "0th" and "100th"
export function pctile(p: unknown): string {
  if (!isNum(p)) return "—";
  if (p < 0.5) return "<1st";
  if (p >= 99.5) return ">99th";
  return ordinal(p);
}

export function pText(p: unknown): string {
  if (!isNum(p)) return "p —";
  return p < 0.001 ? "p < 0.001" : `p ${p < 0.01 ? p.toFixed(3) : p.toFixed(2)}`;
}

export function formatPrice(p: unknown): string {
  if (!p) return "—";
  const n = Number(p);
  if (n >= 100) return "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (n >= 1) return "$" + n.toFixed(4);
  return "$" + n.toLocaleString("en-US", { minimumSignificantDigits: 4, maximumSignificantDigits: 4 });
}

export function formatUsdCompact(v: unknown): string {
  if (!isNum(v) || !v) return "—";
  if (v >= 1e9) return "$" + (v / 1e9).toFixed(1) + "B";
  if (v >= 1e6) return "$" + (v / 1e6).toFixed(1) + "M";
  if (v >= 1e3) return "$" + Math.round(v / 1e3) + "k";
  return "$" + Math.round(v);
}

// Colour by sign, treating values that display as zero at `decimals` as flat
export function toneClass(v: unknown, decimals = 2): string {
  if (!isNum(v)) return "val-muted";
  if (Number(Math.abs(v).toFixed(decimals)) === 0) return "val-flat";
  return v > 0 ? "val-pos" : "val-neg";
}

function clockTime(iso?: string | null): string {
  return iso ? `${iso.slice(11, 16)} UTC` : "—";
}

// Clock time, with the date when it is not today (UTC)
export function whenText(iso?: string | null): string {
  if (!iso) return "—";
  const today = new Date().toISOString().slice(0, 10);
  if (iso.slice(0, 10) === today) return clockTime(iso);
  return `${new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" })} ${clockTime(iso)}`;
}

export function spanText(fromIso?: string | null): string {
  if (!fromIso) return "";
  const days = (Date.now() - new Date(fromIso).getTime()) / 86400000;
  return days >= 365 ? `${formatNumber(days / 365, 1)} years` : `${Math.round(days)} days`;
}
