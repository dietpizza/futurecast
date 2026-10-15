"""Exchange detection and symbol resolution utilities."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def detect_exchange(symbol: str) -> str:
    """Auto-detect exchange based on symbol naming patterns.
    
    CoinDCX pairs typically have prefixes like 'B-' (Binance-backed derivatives),
    'I-' (Indian Rupee pairs), or contain underscores (e.g., 'B-DOGE_USDT', 'I-BTC_INR').
    Binance pairs typically follow 'DOGEUSDT', 'BTCUSDT'.
    """
    s = symbol.strip().upper()
    if s.startswith("B-") or s.startswith("I-") or "_" in s:
        return "coindcx"
    return "binance"


# Recognized crypto sectors and their category leaders
SECTOR_MAP: Dict[str, str] = {
    # Meme Coins -> Sector Leader: DOGE
    "DOGE": "MEME",
    "SHIB": "MEME",
    "PEPE": "MEME",
    "1000PEPE": "MEME",
    "FLOKI": "MEME",
    "1000FLOKI": "MEME",
    "BONK": "MEME",
    "1000BONK": "MEME",
    "WIF": "MEME",
    "MEME": "MEME",
    "POPCAT": "MEME",
    "MOG": "MEME",
    "NEIRO": "MEME",
    "BOME": "MEME",
    "MEW": "MEME",
    "BRETT": "MEME",
    "MYRO": "MEME",
    "TURBO": "MEME",

    # Layer 1 / Alternative L1s -> Sector Leader: SOL
    "SOL": "L1",
    "AVAX": "L1",
    "SUI": "L1",
    "APT": "L1",
    "ADA": "L1",
    "DOT": "L1",
    "FTM": "L1",
    "ATOM": "L1",
    "SEI": "L1",
    "TON": "L1",
    "INJ": "L1",
    "TIA": "L1",
    "KAS": "L1",
    "ALGO": "L1",
    "HBAR": "L1",
    "NEAR": "L1",
    "TRX": "L1",
    "ZK": "L1",
    "ARB": "L1",
    "OP": "L1",
    "STRK": "L1",

    # DeFi (Decentralized Finance) -> Sector Leader: ETH
    "ETH": "DEFI",
    "UNI": "DEFI",
    "AAVE": "DEFI",
    "CRV": "DEFI",
    "MKR": "DEFI",
    "SNX": "DEFI",
    "COMP": "DEFI",
    "LDO": "DEFI",
    "PENDLE": "DEFI",
    "JUP": "DEFI",
    "RAY": "DEFI",
    "SUSHI": "DEFI",
    "1INCH": "DEFI",
    "DYDX": "DEFI",
    "GMX": "DEFI",
    "CAKE": "DEFI",
    "RUNE": "DEFI",

    # NFT / Gaming / Metaverse -> Sector Leader: BLUR
    "BLUR": "NFT_GAMING",
    "SAND": "NFT_GAMING",
    "MANA": "NFT_GAMING",
    "AXS": "NFT_GAMING",
    "GALA": "NFT_GAMING",
    "APE": "NFT_GAMING",
    "ENJ": "NFT_GAMING",
    "IMX": "NFT_GAMING",
    "BEAM": "NFT_GAMING",
    "SUPER": "NFT_GAMING",

    # AI / Compute / DePIN -> Sector Leader: NEAR
    "RENDER": "AI",
    "FET": "AI",
    "TAO": "AI",
    "AGIX": "AI",
    "AR": "AI",
    "AKT": "AI",
    "WLD": "AI",
    "ICP": "AI",
    "GRT": "AI",
    "THETA": "AI",

    # Infrastructure / Oracle
    "LINK": "INFRA",
    "PYTH": "INFRA",
    "FIL": "INFRA",
    "AVA": "INFRA",

    # Majors
    "BTC": "MAJOR",
}

SECTOR_LEADERS: Dict[str, str] = {
    "MEME": "DOGE",
    "L1": "SOL",
    "DEFI": "ETH",
    "NFT_GAMING": "BLUR",
    "AI": "NEAR",
    "INFRA": "LINK",
    "MAJOR": "BTC",
}


def extract_base_and_quote(symbol: str) -> Tuple[str, str]:
    """Extract base asset and quote currency from an exchange symbol.
    
    Examples:
        'B-DOGE_USDT' -> ('DOGE', 'USDT')
        'I-BTC_INR'   -> ('BTC', 'INR')
        'DOGEUSDT'    -> ('DOGE', 'USDT')
        'BTCUSDT'     -> ('BTC', 'USDT')
    """
    s = symbol.strip().upper()
    if "_" in s:
        parts = s.split("_")
        quote = parts[1]
        base = parts[0]
        if base.startswith("B-") or base.startswith("I-"):
            base = base[2:]
        return base, quote

    for q in ["USDT", "USDC", "FDUSD", "BUSD", "TUSD", "EUR", "TRY", "INR", "BTC", "ETH"]:
        if s.endswith(q) and len(s) > len(q):
            base = s[:-len(q)]
            return base, q

    return s, "USDT"


def format_benchmark_symbol(base: str, quote: str, exchange: str) -> str:
    """Format a benchmark asset into an exchange-compliant symbol."""
    ex = exchange.lower().strip()
    if ex == "coindcx":
        if quote == "INR":
            return f"I-{base}_INR"
        return f"B-{base}_{quote}"
    else:
        return f"{base}{quote}"


def get_symbol_sector(symbol: str) -> str:
    """Return the market sector category for a symbol (e.g. MEME, L1, DEFI, AI, etc.)."""
    base, _ = extract_base_and_quote(symbol)
    return SECTOR_MAP.get(base, "GENERAL_ALT")


def resolve_benchmarks(
    target_symbol: str,
    exchange: Optional[str] = None,
    sector_aware: bool = True,
) -> List[str]:
    """Determine the appropriate benchmark assets for the given target token.
    
    When sector_aware=True, dynamically routes category leaders:
    - Meme coins (e.g. SHIB, PEPE, WIF) -> [DOGE, BTC]
    - Alt-L1s (e.g. AVAX, SUI, APT) -> [SOL, BTC]
    - DeFi tokens (e.g. UNI, AAVE, CRV) -> [ETH, BTC]
    - NFT/Gaming tokens (e.g. SAND, AXS) -> [BLUR, BTC]
    - AI tokens (e.g. RENDER, FET, TAO) -> [NEAR, BTC]
    - Category leaders themselves (e.g. DOGE, SOL, BLUR) -> [BTC, ETH]

    A coin is never its own benchmark: BTC -> [ETH, SOL] and ETH -> [BTC, SOL]. Benchmarking
    a coin against itself would duplicate its columns and break feature generation.
    """
    ex = exchange or detect_exchange(target_symbol)
    base, quote = extract_base_and_quote(target_symbol)

    preferred = ["BTC", "ETH"]
    sector = SECTOR_MAP.get(base) if sector_aware else None
    if sector:
        leader_base = SECTOR_LEADERS.get(sector, "BTC")
        # Leaders and majors are benchmarked against BTC / ETH; everyone else against their leader
        if base != leader_base and base not in ("BTC", "ETH", "SOL", "DOGE", "BLUR"):
            preferred = [leader_base, "BTC" if leader_base != "BTC" else "ETH"]

    bases = [b for b in dict.fromkeys(preferred + ["BTC", "ETH", "SOL"]) if b != base][:2]
    return [format_benchmark_symbol(b, quote, ex) for b in bases]


def sanitize_symbol_for_filename(symbol: str) -> str:
    """Convert symbol to safe filename string (e.g. 'B-DOGE_USDT' -> 'b_doge_usdt')."""
    clean = symbol.lower().replace("-", "_").replace(":", "_")
    return clean
