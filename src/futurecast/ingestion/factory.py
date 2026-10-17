"""Exchange detection and symbol resolution utilities."""

from __future__ import annotations

from typing import Dict, Tuple


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


def get_symbol_sector(symbol: str) -> str:
    """Return the market sector category for a symbol (e.g. MEME, L1, DEFI, AI, etc.)."""
    base, _ = extract_base_and_quote(symbol)
    return SECTOR_MAP.get(base, "GENERAL_ALT")


def resolve_benchmark(target_symbol: str) -> str:
    """The CoinDCX pair a coin's moves are measured against, by sector:
    - Meme coins (e.g. SHIB, PEPE, WIF) -> DOGE
    - Alt-L1s (e.g. AVAX, SUI, APT) -> SOL
    - DeFi tokens (e.g. UNI, AAVE, CRV) -> ETH
    - NFT/Gaming tokens (e.g. SAND, AXS) -> BLUR
    - AI tokens (e.g. RENDER, FET, TAO) -> NEAR
    - Infrastructure tokens (e.g. PYTH, FIL) -> LINK
    - Category leaders, majors and unlisted coins -> BTC

    A coin is never its own benchmark: BTC -> ETH. Its candle columns would collide with the
    benchmark's.
    """
    base, quote = extract_base_and_quote(target_symbol)
    leader = SECTOR_LEADERS.get(SECTOR_MAP.get(base, ""), "BTC")
    if leader == base or base in ("ETH", "SOL", "DOGE", "BLUR"):
        leader = "BTC"
    if base == "BTC":
        leader = "ETH"
    return f"I-{leader}_INR" if quote == "INR" else f"B-{leader}_{quote}"
