"""CoinDCX Futures Instrument Catalog and Search Provider.

Provides structured metadata, sector classification, and benchmark routing
for CoinDCX derivatives contracts (B-* pairs) and INR spot pairs (I-* pairs).
Dynamically integrates with CoinDCX's public active instruments API with zero-latency caching.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import httpx

from futurecast.features.shitcoin import is_high_beta_shitcoin
from futurecast.ingestion.factory import extract_base_and_quote, get_symbol_sector, resolve_benchmarks

logger = logging.getLogger(__name__)

# Curated catalog of major, alt, meme, and thin CoinDCX futures pairs (immediate zero-latency baseline)
COINDCX_FUTURES_REGISTRY: List[Dict[str, str]] = [
    # Layer 2 / Scaling / Zero-Knowledge Blockchains
    {"symbol": "B-ZK_USDT", "name": "zkSync", "description": "High-throughput Layer-2 ZK-rollup scaling Ethereum with cryptographic zero-knowledge proofs."},
    {"symbol": "B-ARB_USDT", "name": "Arbitrum", "description": "Leading Layer-2 optimistic rollup ecosystem with the largest DeFi liquidity on Ethereum."},
    {"symbol": "B-OP_USDT", "name": "Optimism", "description": "Pioneering Layer-2 network and Superchain ecosystem for horizontal scaling."},
    {"symbol": "B-STRK_USDT", "name": "Starknet", "description": "Validity-rollup Layer-2 scaling Ethereum using STARK zero-knowledge proofs."},
    {"symbol": "B-ZRO_USDT", "name": "LayerZero", "description": "Omnichain interoperability protocol connecting across 70+ blockchains."},

    # Meme Coins (High-Beta Shitcoins)
    {"symbol": "B-DOGE_USDT", "name": "Dogecoin", "description": "Original internet meme cryptocurrency and market anchor for all meme tokens."},
    {"symbol": "B-PEPE_USDT", "name": "Pepe", "description": "Viral frog meme token with high intraday volatility and retail momentum."},
    {"symbol": "B-SHIB_USDT", "name": "Shiba Inu", "description": "Ecosystem meme token benchmarked against DOGE and Bitcoin."},
    {"symbol": "B-FLOKI_USDT", "name": "Floki", "description": "High-beta meme token with gaming and DeFi utility elements."},
    {"symbol": "B-BONK_USDT", "name": "Bonk", "description": "Solana-native community dog coin with strong ecosystem liquidity."},
    {"symbol": "B-WIF_USDT", "name": "dogwifhat", "description": "Solana meme phenomenon characterized by fast momentum waves."},
    {"symbol": "B-POPCAT_USDT", "name": "Popcat", "description": "Rapidly growing cat meme coin traded on futures."},
    {"symbol": "B-NEIRO_USDT", "name": "First Neiro", "description": "Emergent dog meme coin with high-frequency order flow."},
    {"symbol": "B-BOME_USDT", "name": "Book of Meme", "description": "Solana-based experimental meme archive token."},
    {"symbol": "B-MEW_USDT", "name": "cat in a dogs world", "description": "Feline-themed alternative meme token."},
    {"symbol": "B-BRETT_USDT", "name": "Brett", "description": "Base network cultural meme asset."},
    {"symbol": "B-MOG_USDT", "name": "Mog Coin", "description": "High-beta internet culture meme coin."},
    {"symbol": "B-MYRO_USDT", "name": "Myro", "description": "Community token named after Solana co-founder's dog."},
    {"symbol": "B-TURBO_USDT", "name": "Turbo", "description": "First AI-generated meme token created with GPT-4."},
    {"symbol": "B-MEME_USDT", "name": "Memecoin", "description": "Ecosystem token for Memeland Web3 venture studio."},
    {"symbol": "B-ORDI_USDT", "name": "Ordinals", "description": "First BRC-20 fungible token on the Bitcoin blockchain."},
    {"symbol": "B-1000SATS_USDT", "name": "SATS", "description": "BRC-20 token paying homage to Satoshi Nakamoto."},

    # Layer 1 / Alternative L1 Blockchains
    {"symbol": "B-SOL_USDT", "name": "Solana", "description": "High-throughput Layer-1 smart contract platform; Alt-L1 sector anchor."},
    {"symbol": "B-AVAX_USDT", "name": "Avalanche", "description": "Scalable multi-subnet blockchain ecosystem."},
    {"symbol": "B-SUI_USDT", "name": "Sui Network", "description": "High-performance object-centric Move blockchain."},
    {"symbol": "B-APT_USDT", "name": "Aptos", "description": "High-throughput Layer-1 platform designed by former Diem engineers."},
    {"symbol": "B-NEAR_USDT", "name": "NEAR Protocol", "description": "Sharded developer-friendly blockchain powering open web and AI."},
    {"symbol": "B-ADA_USDT", "name": "Cardano", "description": "Proof-of-stake blockchain built on peer-reviewed research."},
    {"symbol": "B-DOT_USDT", "name": "Polkadot", "description": "Interoperable multi-chain network architecture."},
    {"symbol": "B-SEI_USDT", "name": "Sei Network", "description": "Sector-specific Layer-1 blockchain optimized for trading and exchanges."},
    {"symbol": "B-TON_USDT", "name": "Toncoin", "description": "High-capacity Layer-1 blockchain integrated with Telegram ecosystem."},
    {"symbol": "B-INJ_USDT", "name": "Injective", "description": "Interoperable blockchain optimized for decentralized finance applications."},
    {"symbol": "B-TIA_USDT", "name": "Celestia", "description": "Modular data availability network scaling modular blockchains."},
    {"symbol": "B-FTM_USDT", "name": "Fantom", "description": "Directed Acyclic Graph (DAG) smart contract platform."},
    {"symbol": "B-ATOM_USDT", "name": "Cosmos", "description": "Decentralized network of independent parallel blockchains (IBC)."},
    {"symbol": "B-ALGO_USDT", "name": "Algorand", "description": "Pure proof-of-stake blockchain platform designed for fast finality."},
    {"symbol": "B-HBAR_USDT", "name": "Hedera", "description": "Enterprise-grade public ledger utilizing hashgraph consensus."},
    {"symbol": "B-TRX_USDT", "name": "TRON", "description": "High-throughput blockchain with massive stablecoin settlement volume."},

    # Decentralized Finance (DeFi)
    {"symbol": "B-ETH_USDT", "name": "Ethereum", "description": "Leading global smart contracts platform and DeFi sector benchmark anchor."},
    {"symbol": "B-UNI_USDT", "name": "Uniswap", "description": "Pioneering decentralized automated market maker exchange."},
    {"symbol": "B-AAVE_USDT", "name": "Aave", "description": "Leading liquidity protocol for decentralized borrowing and lending."},
    {"symbol": "B-CRV_USDT", "name": "Curve DAO", "description": "Deep-liquidity exchange protocol optimized for stablecoins and pegged assets."},
    {"symbol": "B-MKR_USDT", "name": "Maker", "description": "Governance protocol behind the DAI decentralized stablecoin."},
    {"symbol": "B-LDO_USDT", "name": "Lido DAO", "description": "Liquid staking solution for Ethereum and proof-of-stake networks."},
    {"symbol": "B-PENDLE_USDT", "name": "Pendle", "description": "Yield-trading protocol tokenizing and trading future yield."},
    {"symbol": "B-ENA_USDT", "name": "Ethena", "description": "Synthetic dollar protocol built on Ethereum providing internet bond yield."},
    {"symbol": "B-JUP_USDT", "name": "Jupiter", "description": "Key liquidity aggregator and trading engine on Solana."},
    {"symbol": "B-RAY_USDT", "name": "Raydium", "description": "On-chain order book AMM on Solana."},
    {"symbol": "B-DYDX_USDT", "name": "dYdX", "description": "Decentralized derivatives exchange for perpetual trading."},
    {"symbol": "B-RUNE_USDT", "name": "THORChain", "description": "Cross-chain decentralized liquidity network for native asset swaps."},
    {"symbol": "B-SNX_USDT", "name": "Synthetix", "description": "Derivatives liquidity protocol powering on-chain synthetic assets."},

    # AI / Compute / DePIN
    {"symbol": "B-RENDER_USDT", "name": "Render Network", "description": "Decentralized GPU rendering and spatial computing network."},
    {"symbol": "B-FET_USDT", "name": "ASI Alliance", "description": "Artificial Superintelligence Alliance token powering autonomous economic agents."},
    {"symbol": "B-TAO_USDT", "name": "Bittensor", "description": "Decentralized machine intelligence network rewarding model contribution."},
    {"symbol": "B-WLD_USDT", "name": "Worldcoin", "description": "Identity and financial network protocol designed for the AI era."},
    {"symbol": "B-ICP_USDT", "name": "Internet Computer", "description": "Decentralized cloud computing platform hosting serverless software."},
    {"symbol": "B-GRT_USDT", "name": "The Graph", "description": "Indexing and querying protocol for querying blockchain data with GraphQL."},
    {"symbol": "B-AKT_USDT", "name": "Akash Network", "description": "Decentralized peer-to-peer open cloud compute marketplace."},
    {"symbol": "B-AR_USDT", "name": "Arweave", "description": "Permanent decentralized data storage protocol."},

    # NFT / Gaming / Metaverse
    {"symbol": "B-BLUR_USDT", "name": "Blur", "description": "Leading NFT marketplace aggregator with pro trading features."},
    {"symbol": "B-SAND_USDT", "name": "The Sandbox", "description": "Virtual metaverse world allowing users to build, own, and monetize assets."},
    {"symbol": "B-MANA_USDT", "name": "Decentraland", "description": "Virtual reality platform powered by the Ethereum blockchain."},
    {"symbol": "B-AXS_USDT", "name": "Axie Infinity", "description": "Blockchain-based gaming universe centered around collectible creatures."},
    {"symbol": "B-GALA_USDT", "name": "Gala Games", "description": "Decentralized gaming ecosystem giving players control of their games."},
    {"symbol": "B-IMX_USDT", "name": "Immutable", "description": "Layer-2 scaling solution for games and NFTs on Ethereum."},

    # Infrastructure / Oracle / Thin Alts
    {"symbol": "B-LINK_USDT", "name": "Chainlink", "description": "Standard decentralized oracle network connecting smart contracts to real-world data."},
    {"symbol": "B-PYTH_USDT", "name": "Pyth Network", "description": "Low-latency financial oracle delivering real-time institutional market data."},
    {"symbol": "B-FIL_USDT", "name": "Filecoin", "description": "Decentralized storage network turning cloud storage into an algorithmic market."},
    {"symbol": "B-AVA_USDT", "name": "Travala", "description": "Blockchain-based travel booking platform token; thin liquidity profile."},
    {"symbol": "B-W_USDT", "name": "Wormhole", "description": "Cross-chain messaging and communication infrastructure protocol."},

    # Majors & INR Spot
    {"symbol": "B-BTC_USDT", "name": "Bitcoin", "description": "The market anchor and reserve asset; primary macro driver for crypto returns."},
    {"symbol": "I-BTC_INR", "name": "Bitcoin INR", "description": "Indian Rupee direct spot trading pair on CoinDCX."},
    {"symbol": "I-ETH_INR", "name": "Ethereum INR", "description": "Indian Rupee Ethereum trading pair on CoinDCX."},
]

# In-memory dynamic catalog cache
_ALL_INSTRUMENTS_CACHE: List[Dict[str, Any]] = []
_TARGET_NAMES_CACHE: Dict[str, str] = {}
_LAST_FETCH_TIME: float = 0.0
_CACHE_TTL_SEC: float = 3600.0  # 1 hour


def _load_cached_catalog_data() -> None:
    """Load cached active instruments and token names from local disk if available."""
    global _TARGET_NAMES_CACHE
    cache_dir = Path(".cache")
    names_file = cache_dir / "coindcx_target_names.json"
    if names_file.exists() and not _TARGET_NAMES_CACHE:
        try:
            with open(names_file, "r", encoding="utf-8") as f:
                _TARGET_NAMES_CACHE = json.load(f)
        except Exception:
            pass


def _sync_coindcx_instruments() -> List[str]:
    """Fetch active futures instruments from CoinDCX API or fallback cache."""
    cache_dir = Path(".cache")
    inst_file = cache_dir / "coindcx_active_instruments.json"

    # Try fetching fresh list from CoinDCX public derivatives API
    try:
        with httpx.Client(timeout=4.0) as client:
            resp = client.get("https://api.coindcx.com/exchange/v1/derivatives/futures/data/active_instruments")
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    try:
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        with open(inst_file, "w", encoding="utf-8") as f:
                            json.dump(data, f)
                    except Exception:
                        pass
                    return data
    except Exception as e:
        logger.debug(f"Live CoinDCX instruments fetch skipped: {e}")

    # Fallback to disk cache if available
    if inst_file.exists():
        try:
            with open(inst_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if isinstance(cached, list) and len(cached) > 0:
                    return cached
        except Exception:
            pass

    # Baseline fallback symbols
    return [item["symbol"] for item in COINDCX_FUTURES_REGISTRY]


def get_all_coindcx_instruments() -> List[Dict[str, Any]]:
    """Return complete list of all 500+ active CoinDCX futures contracts, enriched with metadata."""
    global _ALL_INSTRUMENTS_CACHE, _LAST_FETCH_TIME, _TARGET_NAMES_CACHE
    now = time.time()

    if _ALL_INSTRUMENTS_CACHE and (now - _LAST_FETCH_TIME) < _CACHE_TTL_SEC:
        return _ALL_INSTRUMENTS_CACHE

    _load_cached_catalog_data()
    raw_symbols = _sync_coindcx_instruments()

    # Build lookup of curated registry items
    curated_map = {item["symbol"]: item for item in COINDCX_FUTURES_REGISTRY}

    enriched: List[Dict[str, Any]] = []
    seen_symbols = set()

    # First add all curated registry items (guarantees B-DOGE_USDT, B-SOL_USDT, B-PEPE_USDT, B-ZK_USDT, etc.)
    for item in COINDCX_FUTURES_REGISTRY:
        sym = item["symbol"].strip().upper()
        if sym in seen_symbols:
            continue
        seen_symbols.add(sym)

        base, quote = extract_base_and_quote(sym)
        sector = get_symbol_sector(sym)
        is_shitcoin = is_high_beta_shitcoin(sym, sector)
        benchmarks = resolve_benchmarks(sym, "coindcx", sector_aware=True)
        coindcx_url = f"https://coindcx.com/futures/{sym}" if sym.startswith("B-") else f"https://coindcx.com/trade/{sym}"

        enriched.append({
            "symbol": sym,
            "base": base,
            "quote": quote,
            "name": item["name"],
            "sector": sector,
            "is_shitcoin": is_shitcoin,
            "benchmarks": benchmarks,
            "description": item["description"],
            "coindcx_url": coindcx_url,
        })

    # Then merge all live active instruments from CoinDCX
    for sym in raw_symbols:
        s_clean = sym.strip().upper()
        if s_clean in seen_symbols:
            continue
        seen_symbols.add(s_clean)

        base, quote = extract_base_and_quote(s_clean)
        sector = get_symbol_sector(s_clean)
        is_shitcoin = is_high_beta_shitcoin(s_clean, sector)
        benchmarks = resolve_benchmarks(s_clean, "coindcx", sector_aware=True)
        name = _TARGET_NAMES_CACHE.get(base.upper(), base)
        description = f"CoinDCX {quote}-Margined Perpetual Contract for {name} ({base})."
        coindcx_url = f"https://coindcx.com/futures/{s_clean}" if s_clean.startswith("B-") else f"https://coindcx.com/trade/{s_clean}"

        enriched.append({
            "symbol": s_clean,
            "base": base,
            "quote": quote,
            "name": name,
            "sector": sector,
            "is_shitcoin": is_shitcoin,
            "benchmarks": benchmarks,
            "description": description,
            "coindcx_url": coindcx_url,
        })

    _ALL_INSTRUMENTS_CACHE = enriched
    _LAST_FETCH_TIME = now
    return _ALL_INSTRUMENTS_CACHE


def clean_search_query(raw_query: Optional[str]) -> str:
    """Extract symbol or keyword from user input, supporting direct URL paste."""
    if not raw_query:
        return ""
    q = raw_query.strip().lower()
    # If user pasted a full CoinDCX URL like https://coindcx.com/futures/B-ZK_USDT
    if "coindcx.com/" in q:
        parts = q.split("/")
        q = parts[-1].split("?")[0].split("#")[0].strip()
    return q


def get_coindcx_futures_catalog(
    query: Optional[str] = None,
    sector: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return enriched catalog of CoinDCX futures pairs with search and sector filtering.
    
    Includes all 500+ active CoinDCX futures contracts, dynamically resolves symbols like B-ZK_USDT,
    and handles direct URL paste (e.g. https://coindcx.com/futures/B-ZK_USDT).
    """
    all_items = get_all_coindcx_instruments()
    q = clean_search_query(query)
    sec_filter = (sector or "").strip().upper()

    results: List[Dict[str, Any]] = []

    for item in all_items:
        sym = item["symbol"]
        base = item["base"]
        name = item["name"]
        coin_sector = item["sector"]
        desc = item["description"]

        # Sector filtering
        if sec_filter and sec_filter != "ALL":
            if sec_filter == "MEME" and coin_sector != "MEME":
                continue
            elif sec_filter == "L1" and coin_sector != "L1":
                continue
            elif sec_filter == "DEFI" and coin_sector != "DEFI":
                continue
            elif sec_filter == "AI" and coin_sector != "AI":
                continue
            elif sec_filter == "NFT" and coin_sector != "NFT_GAMING":
                continue
            elif sec_filter == "MAJOR" and coin_sector not in ("MAJOR", "GENERAL_ALT", "INFRA"):
                continue

        # Query filtering (matches symbol, base, name, description, or sector)
        if q:
            match_sym = q in sym.lower()
            match_base = q == base.lower() or q in base.lower()
            match_name = q in name.lower()
            match_desc = q in desc.lower()
            match_sec = q in coin_sector.lower()
            if not (match_sym or match_base or match_name or match_desc or match_sec):
                continue

        results.append(item)

    # Dynamic fallback: If user searched an exact CoinDCX pair not yet in registry, synthesize candidate
    if q and not results:
        candidate_sym = q.upper()
        if not candidate_sym.startswith("B-") and not candidate_sym.startswith("I-"):
            if candidate_sym.endswith("USDT"):
                candidate_sym = f"B-{candidate_sym}"
            else:
                candidate_sym = f"B-{candidate_sym}_USDT"

        base, quote = extract_base_and_quote(candidate_sym)
        c_sector = get_symbol_sector(candidate_sym)
        c_shit = is_high_beta_shitcoin(candidate_sym, c_sector)
        results.append({
            "symbol": candidate_sym,
            "base": base,
            "quote": quote,
            "name": base,
            "sector": c_sector,
            "is_shitcoin": c_shit,
            "benchmarks": resolve_benchmarks(candidate_sym, "coindcx", sector_aware=True),
            "description": f"CoinDCX {quote}-Margined Perpetual Contract for {base}.",
            "coindcx_url": f"https://coindcx.com/futures/{candidate_sym}",
        })

    return results
