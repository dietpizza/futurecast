"""The pairs the add-coin search offers: a curated registry of CoinDCX futures (B-*) and INR
(I-*) pairs with names and descriptions, plus every trading Binance USDT perpetual under its
CoinDCX pair name (1000PEPEUSDT -> B-1000PEPE_USDT), each with its sector. CoinDCX itself is
never queried; the page does the searching.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List
import httpx

from deltarisk.ingestion.binance import BinancePublicClient
from deltarisk.ingestion.factory import extract_base_and_quote, get_symbol_sector

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
_LAST_FETCH_TIME: float = 0.0
_CACHE_TTL_SEC: float = 3600.0  # 1 hour

# CoinDCX lists crypto and commodity perpetuals, never Binance's stock / FX / index ones
_LISTED_UNDERLYINGS = ("COIN", "COMMODITY")
_catalog_client = BinancePublicClient(timeout=4.0, max_retries=1)


def _binance_pairs() -> List[str]:
    """CoinDCX pair names of the trading Binance USDT perpetuals; empty when Binance is unreachable."""
    try:
        perps = _catalog_client.perpetuals().values()
    except httpx.HTTPError as e:
        logger.debug(f"Binance perpetuals fetch skipped: {e}")
        return []
    return sorted(
        f"B-{s['baseAsset']}_USDT" for s in perps
        if s.get("quoteAsset") == "USDT" and s.get("underlyingType") in _LISTED_UNDERLYINGS
        and s["baseAsset"].isascii()  # CoinDCX has no CJK-named pairs
    )


def _instrument(symbol: str, name: str, description: str) -> Dict[str, Any]:
    base = extract_base_and_quote(symbol)[0]
    url = f"https://coindcx.com/futures/{symbol}" if symbol.startswith("B-") else f"https://coindcx.com/trade/{symbol}"
    return {"symbol": symbol, "base": base, "name": name, "sector": get_symbol_sector(symbol),
            "description": description, "coindcx_url": url}


def get_all_coindcx_instruments() -> List[Dict[str, Any]]:
    """The curated registry, then every pair with a trading Binance perpetual (cached for an hour)."""
    global _ALL_INSTRUMENTS_CACHE, _LAST_FETCH_TIME
    now = time.time()
    if _ALL_INSTRUMENTS_CACHE and (now - _LAST_FETCH_TIME) < _CACHE_TTL_SEC:
        return _ALL_INSTRUMENTS_CACHE

    raw_symbols = _binance_pairs()
    enriched: List[Dict[str, Any]] = []
    seen = set()
    for item in COINDCX_FUTURES_REGISTRY:
        sym = item["symbol"].strip().upper()
        if sym not in seen:
            seen.add(sym)
            enriched.append(_instrument(sym, item["name"], item["description"]))
    for sym in raw_symbols:
        if sym not in seen:
            seen.add(sym)
            base, quote = extract_base_and_quote(sym)
            enriched.append(_instrument(sym, base, f"CoinDCX {quote}-Margined Perpetual Contract for {base}."))

    _ALL_INSTRUMENTS_CACHE = enriched
    _LAST_FETCH_TIME = now if raw_symbols else 0.0  # after a Binance outage, retry on the next search
    return _ALL_INSTRUMENTS_CACHE
