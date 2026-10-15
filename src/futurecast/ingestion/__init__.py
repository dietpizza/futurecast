"""Ingestion package initialization."""

from futurecast.ingestion.synthetic import generate_synthetic_data
from futurecast.ingestion.binance import BinanceMarket, BinancePublicClient, binance_candidates
from futurecast.ingestion.storage import align_cross_asset_data
from futurecast.ingestion.cache import CandleCache
from futurecast.ingestion.factory import (
    detect_exchange,
    resolve_benchmarks,
    sanitize_symbol_for_filename,
    get_symbol_sector,
    extract_base_and_quote,
    SECTOR_MAP,
    SECTOR_LEADERS,
)

__all__ = [
    "BinanceMarket",
    "BinancePublicClient",
    "binance_candidates",
    "generate_synthetic_data",
    "align_cross_asset_data",
    "CandleCache",
    "detect_exchange",
    "resolve_benchmarks",
    "sanitize_symbol_for_filename",
    "get_symbol_sector",
    "extract_base_and_quote",
    "SECTOR_MAP",
    "SECTOR_LEADERS",
]
