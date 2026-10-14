"""Ingestion package initialization."""

from futurecast.ingestion.binance import BinancePublicClient, generate_synthetic_data
from futurecast.ingestion.coindcx import CoinDCXPublicClient
from futurecast.ingestion.storage import ParquetStorage, align_cross_asset_data
from futurecast.ingestion.streamer import ContinuousKlineStreamer
from futurecast.ingestion.cache import DataCache
from futurecast.ingestion.factory import (
    detect_exchange,
    resolve_benchmarks,
    get_exchange_client,
    sanitize_symbol_for_filename,
    get_symbol_sector,
    extract_base_and_quote,
    SECTOR_MAP,
    SECTOR_LEADERS,
)

__all__ = [
    "BinancePublicClient",
    "CoinDCXPublicClient",
    "generate_synthetic_data",
    "ParquetStorage",
    "align_cross_asset_data",
    "ContinuousKlineStreamer",
    "DataCache",
    "detect_exchange",
    "resolve_benchmarks",
    "get_exchange_client",
    "sanitize_symbol_for_filename",
    "get_symbol_sector",
    "extract_base_and_quote",
    "SECTOR_MAP",
    "SECTOR_LEADERS",
]
