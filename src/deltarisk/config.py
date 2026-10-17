"""Configuration loader and schema validation for futurecast."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field

from futurecast.paths import resolve


class DataSourceConfig(BaseModel):
    base_url: str = "https://fapi.binance.com"
    request_timeout_sec: int = 15
    max_retries: int = 3


class StorageConfig(BaseModel):
    cache_dir: str = "cache"  # downloaded candles go in <cache_dir>/candles
    use_cache: bool = True


class AssetsConfig(BaseModel):
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


class AppConfig(BaseModel):
    assets: AssetsConfig = Field(default_factory=AssetsConfig)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str | Path = "config") -> AppConfig:
    """Settings from `config_dir` (~/.futurecast/config by default); missing files keep the defaults."""
    cfg_path = resolve(config_dir)
    return AppConfig(assets=AssetsConfig(**load_yaml(cfg_path / "assets.yaml")))
