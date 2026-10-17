"""Configuration loader and schema validation for futurecast.

Enforces zero-execution guarantees at runtime.
"""

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


class SafetyConfig(BaseModel):
    air_gapped: bool = True
    execution_enabled: bool = False  # Hard safety rule: must always be False


class AppConfig(BaseModel):
    assets: AssetsConfig = Field(default_factory=AssetsConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    def enforce_zero_execution_policy(self) -> None:
        """Fail-safe validation: ensure no trading permissions or keys can ever be active."""
        if self.safety.execution_enabled:
            raise RuntimeError(
                "CRITICAL VIOLATION: Execution capability is permanently forbidden. "
                "futurecast is strictly an air-gapped, read-only dashboard."
            )
        if not self.safety.air_gapped:
            raise RuntimeError(
                "CRITICAL VIOLATION: air_gapped must be True."
            )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str | Path = "config") -> AppConfig:
    """Settings from `config_dir` (~/.futurecast/config by default); missing files keep the defaults."""
    cfg_path = resolve(config_dir)
    app_cfg = AppConfig(
        assets=AssetsConfig(**load_yaml(cfg_path / "assets.yaml")),
        safety=SafetyConfig(**load_yaml(cfg_path / "safety.yaml")),
    )
    app_cfg.enforce_zero_execution_policy()
    return app_cfg
