"""Configuration loader and schema validation for futurecast.

Enforces zero-execution guarantees at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List
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
    # Set per coin by the stream manager for the residual-alpha engine; the defaults only serve tests
    target_asset: str = "DOGEUSDT"
    benchmark_assets: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


class CrossAssetFeaturesConfig(BaseModel):
    horizons_min: List[int] = Field(default_factory=lambda: [1, 5, 15, 60])
    beta_windows_min: List[int] = Field(default_factory=lambda: [60, 1440])
    rsi_period: int = 14


class MicrostructureFeaturesConfig(BaseModel):
    volume_z_windows_min: List[int] = Field(default_factory=lambda: [60, 1440])
    rvr_window_min: int = 15
    parkinson_windows_min: List[int] = Field(default_factory=lambda: [15, 60, 240])


class DerivativesFeaturesConfig(BaseModel):
    funding_ema_window_min: int = 4320  # trailing funding mean, in clock minutes (3 days), not candles


class TechnicalFeaturesConfig(BaseModel):
    stochastic_extremes_window_min: int = 60
    vwap_window_min: int = 240


class TargetConfig(BaseModel):
    horizon_min: int = 15  # residual-alpha horizon, in candles of the engine interval


class FeaturesConfig(BaseModel):
    cross_asset: CrossAssetFeaturesConfig = Field(default_factory=CrossAssetFeaturesConfig)
    microstructure: MicrostructureFeaturesConfig = Field(default_factory=MicrostructureFeaturesConfig)
    derivatives: DerivativesFeaturesConfig = Field(default_factory=DerivativesFeaturesConfig)
    technical: TechnicalFeaturesConfig = Field(default_factory=TechnicalFeaturesConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)


class LightGBMConfig(BaseModel):
    objective: str = "regression_l1"
    metric: str = "rmse"
    boosting_type: str = "gbdt"
    learning_rate: float = 0.015
    num_leaves: int = 31
    max_depth: int = 5
    min_child_samples: int = 250
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.7
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    n_estimators: int = 1500
    n_jobs: int = -1
    verbose: int = -1
    early_stopping_rounds: int = 50


class ForecastingConfig(BaseModel):
    zscore_window_minutes: int = 1440
    signal_threshold_z: float = 1.5
    output_dir: str = "logs/forecasts"
    log_limit_mb: float = 32.0  # forecast logs kept in total; the oldest are deleted first
    air_gapped: bool = True
    execution_enabled: bool = False  # Hard safety rule: must always be False


class AppConfig(BaseModel):
    assets: AssetsConfig
    features: FeaturesConfig
    model: LightGBMConfig
    forecasting: ForecastingConfig

    def enforce_zero_execution_policy(self) -> None:
        """Fail-safe validation: ensure no trading permissions or keys can ever be active."""
        if self.forecasting.execution_enabled:
            raise RuntimeError(
                "CRITICAL VIOLATION: Execution capability is permanently forbidden. "
                "futurecast is strictly an air-gapped forecasting engine."
            )
        if not self.forecasting.air_gapped:
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
    assets_raw = load_yaml(cfg_path / "assets.yaml")
    features_raw = load_yaml(cfg_path / "features.yaml")
    model_raw = load_yaml(cfg_path / "model.yaml")

    assets = AssetsConfig(**assets_raw) if assets_raw else AssetsConfig()
    features = FeaturesConfig(**features_raw) if features_raw else FeaturesConfig()
    
    model = LightGBMConfig(**(model_raw.get("lightgbm", {})))
    forecasting = ForecastingConfig(**(model_raw.get("forecasting", {})))

    app_cfg = AppConfig(
        assets=assets,
        features=features,
        model=model,
        forecasting=forecasting,
    )
    app_cfg.enforce_zero_execution_policy()
    return app_cfg
