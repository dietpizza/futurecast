"""Data schemas, event types, and signal state definitions."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class SignalState(str, Enum):
    BULLISH_RESIDUAL = "BULLISH_RESIDUAL"
    BEARISH_RESIDUAL = "BEARISH_RESIDUAL"
    NEUTRAL = "NEUTRAL"


class KlineBar(BaseModel):
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    count: int
    taker_buy_volume: float
    taker_buy_quote_volume: float


class FundingRateRecord(BaseModel):
    timestamp: datetime
    symbol: str
    funding_rate: float


class OpenInterestRecord(BaseModel):
    timestamp: datetime
    symbol: str
    open_interest: float
    open_interest_usd: float


class ModelForecastItem(BaseModel):
    model_name: str
    expected_residual_alpha_bps: float
    z_score: float
    signal: SignalState
    confidence_percentile: float
    hit_rate: Optional[float] = None
    spearman_ic: Optional[float] = None


class CrossVerificationSummary(BaseModel):
    concordance: float
    consensus_signal: SignalState
    models_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    dispersion_bps: float


class ForecastEvent(BaseModel):
    timestamp: datetime
    symbol: str
    benchmark_symbol: str
    current_price: float
    benchmark_price: float
    expected_residual_alpha_15m_bps: float
    z_score: float
    beta_hedge_ratio: float
    signal: SignalState
    confidence_percentile: float
    microstructure_context: Dict[str, Any] = Field(default_factory=dict)
    trade_risk: Dict[str, Any] = Field(default_factory=dict)
    orderbook_depth: Dict[str, Any] = Field(default_factory=dict)
    execution_disabled: bool = True  # Permanently True
    model_predictions: Dict[str, ModelForecastItem] = Field(default_factory=dict)
    cross_verification: Optional[CrossVerificationSummary] = None
    sector: Optional[str] = None


class ValidationFoldMetric(BaseModel):
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    spearman_ic: float
    rmse: float
    mae: float
    n_train_samples: int
    n_test_samples: int


class ValidationSummary(BaseModel):
    mean_ic: float
    std_ic: float
    ic_ir: float
    mean_rmse: float
    mean_mae: float
    fold_metrics: list[ValidationFoldMetric]
