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
    orderbook_depth: Dict[str, Any] = Field(default_factory=dict)
    execution_disabled: bool = True  # Permanently True
    model_predictions: Dict[str, ModelForecastItem] = Field(default_factory=dict)
    cross_verification: Optional[CrossVerificationSummary] = None
    sector: Optional[str] = None
