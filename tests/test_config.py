"""Tests for configuration loading and zero-execution safety assertions."""

from pathlib import Path

import pytest
from futurecast.config import load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"


def test_config_loads_defaults():
    cfg = load_config(REPO_CONFIG)
    assert cfg.assets.target_asset == "DOGEUSDT"
    assert "BTCUSDT" in cfg.assets.benchmark_assets
    assert cfg.forecasting.execution_enabled is False
    assert cfg.forecasting.air_gapped is True


def test_zero_execution_policy_prevents_trading():
    cfg = load_config(REPO_CONFIG)
    
    # Deliberately attempt to enable execution
    cfg.forecasting.execution_enabled = True
    with pytest.raises(RuntimeError, match="Execution capability is permanently forbidden"):
        cfg.enforce_zero_execution_policy()


def test_air_gap_policy_enforcement():
    cfg = load_config(REPO_CONFIG)
    cfg.forecasting.air_gapped = False
    with pytest.raises(RuntimeError, match="air_gapped must be True"):
        cfg.enforce_zero_execution_policy()
