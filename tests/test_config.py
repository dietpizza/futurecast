"""Tests for configuration loading and zero-execution safety assertions."""

from pathlib import Path

import pytest
from futurecast.config import load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"


def test_config_loads_defaults():
    cfg = load_config(REPO_CONFIG)
    assert cfg.assets.data_source.base_url == "https://fapi.binance.com"
    assert cfg.assets.storage.use_cache is True
    assert cfg.safety.execution_enabled is False
    assert cfg.safety.air_gapped is True


def test_zero_execution_policy_prevents_trading():
    cfg = load_config(REPO_CONFIG)

    # Deliberately attempt to enable execution
    cfg.safety.execution_enabled = True
    with pytest.raises(RuntimeError, match="Execution capability is permanently forbidden"):
        cfg.enforce_zero_execution_policy()


def test_air_gap_policy_enforcement():
    cfg = load_config(REPO_CONFIG)
    cfg.safety.air_gapped = False
    with pytest.raises(RuntimeError, match="air_gapped must be True"):
        cfg.enforce_zero_execution_policy()


def test_a_config_file_cannot_switch_execution_on(tmp_path):
    (tmp_path / "safety.yaml").write_text("execution_enabled: true\n")
    with pytest.raises(RuntimeError, match="permanently forbidden"):
        load_config(tmp_path)
