"""Tests for configuration loading."""

from pathlib import Path

from futurecast.config import load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"


def test_config_loads_defaults():
    cfg = load_config(REPO_CONFIG)
    assert cfg.assets.data_source.base_url == "https://fapi.binance.com"
    assert cfg.assets.storage.use_cache is True

