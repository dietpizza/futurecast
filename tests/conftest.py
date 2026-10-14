"""Shared test setup: a throwaway futurecast home folder and an offline coin search."""

import pytest

from futurecast.ingestion import coindcx_catalog


class _OfflineBinance:
    def perpetuals(self) -> dict:
        return {}


@pytest.fixture(autouse=True)
def futurecast_home(tmp_path, monkeypatch):
    """Point FUTURECAST_HOME at a temp folder so tests never touch ~/.futurecast."""
    home = tmp_path / "futurecast_home"
    monkeypatch.setenv("FUTURECAST_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def offline_catalog(monkeypatch):
    """Coin search sees only the curated registry unless a test supplies Binance perpetuals."""
    monkeypatch.setattr(coindcx_catalog, "_catalog_client", _OfflineBinance())
    monkeypatch.setattr(coindcx_catalog, "_ALL_INSTRUMENTS_CACHE", [])
    monkeypatch.setattr(coindcx_catalog, "_LAST_FETCH_TIME", 0.0)
