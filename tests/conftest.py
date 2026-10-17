"""Shared test setup: a throwaway deltarisk home folder and an offline coin search."""

import pytest

from deltarisk.ingestion import coindcx_catalog


class _OfflineBinance:
    def perpetuals(self) -> dict:
        return {}


@pytest.fixture(autouse=True)
def deltarisk_home(tmp_path, monkeypatch):
    """Point DELTARISK_HOME at a temp folder so tests never touch ~/.deltarisk."""
    home = tmp_path / "deltarisk_home"
    monkeypatch.setenv("DELTARISK_HOME", str(home))
    monkeypatch.delenv("FUTURECAST_HOME", raising=False)
    # Safety net: even a test that clears the environment never sees (or moves) the real folders
    from deltarisk import paths
    monkeypatch.setattr(paths, "DEFAULT_HOME", str(tmp_path / "default_home"))
    monkeypatch.setattr(paths, "LEGACY_HOME", str(tmp_path / "legacy_home"))
    return home


@pytest.fixture(autouse=True)
def offline_catalog(monkeypatch):
    """Coin search sees only the curated registry unless a test supplies Binance perpetuals."""
    monkeypatch.setattr(coindcx_catalog, "_catalog_client", _OfflineBinance())
    monkeypatch.setattr(coindcx_catalog, "_ALL_INSTRUMENTS_CACHE", [])
    monkeypatch.setattr(coindcx_catalog, "_LAST_FETCH_TIME", 0.0)
