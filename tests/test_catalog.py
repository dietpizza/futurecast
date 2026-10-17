"""Tests for the coin search catalog built from Binance perpetuals."""

import httpx

from deltarisk.ingestion import coindcx_catalog


class FakeBinance:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def perpetuals(self) -> dict:
        self.calls += 1
        if self.fail:
            raise httpx.ConnectError("offline")
        perps = [
            ("1000PEPE", "USDT", "COIN"), ("ZRO", "USDT", "COIN"), ("XAU", "USDT", "COMMODITY"),
            ("TSLA", "USDT", "EQUITY"), ("币安人生", "USDT", "COIN"), ("BTC", "USDC", "COIN"),
        ]
        return {f"{b}{q}": {"baseAsset": b, "quoteAsset": q, "underlyingType": u} for b, q, u in perps}


def test_catalog_lists_binance_crypto_and_commodity_perpetuals(monkeypatch):
    monkeypatch.setattr(coindcx_catalog, "_catalog_client", FakeBinance())
    symbols = {item["symbol"] for item in coindcx_catalog.get_all_coindcx_instruments()}

    assert {"B-1000PEPE_USDT", "B-ZRO_USDT", "B-XAU_USDT"} <= symbols
    assert "B-DOGE_USDT" in symbols  # curated registry
    assert not {"B-TSLA_USDT", "B-币安人生_USDT", "B-BTC_USDC"} & symbols


def test_catalog_falls_back_to_registry_and_retries_after_outage(monkeypatch):
    offline = FakeBinance(fail=True)
    monkeypatch.setattr(coindcx_catalog, "_catalog_client", offline)

    symbols = {item["symbol"] for item in coindcx_catalog.get_all_coindcx_instruments()}
    assert symbols == {item["symbol"] for item in coindcx_catalog.COINDCX_FUTURES_REGISTRY}

    coindcx_catalog.get_all_coindcx_instruments()
    assert offline.calls == 2  # a failed fetch is not cached for the hour
