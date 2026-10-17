"""Tests for continuous timestamp alignment of a coin and its benchmark."""

import numpy as np
import pandas as pd

from deltarisk.ingestion.storage import align_cross_asset_data


def _candles(symbol: str, periods: int, price: float) -> pd.DataFrame:
    close = price * np.exp(np.cumsum(np.random.default_rng(len(symbol)).normal(0, 0.001, periods)))
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=periods, freq="1min", tz="UTC"), "symbol": symbol,
        "open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0,
    })


def test_cross_asset_alignment_with_gap_ffill():
    target = _candles("DOGEUSDT", 60, 0.15)
    btc = _candles("BTCUSDT", 60, 65000.0)

    # Inject a 2-minute gap in BTC (indices 10 and 11)
    btc_gapped = btc.drop(index=[10, 11]).reset_index(drop=True)

    aligned = align_cross_asset_data(target, {"BTCUSDT": btc_gapped}, max_gap_ffill_minutes=3)

    # Should retain all continuous minutes because 2-min gap <= 3-min max ffill
    assert len(aligned) == 60
    assert "DOGEUSDT_close" in aligned.columns
    assert "BTCUSDT_close" in aligned.columns
    assert not aligned["BTCUSDT_close"].isna().any()
    assert (aligned["BTCUSDT_volume"].iloc[10:12] == 0).all()  # filled candles traded nothing


def test_a_gap_longer_than_the_fill_limit_drops_its_candles():
    target = _candles("DOGEUSDT", 60, 0.15)
    btc = _candles("BTCUSDT", 60, 65000.0).drop(index=range(20, 25)).reset_index(drop=True)

    aligned = align_cross_asset_data(target, {"BTCUSDT": btc}, max_gap_ffill_minutes=3)
    gap = pd.date_range("2025-01-01 00:20", periods=5, freq="1min", tz="UTC")
    assert not aligned.index.isin(gap).any() and aligned["BTCUSDT_close"].notna().all()
    assert aligned.index[-1] == pd.Timestamp("2025-01-01 00:59", tz="UTC")
