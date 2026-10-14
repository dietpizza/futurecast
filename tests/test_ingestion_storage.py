"""Tests for data ingestion, continuous timestamp alignment, and storage."""

from datetime import datetime, timezone
import pandas as pd
import numpy as np
from futurecast.ingestion.binance import generate_synthetic_data
from futurecast.ingestion.storage import align_cross_asset_data, ParquetStorage


def test_synthetic_data_generation():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = generate_synthetic_data("DOGEUSDT", start=start, periods=100, base_price=0.15, volatility=0.6)
    assert len(df) == 100
    assert "timestamp" in df.columns
    assert "close" in df.columns
    assert "funding_rate" in df.columns
    assert "open_interest" in df.columns
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    assert (df["high"] >= df["close"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["low"] <= df["open"]).all()


def test_cross_asset_alignment_with_gap_ffill():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    target = generate_synthetic_data("DOGEUSDT", start=start, periods=60, base_price=0.15, volatility=0.6)
    btc = generate_synthetic_data("BTCUSDT", start=start, periods=60, base_price=65000.0, volatility=0.4)

    # Inject a 2-minute gap in BTC (indices 10 and 11)
    btc_gapped = btc.drop(index=[10, 11]).reset_index(drop=True)

    aligned = align_cross_asset_data(target, {"BTCUSDT": btc_gapped}, max_gap_ffill_minutes=3)

    # Should retain all continuous minutes because 2-min gap <= 3-min max ffill
    assert len(aligned) == 60
    assert "DOGEUSDT_close" in aligned.columns
    assert "BTCUSDT_close" in aligned.columns
    assert not aligned["BTCUSDT_close"].isna().any()


def test_parquet_storage_roundtrip(tmp_path):
    storage = ParquetStorage(base_dir=tmp_path)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    df = generate_synthetic_data("DOGEUSDT", start=start, periods=50, base_price=0.15, volatility=0.6)

    storage.save_raw(df, "DOGEUSDT", category="klines")
    loaded = storage.load_raw("DOGEUSDT", category="klines")

    assert len(loaded) == len(df)
    assert np.allclose(loaded["close"].values, df["close"].values)
