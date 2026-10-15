"""Synthetic 1-minute futures market data for offline tests."""

from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd


def generate_synthetic_data(
    symbol: str,
    start: datetime,
    periods: int,
    base_price: float,
    volatility: float,
    drift: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic 1-minute OHLCV and funding for offline testing."""
    np.random.seed(seed)
    timestamps = pd.date_range(start=start, periods=periods, freq="1min", tz="UTC")
    
    # Geometric Brownian Motion with student-t innovations
    dt = 1.0 / (365 * 24 * 60)
    returns = np.random.standard_t(df=5, size=periods) * volatility * np.sqrt(dt) + drift * dt
    price_series = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    intra_vol = volatility * np.sqrt(dt) * 0.5
    highs = price_series * (1.0 + np.abs(np.random.normal(0, intra_vol, size=periods)))
    lows = price_series * (1.0 - np.abs(np.random.normal(0, intra_vol, size=periods)))
    opens = np.roll(price_series, 1)
    opens[0] = base_price
    closes = price_series
    
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    
    # Volume
    base_vol = 1000.0 * (1.0 / base_price)
    volumes = np.random.lognormal(mean=np.log(base_vol), sigma=0.8, size=periods)
    quote_volumes = volumes * closes
    
    # Taker buy volume (fraction between 0.35 and 0.65)
    buy_ratio = np.clip(0.5 + 0.3 * (closes - opens) / (highs - lows + 1e-8), 0.2, 0.8)
    taker_buy_volumes = volumes * buy_ratio
    taker_buy_quote_volumes = quote_volumes * buy_ratio
    
    # Funding rate: changes every 480 minutes (8 hours)
    n_funding_intervals = (periods // 480) + 1
    funding_rates_raw = np.random.normal(0.0001, 0.0002, size=n_funding_intervals)
    funding_rates = np.repeat(funding_rates_raw, 480)[:periods]
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": symbol,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "quote_volume": quote_volumes,
        "count": np.random.poisson(lam=120, size=periods),
        "taker_buy_volume": taker_buy_volumes,
        "taker_buy_quote_volume": taker_buy_quote_volumes,
        "funding_rate": funding_rates,
    })
    return df
