// Package history downloads a coin's candle history: how much of each chart the risk models
// use, a Binance client with the disk caches, and a coin and its benchmark aligned on one grid.
package history

import (
	"fmt"
	"path/filepath"
	"time"

	"deltarisk/internal/binance"
	"deltarisk/internal/cache"
	"deltarisk/internal/candles"
	"deltarisk/internal/config"
)

// Charts are the candle intervals the windows use, shortest first.
var Charts = []string{"5m", "15m", "1h"}

// Bars are the closed candles kept per chart: 90 days of 5m, 180 of 15m, 2 years of 1h.
var Bars = map[string]int{"5m": 25_920, "15m": 17_280, "1h": 17_520}

// FundingCharts carry the coin's settled funding rate.
var FundingCharts = map[string]bool{"1h": true}

// CacheBars are the candles kept on disk per market and interval (backtests read the cache too).
const CacheBars = 120_000

// Client is a Binance client from the settings, with the candle and funding caches when they are on.
func Client(cfg config.App) *binance.Client {
	src, store := cfg.Assets.DataSource, cfg.Assets.Storage
	o := binance.Options{BaseURL: src.BaseURL, Timeout: time.Duration(src.RequestTimeoutSec) * time.Second, MaxRetries: src.MaxRetries}
	if store.UseCache {
		o.Candles = cache.NewCandles(filepath.Join(store.CacheDir, "candles"), CacheBars)
		o.Funding = cache.NewFunding(filepath.Join(store.CacheDir, "funding"))
	}
	return binance.New(o)
}

// Aligned is the coin's closed candles (with its settled funding rate on FundingCharts) and its
// benchmark's on one time grid, and the data source ("" when Binance has no market).
// progress(symbol, done, total) reports candles downloaded per symbol (0, 0 as each starts).
func Aligned(c *binance.Client, symbol, benchmark, interval string, bars int, endMs int64,
	progress func(symbol string, done, total int)) (*candles.Chart, string, error) {
	fetch := func(sym string, funding bool) (candles.Bars, string, error) {
		var report func(done, total int)
		if progress != nil {
			progress(sym, 0, 0)
			report = func(done, total int) { progress(sym, done, total) }
		}
		return c.CandleHistory(sym, interval, bars, endMs, funding, report)
	}
	target, source, err := fetch(symbol, FundingCharts[interval])
	if err != nil || target.Len() == 0 {
		return &candles.Chart{}, "", err
	}
	bench, _, err := fetch(benchmark, false)
	if err != nil || bench.Len() == 0 {
		return &candles.Chart{}, "", err
	}
	chart, err := candles.Align(candles.MustIntervalMs(interval), 3, candles.Asset{Symbol: symbol, Bars: target},
		candles.Asset{Symbol: benchmark, Bars: bench})
	if err != nil {
		return &candles.Chart{}, "", fmt.Errorf("%s %s: %w", symbol, interval, err)
	}
	return &chart, source, nil
}
