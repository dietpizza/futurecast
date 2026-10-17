// Package funding reads the funding rate at the latest closed candle against the coin's own
// recent rates.
package funding

import (
	"deltarisk/internal/candles"
	"deltarisk/internal/series"
)

const (
	Chart = "1h" // the chart that carries the coin's settled funding rate
	Days  = 90
)

// PercentileOf is the share of history below value, ties counted half, in percent; nil without
// finite history or value.
func PercentileOf(history []float64, value float64) *float64 {
	var less, equal, n float64
	for _, h := range history {
		if !series.Finite(h) {
			continue
		}
		n++
		if h < value {
			less++
		} else if h == value {
			equal++
		}
	}
	if n == 0 || !series.Finite(value) {
		return nil
	}
	p := 100.0 * (less + 0.5*equal) / n
	return &p
}

// Reading is the latest settled funding rate and its place among the rates in force over Days.
type Reading struct {
	RatePct    float64  `json:"rate_pct"`
	MeanPct    float64  `json:"mean_pct"`
	Percentile *float64 `json:"percentile"`
	Days       int      `json:"days"`
}

// Read is the latest rate of a column of rates (one per candle, perDay candles a day), its mean
// and its percentile over days; nil when there is no rate.
func Read(rates []float64, perDay, days int) *Reading {
	r := rates[max(0, len(rates)-days*perDay):]
	var finite []float64
	for _, v := range r {
		if series.Finite(v) {
			finite = append(finite, v)
		}
	}
	if len(finite) == 0 {
		return nil
	}
	last := finite[len(finite)-1]
	out := &Reading{RatePct: 100.0 * last, MeanPct: 100.0 * series.Mean(finite), Days: days}
	if len(finite) > 1 {
		out.Percentile = PercentileOf(finite[:len(finite)-1], last)
	}
	return out
}

// View is the coin's funding reading from its 1h chart, nil when the chart has no funding column.
func View(charts map[string]*candles.Chart, symbol string) *Reading {
	c := charts[Chart]
	if c.Empty() || !c.Has(symbol, "funding_rate") {
		return nil
	}
	return Read(c.Col(symbol, "funding_rate"), 1440/int(candles.MustIntervalMs(Chart)/60_000), Days)
}
