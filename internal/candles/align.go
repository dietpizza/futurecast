package candles

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// Fields are a chart's per-symbol columns, in order; funding_rate only where the bars carry it.
var Fields = []string{"open", "high", "low", "close", "volume", "quote_volume", "count",
	"taker_buy_volume", "taker_buy_quote_volume", "funding_rate"}

// filled are the fields carried forward over a short gap; the others (volumes) read 0 there.
var filled = map[string]bool{"open": true, "high": true, "low": true, "close": true, "funding_rate": true}

// Chart is a coin and its benchmark on one grid of StepMs candles. Columns are named
// "{SYMBOL}_{field}" (B-ETH_USDT_close), all float64, row for row with OpenTime.
type Chart struct {
	StepMs   int64
	OpenTime []int64
	Names    []string
	Columns  map[string][]float64
}

func (c *Chart) Len() int { return len(c.OpenTime) }

// Empty says whether the chart has no candles.
func (c *Chart) Empty() bool { return c == nil || len(c.OpenTime) == 0 }

// Col is a column by symbol and field, nil when absent.
func (c *Chart) Col(symbol, field string) []float64 { return c.Columns[symbol+"_"+field] }

// Has says whether the chart has the column.
func (c *Chart) Has(symbol, field string) bool {
	_, ok := c.Columns[symbol+"_"+field]
	return ok
}

// CloseTimes is each candle's close time (open + step) in ms.
func (c *Chart) CloseTimes() []int64 {
	out := make([]int64, len(c.OpenTime))
	for i, t := range c.OpenTime {
		out[i] = t + c.StepMs
	}
	return out
}

// Asset is one market's bars under its symbol (the column prefix).
type Asset struct {
	Symbol string
	Bars   Bars
}

// Align puts the assets (the coin first, then its benchmark) on one continuous grid of stepMs
// candles from the latest first candle to the earliest last one. A gap of up to maxGapFill
// candles is filled: prices carried forward, volumes 0. A longer gap drops its candles and the
// candle before it (pandas' groupby of the gap with the candle that precedes it); so does any
// candle where an asset has no close.
func Align(stepMs int64, maxGapFill int, assets ...Asset) (Chart, error) {
	if len(assets) == 0 || assets[0].Bars.Len() == 0 {
		return Chart{}, errors.New("target candles are empty")
	}
	minTime, maxTime := int64(math.MinInt64), int64(math.MaxInt64)
	for _, a := range assets {
		if a.Bars.Len() == 0 {
			return Chart{}, fmt.Errorf("no candles for %s", a.Symbol)
		}
		lo, hi := a.Bars.Time[0], a.Bars.Time[0]
		for _, t := range a.Bars.Time {
			lo, hi = min(lo, t), max(hi, t)
		}
		minTime, maxTime = max(minTime, lo), min(maxTime, hi)
	}
	if minTime > maxTime {
		return Chart{}, fmt.Errorf("no overlapping time range found across assets (min: %d, max: %d)", minTime, maxTime)
	}
	n := int((maxTime-minTime)/stepMs) + 1
	grid := make([]int64, n)
	for i := range grid {
		grid[i] = minTime + int64(i)*stepMs
	}

	names := []string{}
	cols := map[string][]float64{}
	excessive := make([]bool, n)
	for _, a := range assets {
		// Reindex onto the grid, the first candle per open time
		pos := make([]int, n)
		for i := range pos {
			pos[i] = -1
		}
		for i, t := range a.Bars.Time {
			if t < minTime || t > maxTime || (t-minTime)%stepMs != 0 {
				continue
			}
			if g := int((t - minTime) / stepMs); pos[g] < 0 {
				pos[g] = i
			}
		}
		missing := make([]bool, n)
		for g, i := range pos {
			missing[g] = i < 0 || math.IsNaN(a.Bars.Close[i])
		}
		// A gap counts against itself and the candle before it
		for g := 0; g < n; {
			end := g + 1
			for end < n && missing[end] {
				end++
			}
			if gap := end - g - boolInt(!missing[g]); gap > maxGapFill {
				for k := g; k < end; k++ {
					excessive[k] = true
				}
			}
			g = end
		}

		for _, field := range Fields {
			src := barField(&a.Bars, field)
			if src == nil {
				continue
			}
			col := make([]float64, n)
			for g, i := range pos {
				if i < 0 {
					col[g] = math.NaN()
				} else {
					col[g] = src[i]
				}
			}
			if filled[field] {
				ffill(col, maxGapFill)
			} else {
				for g := range col {
					if math.IsNaN(col[g]) {
						col[g] = 0
					}
				}
			}
			name := a.Symbol + "_" + field
			names = append(names, name)
			cols[name] = col
		}
	}

	var keep []int
	for g := 0; g < n; g++ {
		if excessive[g] {
			continue
		}
		ok := true
		for _, a := range assets {
			if math.IsNaN(cols[a.Symbol+"_close"][g]) {
				ok = false
				break
			}
		}
		if ok {
			keep = append(keep, g)
		}
	}
	out := Chart{StepMs: stepMs, OpenTime: make([]int64, len(keep)), Names: names, Columns: map[string][]float64{}}
	for j, g := range keep {
		out.OpenTime[j] = grid[g]
	}
	for _, name := range names {
		col := make([]float64, len(keep))
		for j, g := range keep {
			col[j] = cols[name][g]
		}
		out.Columns[name] = col
	}
	return out, nil
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// ffill carries the last value forward over at most limit NaNs in a row (pandas ffill(limit)).
func ffill(col []float64, limit int) {
	last, run := math.NaN(), 0
	for i, v := range col {
		if !math.IsNaN(v) {
			last, run = v, 0
			continue
		}
		run++
		if run <= limit && !math.IsNaN(last) {
			col[i] = last
		}
	}
}

func barField(b *Bars, field string) []float64 {
	switch field {
	case "open":
		return b.Open
	case "high":
		return b.High
	case "low":
		return b.Low
	case "close":
		return b.Close
	case "volume":
		return b.Volume
	case "quote_volume":
		return b.QuoteVolume
	case "count":
		if b.Count == nil {
			return nil
		}
		out := make([]float64, len(b.Count))
		for i, v := range b.Count {
			out[i] = float64(v)
		}
		return out
	case "taker_buy_volume":
		return b.TakerBuyVolume
	case "taker_buy_quote_volume":
		return b.TakerBuyQuoteVolume
	case "funding_rate":
		return b.FundingRate
	}
	return nil
}

// Tail is the newest n candles of the chart.
func (c *Chart) Tail(n int) Chart {
	start := max(len(c.OpenTime)-n, 0)
	return c.slice(start, len(c.OpenTime))
}

func (c *Chart) slice(i, j int) Chart {
	out := Chart{StepMs: c.StepMs, OpenTime: append([]int64(nil), c.OpenTime[i:j]...), Names: c.Names,
		Columns: make(map[string][]float64, len(c.Columns))}
	for name, col := range c.Columns {
		out.Columns[name] = append([]float64(nil), col[i:j]...)
	}
	return out
}

// After is the candles opening after t.
func (c *Chart) After(t int64) Chart {
	i := sort.Search(len(c.OpenTime), func(k int) bool { return c.OpenTime[k] > t })
	return c.slice(i, len(c.OpenTime))
}

// Concat is the chart followed by another with the same columns (later candles).
func (c *Chart) Concat(other *Chart) Chart {
	out := Chart{StepMs: c.StepMs, Names: c.Names, Columns: make(map[string][]float64, len(c.Columns))}
	out.OpenTime = append(append([]int64(nil), c.OpenTime...), other.OpenTime...)
	for name, col := range c.Columns {
		out.Columns[name] = append(append([]float64(nil), col...), other.Columns[name]...)
	}
	return out
}
