// Package candles holds closed candles: one market's raw candles (Bars, as Binance and the
// cache give them), settled funding rates (Funding), and a coin and its benchmark aligned on one
// time grid (Chart). Times are int64 milliseconds; a candle is keyed by its open time.
package candles

import (
	"fmt"
	"math"
	"sort"
)

// Bars are one market's candles, oldest first, columns of equal length. FundingRate is nil
// unless the candles carry the latest rate settled before each close.
type Bars struct {
	Time                []int64
	Open                []float64
	High                []float64
	Low                 []float64
	Close               []float64
	Volume              []float64
	QuoteVolume         []float64
	Count               []int64
	TakerBuyVolume      []float64
	TakerBuyQuoteVolume []float64
	FundingRate         []float64
}

// Bar is one candle (without funding).
type Bar struct {
	Time                                int64
	Open, High, Low, Close              float64
	Volume, QuoteVolume                 float64
	Count                               int64
	TakerBuyVolume, TakerBuyQuoteVolume float64
}

func (b *Bars) Len() int { return len(b.Time) }

// Append adds one candle at the end.
func (b *Bars) Append(c Bar) {
	b.Time = append(b.Time, c.Time)
	b.Open = append(b.Open, c.Open)
	b.High = append(b.High, c.High)
	b.Low = append(b.Low, c.Low)
	b.Close = append(b.Close, c.Close)
	b.Volume = append(b.Volume, c.Volume)
	b.QuoteVolume = append(b.QuoteVolume, c.QuoteVolume)
	b.Count = append(b.Count, c.Count)
	b.TakerBuyVolume = append(b.TakerBuyVolume, c.TakerBuyVolume)
	b.TakerBuyQuoteVolume = append(b.TakerBuyQuoteVolume, c.TakerBuyQuoteVolume)
}

// At is candle i.
func (b *Bars) At(i int) Bar {
	return Bar{b.Time[i], b.Open[i], b.High[i], b.Low[i], b.Close[i], b.Volume[i], b.QuoteVolume[i],
		b.Count[i], b.TakerBuyVolume[i], b.TakerBuyQuoteVolume[i]}
}

// Take is the candles at the given positions, in that order (funding included when present).
func (b *Bars) Take(idx []int) Bars {
	var out Bars
	for _, i := range idx {
		out.Append(b.At(i))
		if b.FundingRate != nil {
			out.FundingRate = append(out.FundingRate, b.FundingRate[i])
		}
	}
	if out.Time == nil {
		out = Bars{}
	}
	return out
}

// Slice is candles i..j-1.
func (b *Bars) Slice(i, j int) Bars {
	idx := make([]int, 0, j-i)
	for k := i; k < j; k++ {
		idx = append(idx, k)
	}
	return b.Take(idx)
}

// Tail is the newest n candles (all of them when there are fewer).
func (b *Bars) Tail(n int) Bars {
	if n >= b.Len() {
		return b.Slice(0, b.Len())
	}
	return b.Slice(b.Len()-n, b.Len())
}

// Between is the candles opening from..to (inclusive).
func (b *Bars) Between(from, to int64) Bars {
	var idx []int
	for i, t := range b.Time {
		if t >= from && t <= to {
			idx = append(idx, i)
		}
	}
	return b.Take(idx)
}

// Merge concatenates parts in order and keeps one candle per open time, sorted: the last one
// when keepLast (a newer download wins), else the first (pandas drop_duplicates keep="last"/"first").
func Merge(keepLast bool, parts ...Bars) Bars {
	var all Bars
	for _, p := range parts {
		for i := 0; i < p.Len(); i++ {
			all.Append(p.At(i))
		}
	}
	pick := map[int64]int{}
	for i, t := range all.Time {
		if _, seen := pick[t]; !seen || keepLast {
			pick[t] = i
		}
	}
	idx := make([]int, 0, len(pick))
	for _, i := range pick {
		idx = append(idx, i)
	}
	sort.Slice(idx, func(a, b int) bool { return all.Time[idx[a]] < all.Time[idx[b]] })
	return all.Take(idx)
}

// Funding is settled funding rates, oldest first: settlement time (ms) and the rate (a fraction
// paid per funding interval).
type Funding struct {
	Time []int64
	Rate []float64
}

func (f *Funding) Len() int { return len(f.Time) }

// Between is the settlements from..to (inclusive).
func (f *Funding) Between(from, to int64) Funding {
	var out Funding
	for i, t := range f.Time {
		if t >= from && t <= to {
			out.Time = append(out.Time, t)
			out.Rate = append(out.Rate, f.Rate[i])
		}
	}
	return out
}

// MergeFunding concatenates and keeps the first event per time, sorted.
func MergeFunding(parts ...Funding) Funding {
	type ev struct {
		t int64
		r float64
	}
	seen := map[int64]bool{}
	var evs []ev
	for _, p := range parts {
		for i, t := range p.Time {
			if !seen[t] {
				seen[t] = true
				evs = append(evs, ev{t, p.Rate[i]})
			}
		}
	}
	sort.SliceStable(evs, func(a, b int) bool { return evs[a].t < evs[b].t })
	var out Funding
	for _, e := range evs {
		out.Time = append(out.Time, e.t)
		out.Rate = append(out.Rate, e.r)
	}
	return out
}

// KnownAt is, for each candle close time (ms), the latest rate settled strictly before it, NaN
// before the first settlement.
func (f *Funding) KnownAt(closeMs []int64) []float64 {
	out := make([]float64, len(closeMs))
	for i, c := range closeMs {
		j := sort.Search(len(f.Time), func(k int) bool { return f.Time[k] >= c }) - 1
		if j >= 0 {
			out[i] = f.Rate[j]
		} else {
			out[i] = math.NaN()
		}
	}
	return out
}

// IntervalMs converts a candle interval ("1m", "5m", "15m", "1h", "1d") to milliseconds.
func IntervalMs(interval string) (int64, error) {
	if len(interval) < 2 {
		return 0, fmt.Errorf("bad interval %q", interval)
	}
	unit := map[byte]int64{'m': 60_000, 'h': 3_600_000, 'd': 86_400_000}[interval[len(interval)-1]]
	var n int64
	for _, c := range interval[:len(interval)-1] {
		if c < '0' || c > '9' {
			return 0, fmt.Errorf("bad interval %q", interval)
		}
		n = 10*n + int64(c-'0')
	}
	if unit == 0 || n == 0 {
		return 0, fmt.Errorf("bad interval %q", interval)
	}
	return n * unit, nil
}

// MustIntervalMs is IntervalMs for intervals known to be valid.
func MustIntervalMs(interval string) int64 {
	ms, err := IntervalMs(interval)
	if err != nil {
		panic(err)
	}
	return ms
}
