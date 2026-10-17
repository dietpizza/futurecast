// Package regime gives momentum and market state per trading window, from closed candles. Rules,
// no model: they describe what has happened, not what comes next.
//
// A window (Windows) is h candles of its chart: 5m to 1h on 5m candles, 2h to 4h on 15m, 8h on
// 1h. Each reading looks back over the last Lookback windows, L = lookback × h candles (at least
// MinCandles). With r the one-candle log returns and σ their standard deviation over the trailing
// VolDays:
//
//   - Momentum, -100..+100: the mean of two z-like readings, divided by MomentumScale and
//     clipped: the log return over L candles in units of σ√L, and the slope of an EMA spanning L
//     candles over the last h candles in units of σ√h. Bullish above +Bullish, bearish below -Bullish.
//   - Efficiency (Kaufman's ratio): |net move over L| / Σ|r| over L, 0 (all noise) to 1 (a straight line).
//   - Volatility percentile: the standard deviation of r over L candles, ranked against the same
//     reading over the trailing RankDays. Labelled calm / normal / high / wild.
//   - State: wild if the volatility percentile is ≥ WildPct; else trending if efficiency ≥ the
//     window's threshold and |momentum| ≥ Bullish; else quiet if the percentile is ≤ QuietPct;
//     else choppy. A random walk's efficiency falls like 1/√L, so the threshold is
//     TrendEfficiency at MinCandles and scales with √(MinCandles / L): noise passes equally
//     often in every window.
//
// Every column at a candle uses that candle's close and earlier ones only. Positions count
// candles, not clock time; gaps the aligner dropped are stepped over.
package regime

import (
	"math"

	"deltarisk/internal/candles"
	"deltarisk/internal/series"
)

// Window is a trading window of Candles candles of the Interval chart (1h = 12 candles of 5m).
type Window struct {
	Name     string
	Interval string
	Candles  int
}

// IntervalMinutes is the window's chart in minutes.
func (w Window) IntervalMinutes() int { return int(candles.MustIntervalMs(w.Interval) / 60_000) }

// Minutes is the window's length.
func (w Window) Minutes() int { return w.IntervalMinutes() * w.Candles }

// Windows are the dashboard's windows, shortest first.
var Windows = []Window{
	{"5m", "5m", 1}, {"15m", "5m", 3}, {"30m", "5m", 6}, {"1h", "5m", 12},
	{"2h", "15m", 8}, {"3h", "15m", 12}, {"4h", "15m", 16}, {"8h", "1h", 8},
}

// ByName finds a window.
func ByName(name string) (Window, bool) {
	for _, w := range Windows {
		if w.Name == name {
			return w, true
		}
	}
	return Window{}, false
}

// Spec holds the rules' settings.
type Spec struct {
	Lookback        int     // windows looked back over
	MinCandles      int     //
	VolDays         float64 // σ that scales moves
	RankDays        float64 // volatility percentile against this much history
	MomentumScale   float64 // mean z at which momentum reaches ±100
	Bullish         float64
	TrendEfficiency float64 // at MinCandles; see TrendThreshold
	WildPct         float64
	HighPct         float64
	QuietPct        float64
}

// DefaultSpec is the dashboard's rules.
var DefaultSpec = Spec{Lookback: 3, MinCandles: 12, VolDays: 14, RankDays: 30, MomentumScale: 2, Bullish: 25,
	TrendEfficiency: 0.35, WildPct: 90, HighPct: 75, QuietPct: 25}

// Span is L, the candles a reading looks back over.
func (s Spec) Span(w Window) int { return max(s.Lookback*w.Candles, s.MinCandles) }

// TrendThreshold is the efficiency a trend needs in this window.
func (s Spec) TrendThreshold(w Window) float64 {
	return s.TrendEfficiency * math.Sqrt(float64(s.MinCandles)/float64(s.Span(w)))
}

// CandlesIn converts days to candles of the window's chart (rounded half to even, like Python).
func (s Spec) CandlesIn(days float64, w Window) int {
	return int(math.RoundToEven(days * 1440 / float64(w.IntervalMinutes())))
}

// Frame is the readings at every candle of the chart; a label is "" where its inputs are missing.
type Frame struct {
	Momentum, Efficiency, VolPct          []float64
	MomentumLabel, VolatilityLabel, State []string
}

// Row is one candle's readings.
type Row struct {
	Momentum, Efficiency, VolPct          float64
	MomentumLabel, VolatilityLabel, State string
}

// At is the readings at position i.
func (f *Frame) At(i int) Row {
	return Row{f.Momentum[i], f.Efficiency[i], f.VolPct[i], f.MomentumLabel[i], f.VolatilityLabel[i], f.State[i]}
}

// Compute is the readings at every candle of the chart for the symbol.
func Compute(chart *candles.Chart, symbol string, w Window, spec Spec) Frame {
	p := series.Map(chart.Col(symbol, "close"), math.Log)
	r := series.Diff(p)
	h, L := w.Candles, spec.Span(w)
	volN := spec.CandlesIn(spec.VolDays, w)
	sigma := series.RollingStd(r, volN, max(volN/2, 2*L))

	pL := series.Shift(p, L)
	ema := series.EWMMean(p, float64(L))
	emaH := series.Shift(ema, h)
	n := len(p)
	momentum := make([]float64, n)
	for i := range p {
		zReturn := (p[i] - pL[i]) / (sigma[i] * math.Sqrt(float64(L)))
		zSlope := (ema[i] - emaH[i]) / (sigma[i] * math.Sqrt(float64(h)))
		momentum[i] = 100.0 * ((zReturn + zSlope) / 2.0 / spec.MomentumScale)
	}
	momentum = series.Clip(momentum, -100, 100)

	path := series.RollingSum(series.Map(r, math.Abs), L, 0)
	efficiency := make([]float64, n)
	for i := range p {
		if path[i] > 0 {
			efficiency[i] = math.Abs(p[i]-pL[i]) / path[i]
		} // else 0, NaN path included (pandas .where)
	}

	vol := series.RollingStd(r, L, 0)
	rankN := spec.CandlesIn(spec.RankDays, w)
	volPct := series.Map(series.RollingRankPct(vol, rankN, max(rankN/4, 2*L)), func(v float64) float64 { return 100.0 * v })

	f := Frame{Momentum: momentum, Efficiency: efficiency, VolPct: volPct,
		MomentumLabel: make([]string, n), VolatilityLabel: make([]string, n), State: make([]string, n)}
	thr := spec.TrendThreshold(w)
	for i := range p {
		m, eff, pct := momentum[i], efficiency[i], volPct[i]
		f.MomentumLabel[i] = momentumLabel(m, spec)
		f.VolatilityLabel[i] = volatilityLabel(pct, spec)
		if !math.IsNaN(m) && !math.IsNaN(eff) && !math.IsNaN(pct) {
			switch {
			case pct >= spec.WildPct:
				f.State[i] = "wild"
			case eff >= thr && math.Abs(m) >= spec.Bullish:
				f.State[i] = "trending"
			case pct <= spec.QuietPct:
				f.State[i] = "quiet"
			default:
				f.State[i] = "choppy"
			}
		}
	}
	return f
}

func momentumLabel(m float64, spec Spec) string {
	switch {
	case math.IsNaN(m):
		return ""
	case m > spec.Bullish:
		return "bullish"
	case m < -spec.Bullish:
		return "bearish"
	}
	return "neutral"
}

func volatilityLabel(pct float64, spec Spec) string {
	switch {
	case math.IsNaN(pct):
		return ""
	case pct >= spec.WildPct:
		return "wild"
	case pct >= spec.HighPct:
		return "high"
	case pct <= spec.QuietPct:
		return "calm"
	}
	return "normal"
}

// Record is how the next window went after one kind of reading.
type Record struct {
	N       int      `json:"n"`
	WinRate *float64 `json:"win_rate"`
	MeanPct *float64 `json:"mean_pct"`
}

// TrendRecord is the in-sample record behind the page's Long / Short call.
type TrendRecord struct {
	Long         Record  `json:"long"`
	Short        Record  `json:"short"`
	Base         Record  `json:"base"`
	RoundTripPct float64 `json:"round_trip_pct"`
}

// ComputeTrendRecord is how the next window went after past trending readings, per side,
// against every window (the call adds a risk check this ignores).
//
// Long: candles labelled trending and bullish; short: trending and bearish. For each, N readings
// with a known next window (overlapping: about N / h independent ones), WinRate (share where the
// next h candles closed the call's way) and MeanPct (mean simple return in the call's direction,
// before fees, against roundTripPct). Base is the same over every reading: the coin's plain drift.
func ComputeTrendRecord(chart *candles.Chart, symbol string, w Window, f *Frame, roundTripPct float64) TrendRecord {
	close := chart.Col(symbol, "close")
	next := series.Shift(close, -w.Candles)
	n := len(close)
	fwd := make([]float64, n)
	known := make([]bool, n)
	for i := range close {
		fwd[i] = next[i]/close[i] - 1.0
		known[i] = series.Finite(fwd[i]) && f.State[i] != ""
	}
	record := func(want func(i int) bool, sign float64) Record {
		var moves []float64
		for i := range fwd {
			if want(i) && known[i] {
				moves = append(moves, sign*fwd[i])
			}
		}
		if len(moves) == 0 {
			return Record{}
		}
		up := make([]float64, len(moves))
		for i, m := range moves {
			if m > 0 {
				up[i] = 1
			}
		}
		win, mean := series.Mean(up), 100.0*series.Mean(moves)
		return Record{N: len(moves), WinRate: &win, MeanPct: &mean}
	}
	return TrendRecord{
		Long:         record(func(i int) bool { return f.State[i] == "trending" && f.MomentumLabel[i] == "bullish" }, 1),
		Short:        record(func(i int) bool { return f.State[i] == "trending" && f.MomentumLabel[i] == "bearish" }, -1),
		Base:         record(func(int) bool { return true }, 1),
		RoundTripPct: roundTripPct,
	}
}
