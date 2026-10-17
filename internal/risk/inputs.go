// Package risk says how badly a long or a short opened now can go against the trader within a
// window, in the trader's money: the stop, the leverage and the 1-in-10 bad case.
//
// For a window of h candles (regime.Windows), entry at the close of candle t:
//
//   - Adverse move: for a long (close_t - lowest low of candles t+1..t+h) / close_t, for a short
//     (highest high of t+1..t+h - close_t) / close_t, at least 0. Highs and lows, so a dip
//     inside the window counts. The last h candles have none yet.
//   - Swing unit: a HAR volatility forecast for the next h candles (Corsi 2009: a linear
//     regression of the log of their mean Parkinson variance on the log realized volatility of
//     the last 1, 4, 24 and 96 windows, close-to-close and high-low), as a price fraction over the
//     window. Targets are adverse moves in swing units; that is what lets thousands of past
//     windows at every volatility level say something about a 0.25% or a 3% move.
//   - The model of the adverse move in swing units, per side: the empirical distribution of the
//     training windows' adverse moves (one for all rows; the HAR forecast alone moves it).
//     (LightGBM quantile regressions on the other inputs were dropped on 2026-10-18 with the Go
//     rewrite: they shipped in 9 of 96 window-sides and moved scores by about 2 points.)
//   - Bad case: the 90% quantile. Stop-hit chance: P(adverse move ≥ the stop's price move).
//     Liquidation chance: the same at the isolated-margin liquidation distance.
//   - Risk score, 0-100: the stop-hit chance in percent. (100 × bad case / stop distance was the
//     first idea; at 20x with a -5% margin stop it read 100 for every window from 1h up.)
//
// The stop is on margin (the PnL % CoinDCX shows): a -5% stop at 20x is a 0.25% price move.
//
// WalkForward is the honesty check: expanding training windows, test blocks after the first
// TrainShare of history, a gap of h + GapHours between them so overlapping targets don't leak. A
// reading counts as checked only if the bad case was exceeded in the coverage band of the test
// windows.
//
// Every input at a candle uses that candle's close and earlier ones only. Positions count
// candles, not clock time.
package risk

import (
	"math"
	"sort"
	"time"

	"deltarisk/internal/candles"
	"deltarisk/internal/regime"
	"deltarisk/internal/series"
)

const (
	FundingHours = 8     // Binance settles most perpetuals at 00, 08 and 16 UTC
	LogFloor     = 1e-12 // variance floor before taking logs (flat candles)
	// TakerBps is CoinDCX's USDT-M taker fee with GST, in bps (the setups' SetupSpec.taker_bps).
	TakerBps = 8.85
)

// RoundTripPct is market in and out on CoinDCX USDT-M, GST included.
var RoundTripPct = 2 * TakerBps / 100.0

// HARLags are the HAR inputs' lookbacks, in windows.
var HARLags = []int{1, 4, 24, 96}

// HARColumns are the HAR inputs: log volatility over each lag, close-to-close then high-low.
var HARColumns = func() []string {
	var out []string
	for _, kind := range []string{"rv", "pk"} {
		for _, k := range HARLags {
			out = append(out, kind+"_"+itoa(k))
		}
	}
	return out
}()

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	s := ""
	for ; n > 0; n /= 10 {
		s = string(rune('0'+n%10)) + s
	}
	return s
}

// Frame is named columns, row for row with a chart.
type Frame struct {
	Names []string
	Cols  map[string][]float64
}

func (f *Frame) add(name string, col []float64) {
	if f.Cols == nil {
		f.Cols = map[string][]float64{}
	}
	if _, ok := f.Cols[name]; !ok {
		f.Names = append(f.Names, name)
	}
	f.Cols[name] = col
}

// Col is a column, nil when absent.
func (f *Frame) Col(name string) []float64 { return f.Cols[name] }

// logVol is half the log of a variance floored at LogFloor (NaN stays NaN).
func logVol(variance []float64) []float64 {
	return series.Map(variance, func(v float64) float64 {
		if v < LogFloor {
			v = LogFloor
		}
		return 0.5 * math.Log(v)
	})
}

// Parkinson is one candle's Parkinson variance estimate from its high-low range.
func Parkinson(high, low []float64) []float64 {
	return series.Zip(high, low, func(h, l float64) float64 {
		x := math.Log(h / l)
		return x * x / (4.0 * math.Log(2.0))
	})
}

func squares(x []float64) []float64 { return series.Map(x, func(v float64) float64 { return v * v }) }

// fundingInputs are the 1h chart's funding rate (settled before that hour closed) and its 90-day
// percentile, as known at each candle's close.
func fundingInputs(chart *candles.Chart, minutes int, hourly *candles.Chart, symbol string) (rate, pct []float64) {
	hr := hourly.Col(symbol, "funding_rate")
	hpct := series.Map(series.RollingRankPct(hr, 90*24, 7*24), func(v float64) float64 { return 100.0 * v })
	knownAt := make([]int64, hourly.Len())
	for i, t := range hourly.OpenTime {
		knownAt[i] = t + 3_600_000
	}
	n := chart.Len()
	rate, pct = series.NaNs(n), series.NaNs(n)
	for i, open := range chart.OpenTime {
		close := open + int64(minutes)*60_000
		// merge_asof backward: the last hourly row known at or before the close
		j := sort.Search(len(knownAt), func(k int) bool { return knownAt[k] > close }) - 1
		if j >= 0 {
			rate[i], pct[i] = hr[j], hpct[j]
		}
	}
	return rate, pct
}

// Inputs are the model inputs at every candle of the window's chart.
//
// HAR columns (HARColumns): log per-candle volatility over the last 1, 4, 24 and 96 windows,
// close-to-close (rv) and high-low (pk). Then momentum, efficiency and volatility percentile
// (from reg); the last window's move in typical moves; taker-buy share of the last 1 and 4
// windows; position in the last 24 windows' range; UTC hour of the close and hours to the next
// funding settlement; the funding rate and its percentile when hourly carries it; and the
// benchmark's volatility when the chart has its columns.
func Inputs(chart *candles.Chart, symbol string, w regime.Window, benchmark string, hourly *candles.Chart, reg *regime.Frame) Frame {
	h := w.Candles
	minutes := w.Minutes() / h
	close, high, low := chart.Col(symbol, "close"), chart.Col(symbol, "high"), chart.Col(symbol, "low")
	r := series.Diff(series.Map(close, math.Log))
	pk := Parkinson(high, low)
	var X Frame
	for _, k := range HARLags {
		X.add("rv_"+itoa(k), logVol(series.RollingMean(squares(r), k*h, 0)))
	}
	for _, k := range HARLags {
		X.add("pk_"+itoa(k), logVol(series.RollingMean(pk, k*h, 0)))
	}

	X.add("momentum", reg.Momentum)
	X.add("efficiency", reg.Efficiency)
	X.add("vol_pct", reg.VolPct)
	prev := series.Shift(close, h)
	rvLast := X.Col("rv_" + itoa(HARLags[len(HARLags)-1]))
	move := make([]float64, len(close))
	for i := range close {
		move[i] = math.Log(close[i]/prev[i]) / (math.Exp(rvLast[i]) * math.Sqrt(float64(h)))
	}
	X.add("move", move)

	if chart.Has(symbol, "taker_buy_quote_volume") && chart.Has(symbol, "quote_volume") {
		buy, total := chart.Col(symbol, "taker_buy_quote_volume"), chart.Col(symbol, "quote_volume")
		for _, k := range []int{1, 4} {
			sums := series.RollingSum(total, k*h, 0)
			bought := series.RollingSum(buy, k*h, 0)
			share := series.NaNs(len(close))
			for i := range share {
				if sums[i] > 0 {
					share[i] = bought[i] / sums[i]
				}
			}
			X.add("taker_"+itoa(k), share)
		}
	}

	n := 24 * h
	lo, hi := series.RollingMin(low, n), series.RollingMax(high, n)
	rangePos := make([]float64, len(close))
	for i := range close {
		rangePos[i] = 0.5
		if hi[i] > lo[i] {
			rangePos[i] = (close[i] - lo[i]) / (hi[i] - lo[i])
		}
	}
	X.add("range_pos", rangePos)

	hours, toFunding := make([]float64, len(close)), make([]float64, len(close))
	for i, open := range chart.OpenTime {
		t := time.UnixMilli(open + int64(minutes)*60_000).UTC()
		hours[i] = float64(t.Hour()) + float64(t.Minute())/60.0
		toFunding[i] = FundingHours - math.Mod(hours[i], FundingHours)
	}
	X.add("hour", hours)
	X.add("to_funding", toFunding)

	if !hourly.Empty() && hourly.Has(symbol, "funding_rate") {
		rate, pct := fundingInputs(chart, minutes, hourly, symbol)
		X.add("funding", rate)
		X.add("funding_pct", pct)
	}

	if benchmark != "" && chart.Has(benchmark, "close") {
		rb := series.Diff(series.Map(chart.Col(benchmark, "close"), math.Log))
		X.add("market_rv", logVol(series.RollingMean(squares(rb), 24*h, 0)))
	}
	return X
}

// AdverseMoves are the long and short adverse move over the next h candles from each close
// (NaN for the last h).
func AdverseMoves(chart *candles.Chart, symbol string, h int) (long, short []float64) {
	close, high, low := chart.Col(symbol, "close"), chart.Col(symbol, "high"), chart.Col(symbol, "low")
	lowest := series.Shift(series.RollingMin(low, h), -h)
	highest := series.Shift(series.RollingMax(high, h), -h)
	long, short = make([]float64, len(close)), make([]float64, len(close))
	for i := range close {
		long[i] = max0((close[i] - lowest[i]) / close[i])
		short[i] = max0((highest[i] - close[i]) / close[i])
	}
	return long, short
}

// max0 is clip(lower=0), leaving NaN alone.
func max0(v float64) float64 {
	if v < 0 {
		return 0
	}
	return v
}

// FutureVol is the HAR target: log per-candle Parkinson volatility over the next h candles.
func FutureVol(chart *candles.Chart, symbol string, h int) []float64 {
	pk := Parkinson(chart.Col(symbol, "high"), chart.Col(symbol, "low"))
	return logVol(series.Shift(series.RollingMean(pk, h, 0), -h))
}

// WindowData is everything the model needs for one window, row for row with the chart.
type WindowData struct {
	Window    regime.Window
	OpenTime  []int64
	Inputs    Frame
	Adverse   [2][]float64 // by side
	FutureVol []float64
	Regime    regime.Frame
	Usable    []int // positions with a HAR forecast (finite HAR inputs)
	Trainable []int // usable positions whose targets are known
}

// NewWindowData computes a window's data from the charts (by interval).
func NewWindowData(charts map[string]*candles.Chart, symbol string, w regime.Window, benchmark string, spec regime.Spec) *WindowData {
	chart := charts[w.Interval]
	reg := regime.Compute(chart, symbol, w, spec)
	d := &WindowData{Window: w, OpenTime: chart.OpenTime, Regime: reg,
		Inputs: Inputs(chart, symbol, w, benchmark, charts["1h"], &reg), FutureVol: FutureVol(chart, symbol, w.Candles)}
	d.Adverse[Long], d.Adverse[Short] = AdverseMoves(chart, symbol, w.Candles)
	har := make([][]float64, len(HARColumns))
	for k, name := range HARColumns {
		har[k] = d.Inputs.Col(name)
	}
	for i := range chart.OpenTime {
		ok := true
		for _, col := range har {
			if !series.Finite(col[i]) {
				ok = false
				break
			}
		}
		if !ok {
			continue
		}
		d.Usable = append(d.Usable, i)
		if series.Finite(d.Adverse[Long][i]) && series.Finite(d.Adverse[Short][i]) && series.Finite(d.FutureVol[i]) {
			d.Trainable = append(d.Trainable, i)
		}
	}
	return d
}
