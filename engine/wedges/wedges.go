// Package wedges finds wedge breakouts with a retest by exact rules on one chart (no models),
// and what their trades did.
//
// A long setup (shorts mirror it):
//
//  1. Wedge: the latest two swing highs and the latest two swing lows (setups.SwingPoints,
//     SwingLen candles on each side, so each is known SwingLen candles after it) give a
//     resistance line through the highs and a support line through the lows. They make a wedge
//     when the lines converge (the resistance slope is below the support slope) toward an apex
//     ApexMin to ApexMax candles ahead, the first of the four pivots is at most Window candles
//     back, no candle has closed outside either line since that pivot, and each line has been
//     tested at least MinTouches times (runs of candles whose high, or low, came within
//     TouchTolerance of it; the pivots count). Both lines rising is a rising wedge, both falling a
//     falling wedge, anything else a triangle. Only pivots known by the previous candle's close
//     count, so a wedge is complete before the candle that breaks it.
//  2. Breakout: a candle closes above the resistance line. That uses the wedge up, whatever
//     happens next. Grade "A+" when the breakout candle's volume is above VolumeMult times its
//     VolumeWindow-candle average and its body (close minus open) is bigger than the ATR, else "B".
//  3. Retest: within RetestWindow candles after the breakout, a low comes back to within
//     RetestTolerance above the broken line. A close more than RetestTolerance below the line
//     fails the breakout, and the search starts over.
//  4. Trigger: the retest candle, or the one right after it, closes up (close > open) in the
//     upper half of its range: buyers defended the line. Without one in time the breakout expires.
//  5. Orders: buy at the trigger candle's close (a market order), stop StopATR ATR below the lower
//     of the retest's low and the line, take-profits at the wedge's widest height (at its first
//     pivot) above the entry ("measured") and at Targets multiples of R. Once price reaches
//     BreakevenR R the stop moves to the entry plus fees.
//
// One state machine per chart: while a breakout waits for its retest, no other wedge is looked
// at. Everything is known at the trigger candle's close.
package wedges

import (
	"math"

	"deltarisk/engine/candles"
	"deltarisk/engine/series"
	"deltarisk/engine/setups"
)

const onLine = 1e-9 // closes this close to a line (relative) count as on it, not through it

// Spec holds the wedge rules. Counts are in candles of the chart; tolerances are shares of the price.
type Spec struct {
	Minutes         int // the chart
	SwingLen        int // candles on each side of a pivot
	Window          int // candles from the wedge's first pivot to the breakout, at most
	ApexMin         int // the lines must meet between ApexMin and ApexMax candles after the breakout
	ApexMax         int
	TouchTolerance  float64 // a high (low) this close to a line tests it
	MinTouches      int     // tests of each line, pivots included
	ATRLen          int     // Wilder's average true range
	VolumeWindow    int     // candles in the breakout's volume average (the breakout included)
	VolumeMult      float64 // an A+ breakout's volume over that average
	RetestWindow    int     // candles after the breakout in which price may come back to the line
	RetestTolerance float64 // how far short of the line a retest may stop, and how far back inside a close fails it
	StopATR         float64 // the stop sits this many ATR beyond the retest
	Targets         []setups.Target
	BreakevenAt     *float64 // the stop moves to break-even once price gets this many R; nil keeps it
	MaxHold         int      // candles a trade is held at most
	// Costs per leg, in bps of the entry price, the same CoinDCX fees as the fair-value-gap rules.
	// The entry, a stop and a timeout exit are market orders (taker; a stop also fills past its
	// price), the take-profit a resting limit order (maker).
	MakerBps, TakerBps, SlippageBps float64
}

// DefaultBreakevenR is where the stop moves to break-even by default.
const DefaultBreakevenR = 1.5

// DefaultSpec is wedges on 1h candles.
func DefaultSpec() Spec {
	be := DefaultBreakevenR
	return Spec{Minutes: 60, SwingLen: 5, Window: 60, ApexMin: 10, ApexMax: 30, TouchTolerance: 0.003, MinTouches: 2,
		ATRLen: 14, VolumeWindow: 20, VolumeMult: 1.5, RetestWindow: 5, RetestTolerance: 0.002, StopATR: 0.5,
		Targets: []setups.Target{setups.Measured, setups.R(2), setups.R(2.5)}, BreakevenAt: &be, MaxHold: 240,
		MakerBps: setups.MakerBps, TakerBps: setups.TakerBps, SlippageBps: setups.SlippageBps}
}

// ForChart is the default rules on another chart, e.g. ForChart("4h").
func ForChart(interval string) (Spec, error) {
	s := DefaultSpec()
	var err error
	s.Minutes, err = setups.IntervalMinutes(interval)
	return s, err
}

func (s Spec) Strategy() string        { return "wedge" }
func (s Spec) MarketEntry() bool       { return true } // entries fill at a candle's close, so trades start on the next one
func (s Spec) EntryMinutes() int       { return s.Minutes }
func (s Spec) EntryInterval() string   { return setups.IntervalName(s.Minutes) }
func (s Spec) ContextInterval() string { return "" }
func (s Spec) BreakevenR() *float64    { return s.BreakevenAt }

// RoundTrip is the fees and slippage of a trade as a share of the entry price, by how it ended:
// "won" (taker in, maker out), "lost" or "breakeven" (taker in, taker plus slippage out) or
// "timeout" (taker in and out).
func (s Spec) RoundTrip(outcome string) float64 {
	stop := s.TakerBps + s.SlippageBps
	exit := map[string]float64{"won": s.MakerBps, "lost": stop, "breakeven": stop, "timeout": s.TakerBps}[outcome]
	return (s.TakerBps + exit) / 10_000.0
}

// BreakevenStop is where the stop moves: the entry plus both legs' taker fees, so the exit nets
// nothing but its slippage.
func (s Spec) BreakevenStop(entry float64, side int) float64 {
	return entry * (1.0 + float64(side)*2.0*s.TakerBps/10_000.0)
}

// Costs in percent, for reports.
func (s Spec) Costs() setups.Costs {
	return setups.Costs{MakerPct: s.MakerBps / 100.0, TakerPct: s.TakerBps / 100.0, SlippagePct: s.SlippageBps / 100.0,
		WonPct: s.RoundTrip("won") * 100.0, LostPct: s.RoundTrip("lost") * 100.0, TimeoutPct: s.RoundTrip("timeout") * 100.0}
}

// AverageTrueRange is Wilder's ATR; the first candle's true range is its high minus its low.
func AverageTrueRange(high, low, close []float64, length int) []float64 {
	tr := make([]float64, len(close))
	for i := range close {
		prev := close[max(i-1, 0)]
		tr[i] = math.Max(high[i]-low[i], math.Max(math.Abs(high[i]-prev), math.Abs(low[i]-prev)))
	}
	return series.EWMMeanAlpha(tr, 1.0/float64(length))
}

// wedge is two lines: resistance through swing highs h1 and h2, support through swing lows l1 and l2.
type wedge struct {
	h1, h2, l1, l2    int
	resAtH2, resSlope float64 // price per candle
	supAtL2, supSlope float64
}

func through(h1, h2, l1, l2 int, high, low []float64) wedge {
	return wedge{h1, h2, l1, l2, high[h2], (high[h2] - high[h1]) / float64(h2-h1), low[l2], (low[l2] - low[l1]) / float64(l2-l1)}
}

func (w wedge) start() int       { return min(w.h1, w.l1) }
func (w wedge) converging() bool { return w.resSlope < w.supSlope }

func (w wedge) shape() string {
	switch {
	case w.resSlope > 0 && w.supSlope > 0:
		return "rising wedge"
	case w.resSlope < 0 && w.supSlope < 0:
		return "falling wedge"
	}
	return "triangle"
}

// float64(...) keeps Go from fusing the multiply-add (FMA), which would round unlike numpy.
func (w wedge) res(x int) float64 { return w.resAtH2 + float64(w.resSlope*float64(x-w.h2)) }
func (w wedge) sup(x int) float64 { return w.supAtL2 + float64(w.supSlope*float64(x-w.l2)) }

// line is the line a breakout to side crossed: resistance for longs, support for shorts.
func (w wedge) line(side, x int) float64 {
	if side > 0 {
		return w.res(x)
	}
	return w.sup(x)
}

// holds: no close outside the lines from the first pivot through candle stop - 1.
func (w wedge) holds(close []float64, stop int) bool {
	for x := w.start(); x < stop; x++ {
		c := close[x]
		if c-w.res(x) > float64(onLine*c) || w.sup(x)-c > float64(onLine*c) {
			return false
		}
	}
	return true
}

// touches are separate tests of the resistance and support lines from the first pivot through
// candle stop - 1: runs of candles whose high (low) came within tolerance of the line or past it.
func (w wedge) touches(high, low []float64, stop int, tolerance float64) (int, int) {
	res, sup := 0, 0
	prevRes, prevSup := false, false
	for x := w.start(); x < stop; x++ {
		nearRes := high[x] >= w.res(x)*(1.0-tolerance)
		nearSup := low[x] <= w.sup(x)*(1.0+tolerance)
		if nearRes && !prevRes {
			res++
		}
		if nearSup && !prevSup {
			sup++
		}
		prevRes, prevSup = nearRes, nearSup
	}
	return res, sup
}

// Setup is one triggered wedge setup: the entry is candle Idx's close.
type Setup struct {
	Time          int64   `json:"time"` // open time of candle Idx (ms)
	Idx           int     `json:"idx"`
	Side          int     `json:"side"`
	Shape         string  `json:"shape"`
	Grade         string  `json:"grade"`
	Entry         float64 `json:"entry"`
	Stop          float64 `json:"stop"`
	Risk          float64 `json:"risk"`
	MeasuredR     float64 `json:"measured_r"`
	BreakoutIdx   int     `json:"breakout_idx"`
	RetestIdx     int     `json:"retest_idx"`
	High1Idx      int     `json:"high_1_idx"`
	High2Idx      int     `json:"high_2_idx"`
	Low1Idx       int     `json:"low_1_idx"`
	Low2Idx       int     `json:"low_2_idx"`
	ResSlope      float64 `json:"res_slope"`
	SupSlope      float64 `json:"sup_slope"`
	Line          float64 `json:"line"`
	Height        float64 `json:"height"`
	ApexAhead     float64 `json:"apex_ahead"`
	ResTouches    int     `json:"res_touches"`
	SupTouches    int     `json:"sup_touches"`
	VolumeRatio   float64 `json:"volume_ratio"`
	BodyATR       float64 `json:"body_atr"`
	BreakoutClose float64 `json:"breakout_close"`
}

// Funnel counts how far the patterns got.
type Funnel struct {
	Wedges    int `json:"wedges"`
	Breakouts int `json:"breakouts"`
	Failed    int `json:"failed"`  // closed back inside
	Expired   int `json:"expired"` // no trigger in time
	Triggered int `json:"triggered"`
}

type watch struct {
	idx, side, touch       int
	w                      wedge
	apex                   float64
	resTouches, supTouches int
	volumeRatio, bodyATR   float64
	grade                  string
}

// pivotsKnown is, for each candle, how many pivots are known by its close.
func pivotsKnown(pivots []int, swingLen, n int) []int {
	out := make([]int, n)
	k := 0
	for t := 0; t < n; t++ {
		for k < len(pivots) && pivots[k]+swingLen <= t {
			k++
		}
		out[t] = k
	}
	return out
}

// Find finds every triggered wedge setup in the candles, oldest first, and the funnel.
func Find(bars *candles.Bars, spec Spec) ([]Setup, Funnel) {
	o, h, l, c, v := bars.Open, bars.High, bars.Low, bars.Close, bars.Volume
	n := len(c)
	atr := AverageTrueRange(h, l, c, spec.ATRLen)
	volumeAvg := series.RollingMean(v, spec.VolumeWindow, 1)
	isHigh, isLow := setups.SwingPoints(h, l, spec.SwingLen)
	var highs, lows []int
	for i := range isHigh {
		if isHigh[i] {
			highs = append(highs, i)
		}
		if isLow[i] {
			lows = append(lows, i)
		}
	}
	knownH, knownL := pivotsKnown(highs, spec.SwingLen, n), pivotsKnown(lows, spec.SwingLen, n)
	eps := spec.RetestTolerance

	var funnel Funnel
	var out []Setup
	key := [2]int{-1, -1}
	var cur wedge
	inside, counted := false, false
	var wt *watch // a breakout waiting for its retest
	for t := 1; t < n; t++ {
		if wt != nil {
			b, d, w := wt.idx, wt.side, wt.w
			df := float64(d)
			line := w.line(d, t)
			if (c[t]-float64(line*(1-df*eps)))*df < 0 { // closed back inside (float64: no FMA)
				funnel.Failed++
				wt = nil
			} else {
				reach := l[t]
				if d < 0 {
					reach = h[t]
				}
				touched := t-b <= spec.RetestWindow && (reach-float64(line*(1+df*eps)))*df <= 0
				body := c[t] - l[t]
				if d < 0 {
					body = h[t] - c[t]
				}
				rejected := (c[t]-o[t])*df > 0 && body >= 0.5*(h[t]-l[t])
				if rejected && (touched || wt.touch == t-1) {
					retest := t - 1
					if touched {
						retest = t
					}
					out = append(out, setupRow(bars, t, retest, wt, atr, spec))
					funnel.Triggered++
					wt = nil
				} else if touched {
					wt.touch = t
				} else if t-b >= spec.RetestWindow { // no candle left that could trigger
					funnel.Expired++
					wt = nil
				}
			}
			if wt != nil {
				continue
			}
		}

		// Searching: the wedge drawn through the pivots known by the previous candle's close
		kh, kl := knownH[t-1], knownL[t-1]
		if kh < 2 || kl < 2 {
			continue
		}
		if [2]int{kh, kl} != key {
			key = [2]int{kh, kl}
			cur = through(highs[kh-2], highs[kh-1], lows[kl-2], lows[kl-1], h, l)
			inside, counted = cur.converging() && cur.holds(c, t), false
		}
		if !inside {
			continue
		}
		res, sup := cur.res(t), cur.sup(t)
		up, down := c[t]-res > float64(onLine*c[t]), sup-c[t] > float64(onLine*c[t])
		gap := res - sup
		apex := gap / (cur.supSlope - cur.resSlope)
		if gap > 0 && float64(spec.ApexMin) <= apex && apex <= float64(spec.ApexMax) && t-cur.start() <= spec.Window {
			resT, supT := cur.touches(h, l, t, spec.TouchTolerance)
			if min(resT, supT) >= spec.MinTouches {
				if !counted {
					funnel.Wedges++
					counted = true
				}
				if up || down {
					d := 1
					if !up {
						d = -1
					}
					funnel.Breakouts++
					volumeRatio, bodyATR := math.NaN(), math.NaN()
					if volumeAvg[t] > 0 {
						volumeRatio = v[t] / volumeAvg[t]
					}
					if atr[t] > 0 {
						bodyATR = (c[t] - o[t]) * float64(d) / atr[t]
					}
					grade := "B"
					if volumeRatio > spec.VolumeMult && bodyATR > 1.0 {
						grade = "A+"
					}
					wt = &watch{idx: t, side: d, touch: -1, w: cur, apex: apex, resTouches: resT, supTouches: supT,
						volumeRatio: volumeRatio, bodyATR: bodyATR, grade: grade}
				}
			}
		}
		if up || down {
			inside = false // a close outside uses the wedge up
		}
	}
	return out, funnel
}

// setupRow is the orders of a breakout whose retest (from candle retest) triggered on candle t.
func setupRow(bars *candles.Bars, t, retest int, wt *watch, atr []float64, spec Spec) Setup {
	d, w := wt.side, wt.w
	line := w.line(d, t)
	var stop float64
	if d > 0 {
		lo := math.Inf(1)
		for k := retest; k <= t; k++ {
			lo = math.Min(lo, bars.Low[k])
		}
		stop = math.Min(lo, line) - float64(spec.StopATR*atr[t])
	} else {
		hi := math.Inf(-1)
		for k := retest; k <= t; k++ {
			hi = math.Max(hi, bars.High[k])
		}
		stop = math.Max(hi, line) + float64(spec.StopATR*atr[t])
	}
	entry := bars.Close[t]
	R := math.Abs(entry - stop)
	height := w.res(w.start()) - w.sup(w.start())
	return Setup{Time: bars.Time[t], Idx: t, Side: d, Shape: w.shape(), Grade: wt.grade, Entry: entry, Stop: stop,
		Risk: R / entry, MeasuredR: height / R, BreakoutIdx: wt.idx, RetestIdx: retest, High1Idx: w.h1, High2Idx: w.h2,
		Low1Idx: w.l1, Low2Idx: w.l2, ResSlope: w.resSlope, SupSlope: w.supSlope, Line: line, Height: height,
		ApexAhead: wt.apex, ResTouches: wt.resTouches, SupTouches: wt.supTouches, VolumeRatio: wt.volumeRatio,
		BodyATR: wt.bodyATR, BreakoutClose: bars.Close[wt.idx]}
}

// Outcomes are the per-target outcome of every wedge setup (fill is always "filled", on the
// trigger candle; outcomes can also be "breakeven"). The entry is a market order at the trigger
// candle's close, so the next candle is the first that can reach the stop or a take-profit.
// Targets are fixed R multiples or "measured": the wedge's height from the entry, a different
// multiple of R for every trade.
func Outcomes(found []Setup, bars *candles.Bars, spec Spec) []setups.Outcome {
	out := make([]setups.Outcome, len(found))
	for row, s := range found {
		o := setups.Outcome{Fill: "filled", FillIdx: s.Idx, OrderEnd: -1, Targets: map[string]setups.TradeResult{}}
		var be *setups.Breakeven
		if spec.BreakevenAt != nil {
			be = &setups.Breakeven{AtR: *spec.BreakevenAt, Stop: spec.BreakevenStop(s.Entry, s.Side)}
		}
		for _, T := range spec.Targets {
			m := T.Multiple
			if T.Measured {
				m = s.MeasuredR
			}
			outcome, gross, end := setups.ResolveTrade(bars.High, bars.Low, bars.Close, s.Idx, s.Entry, s.Stop, s.Side, m,
				spec.MaxHold, true, be)
			o.Record(T.Key(), outcome, gross, end)
		}
		out[row] = o
	}
	return out
}
