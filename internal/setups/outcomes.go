package setups

// What happened to each setup's orders.
//
// The limit order waits up to FillWindow candles after the setup candle. It fills on the first
// candle that trades through the entry, and is cancelled if price first runs CancelR R away
// without filling. One cancel level for every target keeps a setup's fill the same whichever
// target is traded. Once filled the trade is left alone ("set and forget"):
//
//   - It wins if price touches the take-profit before the stop. On the fill candle only the stop
//     counts, because the candle cannot say whether its high came before or after the fill.
//   - It loses if the stop is touched first, or both in the same candle (ambiguous counts as a loss).
//   - Touching neither within MaxHold candles is a timeout, closed at that candle's close.
//
// Results are in R, the entry-to-stop distance: a win is +target, a loss -1, before costs.
// Anything the data does not cover yet is left open (NaN), never guessed.

import (
	"math"

	"deltarisk/internal/candles"
)

// TradeResult is how one trade ended for one target.
type TradeResult struct {
	Outcome string  `json:"outcome"` // "open", "won", "lost", "breakeven", "timeout"; "" when never filled
	Win     float64 `json:"win"`     // 1 / 0, NaN unless resolved
	R       float64 `json:"r"`       // gross result in R, NaN unless resolved
	End     int     `json:"end"`     // candle it resolved on, -1 unless resolved
}

// Outcome is what one setup's order did, and each target's trade.
type Outcome struct {
	Fill     string                 // "pending" while the order can still fill, "filled", "expired", "cancelled"
	FillIdx  int                    // -1 unless filled
	OrderEnd int                    // candle an unfilled order stopped waiting on, -1 otherwise
	Targets  map[string]TradeResult // by Target.Key()
}

func newOutcome(targets []Target) Outcome {
	o := Outcome{Fill: "pending", FillIdx: -1, OrderEnd: -1, Targets: map[string]TradeResult{}}
	for _, t := range targets {
		o.Targets[t.Key()] = TradeResult{Win: math.NaN(), R: math.NaN(), End: -1}
	}
	return o
}

// Record stores one target's trade.
func (o *Outcome) Record(key, outcome string, gross float64, end int) {
	r := TradeResult{Outcome: outcome, Win: math.NaN(), R: math.NaN(), End: -1}
	if outcome != "open" {
		r.Win, r.R, r.End = 0, gross, end
		if outcome == "won" {
			r.Win = 1
		}
	}
	o.Targets[key] = r
}

// first is the position of the first true, or len(mask) when there is none.
func first(n int, hit func(i int) bool) int {
	for i := 0; i < n; i++ {
		if hit(i) {
			return i
		}
	}
	return n
}

// Outcomes are the fill and per-target outcome of every setup, on the candles they were found on.
func Outcomes(found []Setup, bars *candles.Bars, spec Spec) []Outcome {
	h, l, c := bars.High, bars.Low, bars.Close
	n := len(c)
	out := make([]Outcome, len(found))
	for row, s := range found {
		o := newOutcome(spec.Targets)
		R := math.Abs(s.Entry - s.Stop)
		start, end := s.Idx+1, min(s.Idx+1+spec.FillWindow, n)
		run := float64(spec.CancelR * R)
		var f, r int
		if s.Side > 0 {
			f = first(max(end-start, 0), func(i int) bool { return l[start+i] <= s.Entry })
			r = first(max(end-start, 0), func(i int) bool { return h[start+i] >= s.Entry+run })
		} else {
			f = first(max(end-start, 0), func(i int) bool { return h[start+i] >= s.Entry })
			r = first(max(end-start, 0), func(i int) bool { return l[start+i] <= s.Entry-run })
		}
		window := max(end-start, 0)
		if f < window && f <= r {
			fill := start + f
			o.Fill, o.FillIdx = "filled", fill
			for _, T := range spec.Targets {
				outcome, gross, at := ResolveTrade(h, l, c, fill, s.Entry, s.Stop, s.Side, T.Multiple, spec.MaxHold, false, nil)
				o.Record(T.Key(), outcome, gross, at)
			}
		} else if r < window {
			o.Fill, o.OrderEnd = "cancelled", start+r
		} else if s.Idx+spec.FillWindow < n {
			o.Fill, o.OrderEnd = "expired", s.Idx+spec.FillWindow
		}
		out[row] = o
	}
	return out
}

// Breakeven moves the stop: from the candle after the one that first reaches AtR R, the stop
// sits at Stop.
type Breakeven struct {
	AtR, Stop float64
}

// ResolveTrade is how a trade filled on candle fill ended: (outcome, gross result in R, candle it ended on).
//
// Outcomes: "won" (take-profit targetR R away touched before the stop), "lost" (the stop first,
// or both in one candle), "breakeven", "timeout" (neither through candle fill + maxHold, closed
// at that candle's close) or "open" (the data ends first; NaN and -1).
//
// A limit fill (atClose false) happens inside the fill candle, which can stop the trade out but
// not win it, since its high may have come before the fill. A fill at the candle's close
// (atClose true) starts the trade on the next candle. With a Breakeven, touching the moved stop
// is a "breakeven" exit (with the take-profit in the same candle too).
func ResolveTrade(high, low, close []float64, fill int, entry, stop float64, side int, targetR float64, maxHold int,
	atClose bool, be *Breakeven) (string, float64, int) {
	n := len(close)
	R := math.Abs(entry - stop)
	firstC := fill
	if atClose {
		firstC = fill + 1
	}
	last := min(fill+maxHold, n-1)
	span := max(last-firstC+1, 0)
	up := side > 0
	target := entry + float64(targetR*R)
	if !up {
		target = entry - float64(targetR*R)
	}
	stopped := func(i int) bool {
		if up {
			return low[firstC+i] <= stop
		}
		return high[firstC+i] >= stop
	}
	hit := func(i int) bool {
		if !atClose && i == 0 {
			return false
		}
		if up {
			return high[firstC+i] >= target
		}
		return low[firstC+i] <= target
	}
	sl, tp := first(span, stopped), first(span, hit)
	exitKind, exitR := "lost", -1.0
	if be != nil {
		level := entry + float64(be.AtR*R)
		if !up {
			level = entry - float64(be.AtR*R)
		}
		reached := first(span, func(i int) bool {
			if !atClose && i == 0 {
				return false
			}
			if up {
				return high[firstC+i] >= level
			}
			return low[firstC+i] <= level
		})
		if reached < sl && reached < tp {
			later := first(span-reached-1, func(i int) bool {
				k := firstC + reached + 1 + i
				if up {
					return low[k] <= be.Stop
				}
				return high[k] >= be.Stop
			})
			sl, exitKind, exitR = reached+1+later, "breakeven", (be.Stop-entry)*float64(side)/R
		}
	}
	if tp < sl && tp < span {
		return "won", targetR, firstC + tp
	}
	if sl < span {
		return exitKind, exitR, firstC + sl
	}
	if fill+maxHold < n {
		return "timeout", (close[fill+maxHold] - entry) * float64(side) / R, fill + maxHold
	}
	return "open", math.NaN(), -1
}

// BreakEven is the win rate at which a filled trade makes nothing after costs: a win earns
// target - winCostR R, a non-win loses lossRatio + lossCostR R.
func BreakEven(target, lossRatio, winCostR, lossCostR float64) float64 {
	lose := lossRatio + lossCostR
	return lose / (target - winCostR + lose)
}
