// Package setups holds the rule-based setups the backtest replays (not on the dashboard).
//
// Market structure and fair value gaps are found by exact rules (no models):
//
// Swing points: a swing high is a candle whose high is above the swingLen candles before it
// and not below the swingLen candles after it (swing lows mirror this). It is only known
// swingLen candles later, once the right side has closed, so every value below is stored on
// the candle where it first becomes known, never earlier.
//
// Structure: walking forward, a close above the latest unbroken swing high breaks structure up
// and a close below the latest unbroken swing low breaks it down. A break in the direction of
// the current trend is a break of structure (BOS: the trend continues). A break against it is a
// change of character (CHoCH: the trend flips). Closes decide breaks, not wicks. The first break
// of a series sets the trend and counts as a BOS.
//
// Fair value gap (FVG): three candles where the first candle's high is below the third candle's
// low (bullish: price skipped that range on the way up) or its low is above the third candle's
// high (bearish). Known at the third candle's close. A gap stays a zone until a candle closes
// beyond its far side (price sliced through it) or it is older than maxAge candles.
package setups

import "math"

// Directions and break kinds.
const (
	Up    = 1
	Down  = -1
	BOS   = 1
	CHoCH = 2
)

// SwingPoints are masks of swing highs and lows, set on the pivot candle (known swingLen candles later).
func SwingPoints(high, low []float64, swingLen int) (isHigh, isLow []bool) {
	n, k := len(high), swingLen
	isHigh, isLow = make([]bool, n), make([]bool, n)
	if n < 2*k+1 {
		return isHigh, isLow
	}
	for p := k; p < n-k; p++ {
		leftH, rightH := math.Inf(-1), math.Inf(-1)
		leftL, rightL := math.Inf(1), math.Inf(1)
		for j := p - k; j < p; j++ {
			leftH, leftL = math.Max(leftH, high[j]), math.Min(leftL, low[j])
		}
		for j := p + 1; j <= p+k; j++ {
			rightH, rightL = math.Max(rightH, high[j]), math.Min(rightL, low[j])
		}
		isHigh[p] = high[p] > leftH && high[p] >= rightH
		isLow[p] = low[p] < leftL && low[p] <= rightL
	}
	return isHigh, isLow
}

// Structure is trend, breaks and swing levels per candle, each known at that candle's close.
type Structure struct {
	Trend         []int     // 1 up, -1 down, 0 before the first break
	BreakDir      []int     // direction of a break on this candle, else 0
	BreakKind     []int     // BOS / CHoCH, else 0
	BreakLevel    []float64 // the swing level the close broke (NaN without a break)
	LegExtremeIdx []int     // lowest low (up break) or highest high (down break) between the broken swing and the break
	SwingHigh     []float64 // the latest confirmed swing levels (broken or not)
	SwingLow      []float64
	LastBreak     []int // position of the most recent break up to this candle (-1 before the first)
}

// MarketStructure walks the candles once.
func MarketStructure(high, low, close []float64, swingLen int) Structure {
	n := len(close)
	isHigh, isLow := SwingPoints(high, low, swingLen)
	s := Structure{Trend: make([]int, n), BreakDir: make([]int, n), BreakKind: make([]int, n),
		BreakLevel: make([]float64, n), LegExtremeIdx: make([]int, n), SwingHigh: make([]float64, n),
		SwingLow: make([]float64, n), LastBreak: make([]int, n)}
	newHigh, newLow := make([]bool, n+swingLen+1), make([]bool, n+swingLen+1)
	for i := range isHigh {
		if isHigh[i] {
			newHigh[i+swingLen] = true // the candle on which the swing becomes known
		}
		if isLow[i] {
			newLow[i+swingLen] = true
		}
	}
	state, last := 0, -1
	sh, shI, shLive := math.NaN(), -1, false
	sl, slI, slLive := math.NaN(), -1, false
	for t := 0; t < n; t++ {
		s.BreakLevel[t], s.LegExtremeIdx[t] = math.NaN(), -1
		if newHigh[t] {
			shI = t - swingLen
			sh, shLive = high[shI], true
		}
		if newLow[t] {
			slI = t - swingLen
			sl, slLive = low[slI], true
		}
		c := close[t]
		if shLive && c > sh {
			s.BreakDir[t], s.BreakKind[t] = Up, BOS
			if state == Down {
				s.BreakKind[t] = CHoCH
			}
			s.BreakLevel[t], s.LegExtremeIdx[t] = sh, shI+1+argmin(low[shI+1:t+1])
			state, shLive, last = Up, false, t
		} else if slLive && c < sl {
			s.BreakDir[t], s.BreakKind[t] = Down, BOS
			if state == Up {
				s.BreakKind[t] = CHoCH
			}
			s.BreakLevel[t], s.LegExtremeIdx[t] = sl, slI+1+argmax(high[slI+1:t+1])
			state, slLive, last = Down, false, t
		}
		s.Trend[t] = state
		s.SwingHigh[t], s.SwingLow[t] = sh, sl
		s.LastBreak[t] = last
	}
	return s
}

// argmin is the first position of the smallest value (np.argmin).
func argmin(x []float64) int {
	best := 0
	for i, v := range x {
		if v < x[best] {
			best = i
		}
	}
	return best
}

func argmax(x []float64) int {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return best
}

// Gap is one fair value gap: Idx is the third candle (where it becomes known); it is a zone on
// candles Idx .. ActiveUntil - 1 (a close beyond its far side, or maxAge candles after it formed).
type Gap struct {
	Idx         int
	Dir         int
	Top, Bottom float64
	ActiveUntil int
}

// FairValueGaps finds every gap, ordered by Idx.
func FairValueGaps(high, low, close []float64, maxAge int) []Gap {
	n := len(close)
	var gaps []Gap
	for t := 2; t < n; t++ {
		switch {
		case low[t] > high[t-2]:
			gaps = append(gaps, Gap{Idx: t, Dir: Up, Top: low[t], Bottom: high[t-2]})
		case high[t] < low[t-2]:
			gaps = append(gaps, Gap{Idx: t, Dir: Down, Top: low[t-2], Bottom: high[t]})
		}
	}
	for g := range gaps {
		gap := &gaps[g]
		gap.ActiveUntil = gap.Idx + maxAge + 1
		for j := gap.Idx + 1; j < min(gap.Idx+maxAge+1, n); j++ {
			if (gap.Dir == Up && close[j] < gap.Bottom) || (gap.Dir == Down && close[j] > gap.Top) {
				gap.ActiveUntil = j
				break
			}
		}
	}
	return gaps
}
