package setups

// Fair-value-gap setups: a higher context chart, a lower entry chart (rules only, no models).
//
// The same rules run on any pair of charts (Spec.LtfMinutes for entries, HtfMinutes for
// context): 15m entries in 1h context by default, the pair that held up best in backtests of
// 1m/15m, 5m/1h, 15m/1h and 15m/4h. A long setup (shorts mirror it):
//
//  1. Context: the context chart's trend is up (its latest structure break was upward), and the
//     entry chart's pullback before the setup reached into a bullish context-chart fair value
//     gap that existed before price got there and that no context candle has closed below
//     since (price was absorbed, not sliced through).
//  2. Shift: an entry-chart change of character (CHoCH) up, i.e. a close above the last swing
//     high while the entry chart's trend was down.
//  3. Gap: the move that broke structure left a bullish entry-chart fair value gap behind (the
//     gap's first candle at or after the pullback's low, its third candle at most GapAfterBreak
//     candles after the CHoCH).
//  4. Orders: a limit buy at the gap's midpoint, the stop just below the pullback's low, and
//     take-profits at Targets multiples of the entry-to-stop distance (R): 2R, 3R and 4R.
//
// Every entry-chart CHoCH that leaves a gap is a setup. Step 1's two checks are recorded, not
// required: a setup passing all four steps is grade "A+", the rest are grade "B". Backtests take
// A+ setups by default. Everything is known at the setup candle's close; outcomes are in Outcomes.

import (
	"fmt"
	"math"
	"sort"
	"strconv"

	"deltarisk/internal/candles"
)

// IntervalMinutes converts a chart interval name to minutes: "15m" -> 15, "4h" -> 240.
func IntervalMinutes(interval string) (int, error) {
	ms, err := candles.IntervalMs(interval)
	return int(ms / 60_000), err
}

// IntervalName converts minutes to a chart interval name: 240 -> "4h", 15 -> "15m".
func IntervalName(minutes int) string {
	for _, u := range []struct {
		unit string
		size int
	}{{"d", 1440}, {"h", 60}} {
		if minutes%u.size == 0 {
			return fmt.Sprintf("%d%s", minutes/u.size, u.unit)
		}
	}
	return fmt.Sprintf("%dm", minutes)
}

// Costs are the per-leg fees and each outcome's round trip, in percent.
type Costs struct {
	MakerPct    float64 `json:"maker_pct"`
	TakerPct    float64 `json:"taker_pct"`
	SlippagePct float64 `json:"slippage_pct"`
	WonPct      float64 `json:"won_pct"`
	LostPct     float64 `json:"lost_pct"`
	TimeoutPct  float64 `json:"timeout_pct"`
}

// Default fees: CoinDCX USDT-M futures, General tier, 0.025% maker and 0.075% taker plus 18% GST
// on fees (https://coindcx.com/vip, checked October 2026); stop slippage is an assumption.
const (
	MakerBps    = 2.95
	TakerBps    = 8.85
	SlippageBps = 2.0
)

// Spec holds the setup rules. Counts are in candles of each chart (entry chart for Ltf*,
// FillWindow and MaxHold, context chart for Htf*), so on higher charts every window scales with
// the chart: the same rules, zoomed out. Comments give the clock time on 15m entries / 1h context.
type Spec struct {
	LtfMinutes    int     // entry chart
	HtfMinutes    int     // context chart
	LtfSwingLen   int     // entry candles on each side of a swing point (internal structure, 45 minutes)
	HtfSwingLen   int     // context candles on each side (5 hours)
	HtfZoneMaxAge int     // context candles a gap stays a zone (12 days)
	LtfGapMaxAge  int     // entry candles an entry-chart gap stays a zone (30 hours)
	GapAfterBreak int     // entry candles after the CHoCH in which its gap may still complete
	ZoneTolerance float64 // the pullback may poke past a zone's far side by this share of its height
	StopBufferBps float64 // the stop sits this far beyond the pullback's extreme
	MinRiskBps    float64 // tighter entry-to-stop distances are skipped: fees would swallow them
	FillWindow    int     // entry candles the limit order waits for a fill (7.5 hours)
	MaxHold       int     // entry candles a filled trade is held at most (2.5 days)
	Targets       []Target
	CancelR       float64 // an unfilled limit is cancelled once price runs this many R away, for every target
	MakerBps      float64 // costs per leg, in bps of the entry price: the entry and the take-profit are
	TakerBps      float64 // resting limit orders (maker); the stop and a timeout exit are market orders
	SlippageBps   float64 // (taker), and a triggered stop also fills past its price
}

// DefaultSpec is 15m entries in 1h context.
func DefaultSpec() Spec {
	return Spec{LtfMinutes: 15, HtfMinutes: 60, LtfSwingLen: 3, HtfSwingLen: 5, HtfZoneMaxAge: 288, LtfGapMaxAge: 120,
		GapAfterBreak: 2, ZoneTolerance: 0.25, StopBufferBps: 2, MinRiskBps: 10, FillWindow: 30, MaxHold: 240,
		Targets: []Target{R(2), R(3), R(4)}, CancelR: 3, MakerBps: MakerBps, TakerBps: TakerBps, SlippageBps: SlippageBps}
}

// ForCharts is the default rules on another pair of charts, e.g. ForCharts("5m", "1h").
func ForCharts(entry, context string) (Spec, error) {
	s := DefaultSpec()
	var err error
	if s.LtfMinutes, err = IntervalMinutes(entry); err != nil {
		return s, err
	}
	s.HtfMinutes, err = IntervalMinutes(context)
	return s, err
}

func (s Spec) Strategy() string        { return "fvg" }
func (s Spec) MarketEntry() bool       { return false } // entries are limit orders filled inside a candle
func (s Spec) EntryMinutes() int       { return s.LtfMinutes }
func (s Spec) EntryInterval() string   { return IntervalName(s.LtfMinutes) }
func (s Spec) ContextInterval() string { return IntervalName(s.HtfMinutes) }
func (s Spec) BreakevenR() *float64    { return nil }

// RoundTrip is the fees and slippage of a filled trade as a share of the entry price, by how it
// ended: "won" (maker in, maker out), "lost" (maker in, taker plus slippage out) or "timeout"
// (maker in, taker out). Both legs are charged on the entry price; the exit price differs from
// it by the trade's move, a rounding error on the fee at these stop sizes.
func (s Spec) RoundTrip(outcome string) float64 {
	exit := map[string]float64{"won": s.MakerBps, "lost": s.TakerBps + s.SlippageBps, "timeout": s.TakerBps}[outcome]
	return (s.MakerBps + exit) / 10_000.0
}

// Costs in percent, for reports.
func (s Spec) Costs() Costs { return costsOf(s.MakerBps, s.TakerBps, s.SlippageBps, s.RoundTrip) }

func costsOf(maker, taker, slippage float64, roundTrip func(string) float64) Costs {
	return Costs{MakerPct: maker / 100.0, TakerPct: taker / 100.0, SlippagePct: slippage / 100.0,
		WonPct: roundTrip("won") * 100.0, LostPct: roundTrip("lost") * 100.0, TimeoutPct: roundTrip("timeout") * 100.0}
}

// Target is a take-profit: a fixed multiple of R, or the wedge's measured move.
type Target struct {
	Measured bool
	Multiple float64
}

// R is a fixed take-profit of m R.
func R(m float64) Target { return Target{Multiple: m} }

// Measured is the wedge's height.
var Measured = Target{Measured: true}

// Key is the column suffix: 3 -> "3", 2.5 -> "2.5", measured -> "measured" (Python's :g).
func (t Target) Key() string {
	if t.Measured {
		return "measured"
	}
	return PyG(t.Multiple)
}

func (t Target) MarshalJSON() ([]byte, error) {
	if t.Measured {
		return []byte(`"measured"`), nil
	}
	return []byte(strconv.FormatFloat(t.Multiple, 'g', -1, 64)), nil
}

func (t *Target) UnmarshalJSON(b []byte) error {
	if string(b) == `"measured"` {
		*t = Measured
		return nil
	}
	v, err := strconv.ParseFloat(string(b), 64)
	*t = R(v)
	return err
}

// PyG formats like Python's {:g}: six significant digits, trailing zeros dropped.
func PyG(v float64) string { return strconv.FormatFloat(v, 'g', 6, 64) }

// Setup is one fair-value-gap setup, complete at candle Idx's close.
type Setup struct {
	Time       int64   `json:"time"` // open time of candle Idx (ms)
	Idx        int     `json:"idx"`
	Side       int     `json:"side"`
	BreakIdx   int     `json:"break_idx"`
	BreakLevel float64 `json:"break_level"`
	Extreme    float64 `json:"extreme"`
	GapTop     float64 `json:"gap_top"`
	GapBottom  float64 `json:"gap_bottom"`
	Entry      float64 `json:"entry"`
	Stop       float64 `json:"stop"`
	Risk       float64 `json:"risk"`
	HTFTrend   int     `json:"htf_trend"`
	HTFBias    bool    `json:"htf_bias"`
	HTFZone    bool    `json:"htf_zone"`
	ZoneTop    float64 `json:"zone_top"`    // NaN without a zone
	ZoneBottom float64 `json:"zone_bottom"` // NaN without a zone
	Grade      string  `json:"grade"`
}

// CloseTimes is each candle's close time in ms (open + minutes).
func CloseTimes(open []int64, minutes int) []int64 {
	out := make([]int64, len(open))
	for i, t := range open {
		out[i] = t + int64(minutes)*60_000
	}
	return out
}

// htfPositions is, for each entry candle, the position of the latest context candle closed by
// its close (-1 if none).
func htfPositions(ltf, htf []int64, spec Spec) []int {
	hc, lc := CloseTimes(htf, spec.HtfMinutes), CloseTimes(ltf, spec.LtfMinutes)
	out := make([]int, len(lc))
	for i, c := range lc {
		out[i] = sort.Search(len(hc), func(k int) bool { return hc[k] > c }) - 1
	}
	return out
}

// FindSetups finds every setup in ltf (entry-chart candles) with its context from htf
// (context-chart candles), oldest first.
func FindSetups(ltf, htf *candles.Bars, spec Spec) []Setup {
	high, low, close := ltf.High, ltf.Low, ltf.Close
	ls := MarketStructure(high, low, close, spec.LtfSwingLen)
	lg := FairValueGaps(high, low, close, spec.LtfGapMaxAge)
	hs := MarketStructure(htf.High, htf.Low, htf.Close, spec.HtfSwingLen)
	hg := FairValueGaps(htf.High, htf.Low, htf.Close, spec.HtfZoneMaxAge)
	pos := htfPositions(ltf.Time, htf.Time, spec)

	byDir := func(gaps []Gap, d int) []Gap {
		var out []Gap
		for _, g := range gaps {
			if g.Dir == d {
				out = append(out, g)
			}
		}
		return out
	}
	gaps := map[int][]Gap{Up: byDir(lg, Up), Down: byDir(lg, Down)}
	zones := map[int][]Gap{Up: byDir(hg, Up), Down: byDir(hg, Down)}
	buffer := spec.StopBufferBps / 10_000.0

	var out []Setup
	for t := range close {
		if ls.BreakKind[t] != CHoCH {
			continue
		}
		d := ls.BreakDir[t]
		extI := ls.LegExtremeIdx[t]

		// 3. a gap left by the move that broke structure
		gs := gaps[d]
		lo := sort.Search(len(gs), func(k int) bool { return gs[k].Idx >= extI+2 })
		hi := sort.Search(len(gs), func(k int) bool { return gs[k].Idx > t+spec.GapAfterBreak })
		if lo >= hi {
			continue
		}
		g := lo // the latest gap by the CHoCH, else the first one after it
		for k := lo; k < hi; k++ {
			if gs[k].Idx <= t {
				g = k
			}
		}
		s := max(t, gs[g].Idx)
		turned := false
		for k := t + 1; k <= s && k < len(close); k++ {
			if ls.BreakDir[k] == -d { // structure turned back before the gap completed
				turned = true
			}
		}
		if turned {
			continue
		}

		// 4. orders
		top, bottom := gs[g].Top, gs[g].Bottom
		var extreme, stop float64
		if d == Up {
			extreme = math.Inf(1)
			for k := extI; k <= s; k++ {
				extreme = math.Min(extreme, low[k])
			}
			stop = extreme * (1.0 - buffer)
		} else {
			extreme = math.Inf(-1)
			for k := extI; k <= s; k++ {
				extreme = math.Max(extreme, high[k])
			}
			stop = extreme * (1.0 + buffer)
		}
		entry := (top + bottom) / 2.0
		R := math.Abs(entry - stop)
		if R/entry < spec.MinRiskBps/10_000.0 {
			continue
		}

		// 1. context as of the setup candle's close
		p, pExt := pos[s], pos[extI]
		row := Setup{Time: ltf.Time[s], Idx: s, Side: d, BreakIdx: t, BreakLevel: ls.BreakLevel[t], Extreme: extreme,
			GapTop: top, GapBottom: bottom, Entry: entry, Stop: stop, Risk: R / entry, ZoneTop: math.NaN(), ZoneBottom: math.NaN()}
		if p >= 0 {
			row.HTFTrend = hs.Trend[p]
		}
		row.HTFBias = row.HTFTrend == d

		// zones that existed before price reached the extreme and still stand at the setup
		zs := zones[d]
		if pExt >= 0 && len(zs) > 0 {
			zlo := sort.Search(len(zs), func(k int) bool { return zs[k].Idx >= p-spec.HtfZoneMaxAge-1 })
			zhi := sort.Search(len(zs), func(k int) bool { return zs[k].Idx > pExt })
			for k := zlo; k < zhi; k++ {
				z := zs[k]
				if z.ActiveUntil <= p {
					continue
				}
				height := z.Top - z.Bottom
				var inside bool
				if d == Up {
					inside = extreme <= z.Top && extreme >= z.Bottom-float64(spec.ZoneTolerance*height)
				} else {
					inside = extreme >= z.Bottom && extreme <= z.Top+float64(spec.ZoneTolerance*height)
				}
				if inside { // the most recent zone holding the extreme wins
					row.HTFZone, row.ZoneTop, row.ZoneBottom = true, z.Top, z.Bottom
				}
			}
		}
		row.Grade = "B"
		if row.HTFBias && row.HTFZone {
			row.Grade = "A+"
		}
		out = append(out, row)
	}
	return out
}
