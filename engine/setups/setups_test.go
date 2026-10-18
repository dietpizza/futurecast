package setups

import (
	"math"
	"math/rand/v2"
	"reflect"
	"testing"

	"deltarisk/engine/candles"
)

func walk(n int, seed uint64, minutes int64, start int64) candles.Bars {
	rng := rand.New(rand.NewPCG(seed, 5))
	var b candles.Bars
	p := 100.0
	for i := 0; i < n; i++ {
		open := p
		p *= math.Exp(rng.NormFloat64() * 0.002)
		wick := math.Abs(rng.NormFloat64() * 0.001)
		b.Append(candles.Bar{Time: start + int64(i)*minutes*60_000, Open: open, High: math.Max(open, p) * (1 + wick),
			Low: math.Min(open, p) * (1 - wick), Close: p, Volume: 1})
	}
	return b
}

func TestSwingPointsNeedBothSides(t *testing.T) {
	high := []float64{1, 2, 3, 2, 1, 2, 5, 2, 1}
	low := make([]float64, len(high))
	for i := range high {
		low[i] = high[i] - 1
	}
	isHigh, _ := SwingPoints(high, low, 2)
	var got []int
	for i, h := range isHigh {
		if h {
			got = append(got, i)
		}
	}
	if !reflect.DeepEqual(got, []int{2, 6}) {
		t.Fatal(got)
	}
	// The last two candles cannot be swings yet: their right side has not closed
	h, _ := SwingPoints([]float64{1, 2, 3, 4, 9}, make([]float64, 5), 2)
	if h[3] || h[4] {
		t.Fatal(h)
	}
}

func TestBOSThenCHoCHWithTheirLegs(t *testing.T) {
	high := []float64{10, 11, 12, 11.5, 10.5, 11, 12.5, 12.4, 12, 11}
	low := []float64{9, 10, 11, 10, 9, 9.5, 10.8, 11.5, 10.5, 8.5}
	close := []float64{9.5, 10.5, 11.5, 10.2, 9.2, 10.8, 12.3, 11.8, 10.8, 8.8}
	s := MarketStructure(high, low, close, 1)
	// Close 12.3 clears the swing high at 12 (candle 2): first break, so a BOS that sets the trend up
	if s.BreakDir[6] != Up || s.BreakKind[6] != BOS || s.BreakLevel[6] != 12 || s.LegExtremeIdx[6] != 4 {
		t.Fatal("BOS at 6")
	}
	// Close 8.8 loses the swing low at 9 (candle 4) while the trend is up: a change of character
	if s.BreakDir[9] != Down || s.BreakKind[9] != CHoCH || s.BreakLevel[9] != 9 || s.LegExtremeIdx[9] != 6 {
		t.Fatal("CHoCH at 9")
	}
	if !reflect.DeepEqual(s.Trend, []int{0, 0, 0, 0, 0, 0, 1, 1, 1, -1}) || s.LastBreak[8] != 6 {
		t.Fatal(s.Trend, s.LastBreak)
	}
	for _, i := range []int{0, 1, 2, 3, 4, 5, 7, 8} {
		if s.BreakDir[i] != 0 {
			t.Fatalf("break at %d", i)
		}
	}
}

func TestFairValueGapsAndTheirZones(t *testing.T) {
	high := []float64{10, 12, 13, 12.5, 12, 11, 11}
	low := []float64{9, 10, 10.5, 11, 11.5, 10.6, 9.5}
	close := []float64{9.8, 11.8, 12.8, 12, 11.8, 10.8, 9.8}
	var bull *Gap
	gaps := FairValueGaps(high, low, close, 10)
	for i := range gaps {
		if gaps[i].Dir == Up {
			bull = &gaps[i]
			break
		}
	}
	// candle 0 high to candle 2 low; candle 6 closes below the bottom: no longer a zone there
	if bull == nil || bull.Idx != 2 || bull.Bottom != 10 || bull.Top != 10.5 || bull.ActiveUntil != 6 {
		t.Fatalf("%+v", bull)
	}
	// A gap also ages out
	for _, g := range FairValueGaps(high[:5], low[:5], close[:5], 1) {
		if g.Idx == 2 && g.ActiveUntil != 4 {
			t.Fatalf("%+v", g)
		}
	}
}

func TestStructureAndZonesNeverLookAhead(t *testing.T) {
	b := walk(3000, 3, 1, 0)
	full, part := MarketStructure(b.High, b.Low, b.Close, 3), MarketStructure(b.High[:2000], b.Low[:2000], b.Close[:2000], 3)
	cut := func(s Structure) Structure {
		return Structure{s.Trend[:2000], s.BreakDir[:2000], s.BreakKind[:2000], s.BreakLevel[:2000], s.LegExtremeIdx[:2000],
			s.SwingHigh[:2000], s.SwingLow[:2000], s.LastBreak[:2000]}
	}
	if a, p := cut(full), cut(part); !reflect.DeepEqual(a.Trend, p.Trend) || !reflect.DeepEqual(a.BreakDir, p.BreakDir) ||
		!reflect.DeepEqual(a.LegExtremeIdx, p.LegExtremeIdx) || !reflect.DeepEqual(a.LastBreak, p.LastBreak) {
		t.Fatal("structure changed with later candles")
	}
	// Gaps known by candle 1999 are the same, and so is whether each is still a zone up to there
	known := func(gs []Gap) []Gap {
		var out []Gap
		for _, g := range gs {
			if g.Idx < 2000 {
				g.ActiveUntil = min(g.ActiveUntil, 2000)
				out = append(out, g)
			}
		}
		return out
	}
	if !reflect.DeepEqual(known(FairValueGaps(b.High, b.Low, b.Close, 100)), known(FairValueGaps(b.High[:2000], b.Low[:2000], b.Close[:2000], 100))) {
		t.Fatal("gaps changed with later candles")
	}
}

// resample15 builds 15-minute candles from 1-minute ones.
func resample15(b candles.Bars) candles.Bars {
	var out candles.Bars
	for i := 0; i < b.Len(); {
		bucket := b.Time[i] / 900_000 * 900_000
		c := candles.Bar{Time: bucket, Open: b.Open[i], High: b.High[i], Low: b.Low[i], Close: b.Close[i]}
		for i < b.Len() && b.Time[i]/900_000*900_000 == bucket {
			c.High, c.Low, c.Close = math.Max(c.High, b.High[i]), math.Min(c.Low, b.Low[i]), b.Close[i]
			i++
		}
		out.Append(c)
	}
	return out
}

func TestLiveTailFindsTheSameSetupsAsTheFullHistory(t *testing.T) {
	ltf := walk(12_000, 5, 1, 1_767_225_600_000)
	htf := resample15(ltf)
	spec := DefaultSpec()
	spec.LtfMinutes, spec.HtfMinutes = 1, 15
	full := FindSetups(&ltf, &htf, spec)
	ltfTail, htfTail := ltf.Tail(6000), htf.Tail(400) // what the live buffers held
	tail := FindSetups(&ltfTail, &htfTail, spec)
	recent := ltf.Time[ltf.Len()-1] - 40*3_600_000
	type key struct {
		time                 int64
		side, trend          int
		entry, stop, zoneTop float64
		grade                string
		zone                 bool
	}
	pick := func(ss []Setup) []key {
		var out []key
		for _, s := range ss {
			if s.Time > recent {
				zt := s.ZoneTop
				if math.IsNaN(zt) {
					zt = -1
				}
				out = append(out, key{s.Time, s.Side, s.HTFTrend, s.Entry, s.Stop, zt, s.Grade, s.HTFZone})
			}
		}
		return out
	}
	a, b := pick(full), pick(tail)
	if len(a) <= 10 || !reflect.DeepEqual(a, b) {
		t.Fatalf("%d recent setups; tail matches: %v", len(a), reflect.DeepEqual(a, b))
	}
}

// ───────── setups and their orders ─────────

var start = int64(1_772_409_600_000) // 2026-03-02 00:00 UTC, a Monday

func testSpec() Spec {
	s := DefaultSpec()
	s.LtfMinutes, s.HtfMinutes, s.LtfSwingLen, s.HtfSwingLen, s.FillWindow, s.MaxHold = 1, 15, 1, 1, 5, 20
	return s
}

func frame(high, low, close []float64, start int64, minutes int64) candles.Bars {
	var b candles.Bars
	for i := range close {
		open := close[0]
		if i > 0 {
			open = close[i-1]
		}
		b.Append(candles.Bar{Time: start + int64(i)*minutes*60_000, Open: open, High: high[i], Low: low[i], Close: close[i]})
	}
	return b
}

// aPlusLong: 15m has an up break and a bullish gap at 101.2-102.5; 1m pulls back into it, then a CHoCH up with a gap.
func aPlusLong() (ltf, htf candles.Bars) {
	htf = frame([]float64{100.5, 101.5, 101.0, 101.2, 104.0, 105.0, 105.0, 104.5, 104.8, 105.2},
		[]float64{99.0, 99.5, 100.0, 100.2, 101.0, 102.5, 103.0, 102.6, 103.5, 104.0},
		[]float64{100.0, 101.0, 100.5, 100.8, 103.8, 104.8, 104.0, 103.0, 104.5, 105.0}, start, 15)
	ltf = frame([]float64{104.0, 104.4, 104.1, 103.9, 103.8, 103.3, 103.0, 102.8, 103.0, 104.2, 104.9, 104.6, 104.0, 105.0, 106.0, 106.8, 106.5, 104.0},
		[]float64{103.5, 103.6, 103.2, 103.4, 103.0, 102.6, 102.3, 102.0, 102.4, 102.95, 103.3, 103.9, 103.1, 103.2, 104.8, 105.8, 104.0, 101.5},
		[]float64{103.8, 104.0, 103.4, 103.7, 103.1, 102.7, 102.4, 102.6, 102.9, 104.1, 104.8, 104.0, 103.3, 104.9, 105.9, 106.7, 104.2, 101.6},
		start+90*60_000, 1) // after the 15m gap formed
	return ltf, htf
}

// mirror turns candles upside down around a level close to the prices.
func mirror(b candles.Bars, around float64) candles.Bars {
	var out candles.Bars
	for i := 0; i < b.Len(); i++ {
		c := b.At(i)
		out.Append(candles.Bar{Time: c.Time, Open: around - c.Open, High: around - c.Low, Low: around - c.High,
			Close: around - c.Close, Volume: c.Volume})
	}
	return out
}

func side(ss []Setup, d int) []Setup {
	var out []Setup
	for _, s := range ss {
		if s.Side == d {
			out = append(out, s)
		}
	}
	return out
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestAPlusLongSetupOrdersAndOutcomes(t *testing.T) {
	ltf, htf := aPlusLong()
	spec := testSpec()
	longs := side(FindSetups(&ltf, &htf, spec), Up)
	if len(longs) != 1 {
		t.Fatalf("%d longs", len(longs))
	}
	s := longs[0]
	if s.Idx != 10 || s.BreakIdx != 10 || s.BreakLevel != 104.4 || s.GapBottom != 103.0 || s.GapTop != 103.3 || !near(s.Entry, 103.15) {
		t.Fatalf("%+v", s)
	}
	if s.Extreme != 102.0 || !near(s.Stop, 102.0*(1-2e-4)) || s.HTFTrend != 1 || !s.HTFBias || !s.HTFZone || s.Grade != "A+" ||
		s.ZoneBottom != 101.2 || s.ZoneTop != 102.5 {
		t.Fatalf("%+v", s)
	}
	o := Outcomes(longs, &ltf, spec)[0]
	if o.Fill != "filled" || o.FillIdx != 12 { // low 103.1 trades through 103.15
		t.Fatalf("%+v", o)
	}
	want := map[string]TradeResult{"2": {"won", 1, 2, 14}, "3": {"won", 1, 3, 15}, "4": {"lost", 0, -1, 17}}
	if !reflect.DeepEqual(o.Targets, want) {
		t.Fatalf("%+v", o.Targets)
	}
}

func TestShortsMirrorLongsExactly(t *testing.T) {
	ltf, htf := aPlusLong()
	spec := testSpec()
	long := side(FindSetups(&ltf, &htf, spec), Up)
	mltf, mhtf := mirror(ltf, 206), mirror(htf, 206)
	short := side(FindSetups(&mltf, &mhtf, spec), Down)
	if len(short) != 1 || short[0].Grade != "A+" || !near(short[0].Entry, 206-103.15) {
		t.Fatalf("%+v", short)
	}
	if !reflect.DeepEqual(Outcomes(long, &ltf, spec), Outcomes(short, &mltf, spec)) {
		t.Fatal("outcomes differ")
	}
}

func TestWithoutContextASetupIsGradeB(t *testing.T) {
	ltf, htf := aPlusLong()
	for i := range htf.Close {
		htf.High[i], htf.Low[i], htf.Close[i] = 90, 90, 90 // no trend, no gaps on the 15m chart
	}
	s := side(FindSetups(&ltf, &htf, testSpec()), Up)[0]
	if s.HTFBias || s.HTFZone || s.Grade != "B" {
		t.Fatalf("%+v", s)
	}
}

func TestOrderAndTradeEdgeCases(t *testing.T) {
	spec := DefaultSpec()
	spec.LtfMinutes, spec.HtfMinutes, spec.FillWindow, spec.MaxHold, spec.Targets = 1, 15, 3, 4, []Target{R(3)}
	n := 12
	fill := func(v float64) []float64 {
		out := make([]float64, n)
		for i := range out {
			out[i] = v
		}
		return out
	}
	run := func(idx int, entry, stop float64, h, l []float64) Outcome {
		b := candles.Bars{Time: make([]int64, n), High: h, Low: l, Close: fill(100.8)}
		return Outcomes([]Setup{{Idx: idx, Side: 1, Entry: entry, Stop: stop}}, &b, spec)[0]
	}
	high, low := fill(101), fill(100.5)

	if o := run(0, 99, 98, high, low); o.Fill != "expired" { // never trades back to the entry
		t.Fatal(o.Fill)
	}
	if o := run(10, 99, 98, high, low); o.Fill != "pending" { // window still running at the end of the data
		t.Fatal(o.Fill)
	}
	h, l := fill(101), fill(100.6)
	h[1] = 103.5
	if o := run(0, 100, 99, h, l); o.Fill != "cancelled" || o.OrderEnd != 1 { // runs to 3R before filling
		t.Fatalf("%+v", o)
	}
	l = fill(100.6)
	l[1] = 99.9
	if tr := run(0, 100, 99, high, l).Targets["3"]; tr.Outcome != "timeout" || tr.End != 5 || !near(tr.R, 0.8) {
		t.Fatalf("%+v", tr) // neither level within max_hold
	}
	h, l = fill(101), fill(100.6)
	h[1], l[1] = 104, 98.9
	if tr := run(0, 100, 99, h, l).Targets["3"]; tr.Outcome != "lost" || tr.End != 1 {
		t.Fatalf("%+v", tr) // the fill candle reaches the stop: a loss even if it also reached the target
	}
	h, l = fill(101), fill(100.6)
	h[2], l[1], l[2] = 104, 99.9, 98.9
	if tr := run(0, 100, 99, h, l).Targets["3"]; tr.Outcome != "lost" {
		t.Fatalf("%+v", tr) // target and stop inside one later candle: a loss
	}
	l = fill(100.6)
	l[9] = 99.9
	if tr := run(8, 100, 99, high, l).Targets["3"]; tr.Outcome != "open" || !math.IsNaN(tr.Win) {
		t.Fatalf("%+v", tr) // the hold window runs past the data: still open
	}
}

func TestCostsDependOnHowATradeEnds(t *testing.T) {
	spec := DefaultSpec()
	spec.MakerBps, spec.TakerBps, spec.SlippageBps = 2, 5, 3
	if !near(spec.RoundTrip("won"), 4e-4) || !near(spec.RoundTrip("lost"), 10e-4) || !near(spec.RoundTrip("timeout"), 7e-4) {
		t.Fatal("round trips")
	}
	c := spec.Costs()
	if !near(c.MakerPct, 0.02) || !near(c.TakerPct, 0.05) || !near(c.SlippagePct, 0.03) || !near(c.WonPct, 0.04) ||
		!near(c.LostPct, 0.1) || !near(c.TimeoutPct, 0.07) {
		t.Fatalf("%+v", c)
	}
	if !near(BreakEven(3, 1, 0.08, 0.2), 1.2/(2.92+1.2)) || !(BreakEven(3, 1, 0.08, 0.2) < BreakEven(3, 1, 0.2, 0.2)) ||
		!near(BreakEven(3, 1, 0.2, 0.2), 1.2/4) {
		t.Fatal("break even")
	}
}

func TestBreakevenStopAndMarketEntries(t *testing.T) {
	trade := func(high, low, close []float64, be *Breakeven, atClose bool, maxHold int) (string, float64, int) {
		if close == nil {
			close = make([]float64, len(high))
			for i := range high {
				close[i] = (high[i] + low[i]) / 2
			}
		}
		return ResolveTrade(high, low, close, 0, 100, 99, 1, 2, maxHold, atClose, be)
	}
	be := &Breakeven{AtR: 1.5, Stop: 100.1}
	check := func(name string, gotO string, gotR float64, gotE int, o string, r float64, e int) {
		t.Helper()
		if gotO != o || !near(gotR, r) || gotE != e {
			t.Errorf("%s: (%s, %g, %d), want (%s, %g, %d)", name, gotO, gotR, gotE, o, r, e)
		}
	}
	o, r, e := trade([]float64{100.5, 101.6, 101.0, 103.0}, []float64{99.5, 100.6, 100.05, 101.0}, nil, be, false, 10)
	check("back to the moved stop", o, r, e, "breakeven", 0.1, 2)
	o, r, e = trade([]float64{100.5, 101.6, 101.0, 103.0}, []float64{99.5, 100.6, 100.05, 101.0}, nil, nil, false, 10)
	check("without the move", o, r, e, "won", 2, 3)
	o, r, e = trade([]float64{100.5, 102.2}, []float64{99.5, 100.6}, nil, be, false, 10)
	check("1.5R and 2R in one candle", o, r, e, "won", 2, 1)
	o, r, e = trade([]float64{100.5, 101.6}, []float64{99.5, 98.9}, nil, be, false, 10)
	check("1.5R and the old stop in one candle", o, r, e, "lost", -1, 1)
	o, _, _ = trade([]float64{100.5, 101.6, 102.5}, []float64{99.5, 100.6, 100.0}, nil, be, false, 10)
	if o != "breakeven" {
		t.Error("take-profit and the moved stop in one candle count as the stop", o)
	}
	o, r, e = trade([]float64{100.5, 101.0, 102.5}, []float64{98.0, 100.2, 100.5}, nil, nil, true, 10)
	check("fill at the close", o, r, e, "won", 2, 2)
	o, r, e = trade([]float64{100.5, 101.0, 102.5}, []float64{98.0, 100.2, 100.5}, nil, nil, false, 10)
	check("limit fill", o, r, e, "lost", -1, 0)
	o, r, e = trade([]float64{100.5, 101.0, 101.0}, []float64{99.5, 99.5, 99.5}, []float64{100.0, 100.4, 100.6}, nil, true, 2)
	check("timeout", o, r, e, "timeout", 0.6, 2)
}

func TestTargetKeysAndIntervals(t *testing.T) {
	for _, c := range []struct {
		t    Target
		want string
	}{{R(3), "3"}, {R(2.5), "2.5"}, {Measured, "measured"}} {
		if c.t.Key() != c.want {
			t.Errorf("%+v: %s", c.t, c.t.Key())
		}
	}
	for m, want := range map[int]string{15: "15m", 60: "1h", 240: "4h", 1440: "1d", 90: "90m"} {
		if got := IntervalName(m); got != want {
			t.Errorf("IntervalName(%d) = %s", m, got)
		}
	}
}
