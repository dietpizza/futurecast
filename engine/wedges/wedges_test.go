package wedges

import (
	"math"
	"math/rand/v2"
	"reflect"
	"testing"

	"deltarisk/engine/candles"
	"deltarisk/engine/setups"
)

// spec: resistance 113 - 0.5 (x - 14) through the highs at candles 2 and 14, support 98 - 0.1 (x - 20)
// through the lows at 8 and 20: a falling wedge whose lines meet at candle 50.
func spec() Spec {
	s := DefaultSpec()
	s.Minutes, s.SwingLen, s.Window, s.ApexMin, s.ApexMax, s.ATRLen, s.VolumeWindow = 60, 2, 40, 3, 40, 3, 5
	s.RetestWindow, s.Targets, s.BreakevenAt, s.MaxHold = 3, []setups.Target{setups.R(2), setups.Measured}, nil, 30
	return s
}

var (
	turns = [][2]float64{{0, 110}, {2, 119}, {8, 99.2}, {14, 113}, {20, 98}, {22, 102}, {23, 110}} // candle 23 breaks out
	after = []float64{110.4, 113.0, 116.0, 119.0, 112.0, 105.0, 103.0}                             // retest and trigger, up past 2R, then down to the stop
)

func interp(x float64) float64 {
	for i := 1; i < len(turns); i++ {
		x0, y0, x1, y1 := turns[i-1][0], turns[i-1][1], turns[i][0], turns[i][1]
		if x <= x1 {
			return y0 + (y1-y0)*(x-x0)/(x1-x0)
		}
	}
	return turns[len(turns)-1][1]
}

// build is the wedge, then the after closes; each candle opens at the last close, no wicks unless given.
func build(after []float64, high, low map[int]float64, breakoutVolume float64) candles.Bars {
	var close []float64
	for i := 0; i < 24; i++ {
		close = append(close, interp(float64(i)))
	}
	close = append(close, after...)
	var b candles.Bars
	for i, c := range close {
		open := close[max(i-1, 0)]
		h, l := math.Max(open, c), math.Min(open, c)
		if v, ok := high[i]; ok {
			h = v
		}
		if v, ok := low[i]; ok {
			l = v
		}
		vol := 100.0
		if i == 23 {
			vol = breakoutVolume
		}
		b.Append(candles.Bar{Time: 1_772_409_600_000 + int64(i)*3_600_000, Open: open, High: h, Low: l, Close: c, Volume: vol})
	}
	return b
}

// retested: candle 24 dips to 108.1, within 0.2% of the broken line (108 there), and closes near its high.
func retested() candles.Bars {
	return build(after, map[int]float64{24: 110.6}, map[int]float64{24: 108.1}, 400)
}

func mirror(b candles.Bars, around float64) candles.Bars {
	var out candles.Bars
	for i := 0; i < b.Len(); i++ {
		c := b.At(i)
		out.Append(candles.Bar{Time: c.Time, Open: around - c.Open, High: around - c.Low, Low: around - c.High, Close: around - c.Close, Volume: c.Volume})
	}
	return out
}

func near(a, b float64) bool { return math.Abs(a-b) <= 1e-9*math.Max(1, math.Abs(b)) }

func TestFallingWedgeBreakoutRetestAndTrigger(t *testing.T) {
	b := retested()
	found, funnel := Find(&b, spec())
	if funnel != (Funnel{Wedges: 1, Breakouts: 1, Triggered: 1}) || len(found) != 1 {
		t.Fatalf("%+v, %d setups", funnel, len(found))
	}
	s := found[0]
	if s.High1Idx != 2 || s.High2Idx != 14 || s.Low1Idx != 8 || s.Low2Idx != 20 || !near(s.ResSlope, -0.5) || !near(s.SupSlope, -0.1) {
		t.Fatalf("%+v", s)
	}
	if s.Shape != "falling wedge" || !near(s.ApexAhead, 27) || s.ResTouches != 2 || s.SupTouches != 2 {
		t.Fatalf("%+v", s) // gap 10.8 closing 0.4 a candle
	}
	if s.BreakoutIdx != 23 || s.BreakoutClose != 110 || !near(s.VolumeRatio, 2.5) || !(s.BodyATR > 1) || s.Grade != "A+" {
		t.Fatalf("%+v", s)
	}
	atr := AverageTrueRange(b.High, b.Low, b.Close, 3)[24]
	if s.Idx != 24 || s.RetestIdx != 24 || s.Side != 1 || s.Entry != 110.4 || !near(s.Line, 108) || !near(s.Stop, 108-0.5*atr) {
		t.Fatalf("%+v", s)
	}
	R := 110.4 - s.Stop
	if !near(s.Height, 19.2) || !near(s.MeasuredR, 19.2/R) { // 119 - 99.8 at candle 2
		t.Fatalf("%+v", s)
	}
	o := Outcomes(found, &b, spec())[0]
	if o.Fill != "filled" || o.FillIdx != 24 {
		t.Fatalf("%+v", o)
	}
	// candle 27 reaches 2R; the measured move (129.6) never comes and candle 29's low of 105 takes the stop
	if tr := o.Targets["2"]; tr.Outcome != "won" || tr.R != 2 || tr.End != 27 {
		t.Fatalf("%+v", tr)
	}
	if tr := o.Targets["measured"]; tr.Outcome != "lost" || tr.R != -1 || tr.End != 29 {
		t.Fatalf("%+v", tr)
	}
}

func TestShortsMirrorLongsExactly(t *testing.T) {
	b := retested()
	longs, _ := Find(&b, spec())
	m := mirror(b, 220)
	shorts, _ := Find(&m, spec())
	if len(shorts) != 1 {
		t.Fatalf("%d shorts", len(shorts))
	}
	l, s := longs[0], shorts[0]
	if s.Side != -1 || s.Shape != "rising wedge" || s.Grade != "A+" || s.Low1Idx != 2 || s.Low2Idx != 14 || s.High1Idx != 8 || s.High2Idx != 20 {
		t.Fatalf("%+v", s)
	}
	if s.Idx != l.Idx || s.BreakoutIdx != l.BreakoutIdx || s.RetestIdx != l.RetestIdx {
		t.Fatal("positions")
	}
	for _, p := range [][2]float64{{s.Entry, 220 - l.Entry}, {s.Stop, 220 - l.Stop}, {s.Line, 220 - l.Line}, {s.Height, l.Height},
		{s.MeasuredR, l.MeasuredR}, {s.ApexAhead, l.ApexAhead}, {s.VolumeRatio, l.VolumeRatio}, {s.BodyATR, l.BodyATR}} {
		if !near(p[0], p[1]) {
			t.Fatalf("%v", p)
		}
	}
	if !reflect.DeepEqual(Outcomes(longs, &b, spec()), Outcomes(shorts, &m, spec())) {
		t.Fatal("outcomes differ")
	}
}

func TestACloseBackInsideFailsTheBreakout(t *testing.T) {
	// Candle 24 closes at 107, more than 0.2% under the line (108): the breakout failed
	b := build([]float64{107, 105, 104, 103}, nil, map[int]float64{24: 106.9}, 400)
	if found, funnel := Find(&b, spec()); len(found) != 0 || funnel.Failed != 1 {
		t.Fatalf("%+v", funnel)
	}
}

func TestABreakoutThatNeverComesBackExpires(t *testing.T) {
	b := build([]float64{113, 115, 117, 119, 121}, nil, nil, 400) // lows stay far above the line
	if found, funnel := Find(&b, spec()); len(found) != 0 || funnel.Expired != 1 {
		t.Fatalf("%+v", funnel)
	}
}

func TestTheTriggerMayComeOneCandleAfterTheRetest(t *testing.T) {
	// Candle 24 touches the line but closes down; candle 25 closes up near its high
	b := build([]float64{109, 110.2, 114, 118, 121}, map[int]float64{25: 110.3}, map[int]float64{24: 108.05, 25: 108.6}, 400)
	found, _ := Find(&b, spec())
	s := found[0]
	atr := AverageTrueRange(b.High, b.Low, b.Close, 3)[25]
	if s.RetestIdx != 24 || s.Idx != 25 || s.Entry != 110.2 || !near(s.Line, 107.5) || !near(s.Stop, 107.5-0.5*atr) {
		t.Fatalf("%+v", s) // the line is below the retest's low of 108.05
	}
}

func TestABreakoutWithoutVolumeIsGradeB(t *testing.T) {
	b := build(after, map[int]float64{24: 110.6}, map[int]float64{24: 108.1}, 100)
	found, _ := Find(&b, spec())
	if !near(found[0].VolumeRatio, 1) || found[0].Grade != "B" {
		t.Fatalf("%+v", found[0])
	}
}

func TestWedgesNeverLookAhead(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 7))
	var b candles.Bars
	p := 100.0
	for i := 0; i < 4000; i++ {
		open := p
		p *= math.Exp(rng.NormFloat64() * 0.004)
		b.Append(candles.Bar{Time: int64(i) * 3_600_000, Open: open, High: math.Max(open, p) * (1 + math.Abs(rng.NormFloat64()*0.002)),
			Low: math.Min(open, p) * (1 - math.Abs(rng.NormFloat64()*0.002)), Close: p, Volume: math.Exp(rng.NormFloat64() * 0.5)})
	}
	full, _ := Find(&b, DefaultSpec())
	if len(full) < 5 {
		t.Fatalf("%d setups", len(full))
	}
	for _, s := range full {
		if s.Shape != "rising wedge" && s.Shape != "falling wedge" && s.Shape != "triangle" {
			t.Fatal(s.Shape)
		}
	}
	for _, cut := range []int{full[2].Idx + 1, full[len(full)-1].Idx, 2000} {
		head := b.Slice(0, cut)
		part, _ := Find(&head, DefaultSpec())
		var want []Setup
		for _, s := range full {
			if s.Idx < cut {
				want = append(want, s)
			}
		}
		if !reflect.DeepEqual(part, want) {
			t.Fatalf("cut %d: %d setups, want %d", cut, len(part), len(want))
		}
	}
}

func TestWedgeCostsStartWithAMarketOrder(t *testing.T) {
	s := DefaultSpec()
	s.MakerBps, s.TakerBps, s.SlippageBps = 2, 5, 3
	if !near(s.RoundTrip("won"), 7e-4) || !near(s.RoundTrip("lost"), 13e-4) || s.RoundTrip("breakeven") != s.RoundTrip("lost") ||
		!near(s.RoundTrip("timeout"), 10e-4) {
		t.Fatal("round trips")
	}
	// The break-even stop covers both legs' fees, so such an exit loses only its slippage
	if !near(s.BreakevenStop(100, 1), 100.1) || !near(s.BreakevenStop(100, -1), 99.9) {
		t.Fatal("break-even stop")
	}
	risk := 0.01
	gross := (s.BreakevenStop(100, 1) - 100) / (100 * risk)
	if !near(gross-s.RoundTrip("breakeven")/risk, -3e-4/risk) {
		t.Fatal("break-even exit")
	}
}
