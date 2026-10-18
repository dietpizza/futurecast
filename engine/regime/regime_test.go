package regime

import (
	"math"
	"math/rand/v2"
	"testing"

	"deltarisk/engine/candles"
)

const (
	sym   = "B-TEST_USDT"
	sigma = 0.002
)

var hour, _ = ByName("1h") // 12 candles of 5m, looking back 36

// chart builds 5m candles from log returns.
func chart(returns []float64) *candles.Chart {
	c := &candles.Chart{StepMs: 300_000, Columns: map[string][]float64{}}
	p := 0.0
	for i, r := range returns {
		p += r
		close := 100 * math.Exp(p)
		c.OpenTime = append(c.OpenTime, 1_767_225_600_000+int64(i)*300_000)
		c.Columns[sym+"_close"] = append(c.Columns[sym+"_close"], close)
		c.Columns[sym+"_high"] = append(c.Columns[sym+"_high"], close*1.001)
		c.Columns[sym+"_low"] = append(c.Columns[sym+"_low"], close*0.999)
	}
	return c
}

func noise(seed uint64, n int) []float64 {
	rng := rand.New(rand.NewPCG(seed, 2))
	out := make([]float64, n)
	for i := range out {
		out[i] = rng.NormFloat64() * sigma
	}
	return out
}

func repeat(vals []float64, times int) []float64 {
	var out []float64
	for range times {
		out = append(out, vals...)
	}
	return out
}

func last(tail []float64) Row {
	c := chart(append(noise(3, 4000), tail...))
	f := Compute(c, sym, hour, DefaultSpec)
	return f.At(c.Len() - 1)
}

func TestWindowsCover5mTo8h(t *testing.T) {
	want := []string{"5m", "15m", "30m", "1h", "2h", "3h", "4h", "8h"}
	for i, w := range Windows {
		unit := 1
		if w.Name[len(w.Name)-1] == 'h' {
			unit = 60
		}
		n := 0
		for _, c := range w.Name[:len(w.Name)-1] {
			n = 10*n + int(c-'0')
		}
		if w.Name != want[i] || w.Minutes() != n*unit {
			t.Errorf("%+v: %d minutes", w, w.Minutes())
		}
	}
}

func TestStraightRiseIsTrendingAndBullish(t *testing.T) {
	r := last(repeat([]float64{sigma}, 36))
	if r.State != "trending" || r.MomentumLabel != "bullish" || math.Abs(r.Efficiency-1) > 1e-9 || r.Momentum != 100 {
		t.Fatalf("%+v", r)
	}
}

func TestStraightFallIsBearish(t *testing.T) {
	if r := last(repeat([]float64{-sigma}, 36)); r.State != "trending" || r.MomentumLabel != "bearish" {
		t.Fatalf("%+v", r)
	}
}

func TestZigzagIsChoppy(t *testing.T) {
	if r := last(repeat([]float64{sigma, -sigma}, 18)); r.State != "choppy" || r.Efficiency >= 0.1 || r.MomentumLabel != "neutral" {
		t.Fatalf("%+v", r)
	}
}

func TestFlatIsQuiet(t *testing.T) {
	if r := last(repeat([]float64{sigma / 50, -sigma / 50}, 18)); r.State != "quiet" || r.VolatilityLabel != "calm" {
		t.Fatalf("%+v", r)
	}
}

func TestSpikeIsWild(t *testing.T) {
	if r := last(repeat([]float64{6 * sigma, -6 * sigma}, 18)); r.State != "wild" || r.VolatilityLabel != "wild" {
		t.Fatalf("%+v", r)
	}
}

func TestNoReadingWithoutHistory(t *testing.T) {
	c := chart(repeat([]float64{sigma}, 20))
	if f := Compute(c, sym, hour, DefaultSpec); f.State[c.Len()-1] != "" {
		t.Fatal(f.State[c.Len()-1])
	}
}

func TestTrendRecordCountsWhatFollowedTrendingReadings(t *testing.T) {
	// A random history, then a long straight rise: its trending-bullish readings are followed by more rise
	c := chart(append(noise(5, 4000), repeat([]float64{sigma}, 200)...))
	f := Compute(c, sym, hour, DefaultSpec)
	out := ComputeTrendRecord(c, sym, hour, &f, 0.177)
	if out.RoundTripPct != 0.177 || out.Long.N <= 100 || *out.Long.MeanPct <= *out.Base.MeanPct || *out.Long.WinRate <= *out.Base.WinRate {
		t.Fatalf("%+v", out)
	}
	base := 0
	for i := 0; i < c.Len()-hour.Candles; i++ {
		if f.State[i] != "" {
			base++
		}
	}
	if out.Base.N != base || out.Short.N >= out.Long.N {
		t.Fatalf("base %d (want %d), short %d", out.Base.N, base, out.Short.N)
	}
	if out.Short.N == 0 && out.Short.WinRate != nil || out.Short.N > 0 && (*out.Short.WinRate < 0 || *out.Short.WinRate > 1) {
		t.Fatalf("%+v", out.Short)
	}
}
