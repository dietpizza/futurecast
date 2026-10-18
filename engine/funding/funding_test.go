package funding

import (
	"math"
	"testing"

	"deltarisk/engine/candles"
)

const sym = "B-TEST_USDT"

func fill(n int, v float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func TestFundingAgainstItsRecentRange(t *testing.T) {
	out := Read(append(fill(24*89, 1e-4), fill(24, 5e-4)...), 24, Days)
	if math.Abs(out.RatePct-0.05) > 1e-12 || *out.Percentile <= 98 || out.Days != 90 {
		t.Fatalf("%+v", out)
	}
	if Read(fill(10, math.NaN()), 24, Days) != nil {
		t.Fatal("no rates: no reading")
	}
	if one := Read([]float64{2e-4}, 24, Days); one.Percentile != nil || math.Abs(one.MeanPct-0.02) > 1e-12 {
		t.Fatalf("%+v", one)
	}
}

func TestViewNeedsTheHourlyFundingColumn(t *testing.T) {
	hourly := &candles.Chart{StepMs: 3_600_000, Columns: map[string][]float64{sym + "_close": fill(48, 1), sym + "_funding_rate": fill(48, 1e-4)}}
	hourly.OpenTime = make([]int64, 48)
	if v := View(map[string]*candles.Chart{"1h": hourly}, sym); v == nil || math.Abs(v.RatePct-0.01) > 1e-12 {
		t.Fatalf("%+v", v)
	}
	bare := &candles.Chart{StepMs: 3_600_000, OpenTime: hourly.OpenTime, Columns: map[string][]float64{sym + "_close": fill(48, 1)}}
	if View(map[string]*candles.Chart{"1h": bare}, sym) != nil || View(map[string]*candles.Chart{"15m": hourly}, sym) != nil {
		t.Fatal("no funding column or no 1h chart: no reading")
	}
}

func TestPercentileCountsTiesHalf(t *testing.T) {
	if p := PercentileOf([]float64{1, 2, 3, 4}, 3); *p != 62.5 {
		t.Fatal(*p)
	}
	if PercentileOf([]float64{math.NaN()}, 1) != nil {
		t.Fatal("no history")
	}
}
