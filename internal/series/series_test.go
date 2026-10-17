package series

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

// F is a float that reads JSON null as NaN.
type F float64

func (f *F) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		*f = F(math.NaN())
		return nil
	}
	var v float64
	err := json.Unmarshal(b, &v)
	*f = F(v)
	return err
}

func floats(v []F) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

type cases struct {
	Inputs  map[string][]F `json:"inputs"`
	Rolling []struct {
		Input      string `json:"input"`
		N          int    `json:"n"`
		MinPeriods int    `json:"min_periods"`
		Sum        []F    `json:"sum"`
		Mean       []F    `json:"mean"`
		Std        []F    `json:"std"`
		RankPct    []F    `json:"rank_pct"`
		Min        []F    `json:"min"`
		Max        []F    `json:"max"`
	} `json:"rolling"`
	EWM []struct {
		Input string  `json:"input"`
		Span  float64 `json:"span"`
		Mean  []F     `json:"mean"`
	} `json:"ewm"`
	Quantile []struct {
		Input  string      `json:"input"`
		Size   int         `json:"size"`
		Q      []float64   `json:"q"`
		Value  []float64   `json:"value"`
		Median float64     `json:"median"`
		Mean   float64     `json:"mean"`
		Search [][]float64 `json:"search"`
	} `json:"quantile"`
}

func load(t *testing.T) cases {
	raw, err := os.ReadFile("../testdata/parity/series.json")
	if err != nil {
		t.Fatal(err)
	}
	var c cases
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	return c
}

// same is equality to rel relative error (NaN equals NaN).
func same(got, want, rel float64) bool {
	if math.IsNaN(got) || math.IsNaN(want) {
		return math.IsNaN(got) && math.IsNaN(want)
	}
	return got == want || math.Abs(got-want) <= rel*math.Max(math.Abs(want), 1e-300)
}

func check(t *testing.T, name string, got []float64, want []F, rel float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	bad := 0
	for i := range got {
		if !same(got[i], float64(want[i]), rel) {
			if bad < 3 {
				t.Errorf("%s[%d] = %.17g, want %.17g", name, i, got[i], float64(want[i]))
			}
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("%s: %d of %d differ", name, bad, len(got))
	}
}

func TestRollingMatchesPandas(t *testing.T) {
	c := load(t)
	for _, r := range c.Rolling {
		x := floats(c.Inputs[r.Input])
		name := fmt.Sprintf("%s n=%d minp=%d", r.Input, r.N, r.MinPeriods)
		check(t, name+" sum", RollingSum(x, r.N, r.MinPeriods), r.Sum, 1e-12)
		check(t, name+" mean", RollingMean(x, r.N, r.MinPeriods), r.Mean, 1e-12)
		check(t, name+" std", RollingStd(x, r.N, r.MinPeriods), r.Std, 1e-9)
		minp := r.MinPeriods
		if minp == 0 {
			minp = r.N
		}
		check(t, name+" rank", RollingRankPct(x, r.N, minp), r.RankPct, 0)
		if r.Min != nil {
			check(t, name+" min", RollingMin(x, r.N), r.Min, 0)
			check(t, name+" max", RollingMax(x, r.N), r.Max, 0)
		}
	}
}

func TestEWMMatchesPandas(t *testing.T) {
	c := load(t)
	for _, e := range c.EWM {
		check(t, fmt.Sprintf("%s span=%g", e.Input, e.Span), EWMMean(floats(c.Inputs[e.Input]), e.Span), e.Mean, 1e-14)
	}
}

func TestQuantilesMatchNumpy(t *testing.T) {
	c := load(t)
	for _, q := range c.Quantile {
		var clean []float64
		for _, v := range floats(c.Inputs[q.Input]) {
			if Finite(v) {
				clean = append(clean, v)
			}
		}
		sortFloats(clean)
		part := clean[:q.Size]
		name := fmt.Sprintf("%s size=%d", q.Input, q.Size)
		for i, a := range q.Q {
			if got := Quantile(part, a); got != q.Value[i] {
				t.Errorf("%s q=%g: %.17g, want %.17g", name, a, got, q.Value[i])
			}
		}
		if got := Median(part); got != q.Median {
			t.Errorf("%s median %.17g, want %.17g", name, got, q.Median)
		}
		if got := Mean(part); got != q.Mean {
			t.Errorf("%s mean %.17g, want %.17g", name, got, q.Mean)
		}
		for _, s := range q.Search {
			if got := SearchLeft(part, s[0]); got != int(s[1]) {
				t.Errorf("%s searchsorted(%g) = %d, want %g", name, s[0], got, s[1])
			}
		}
	}
}

func sortFloats(x []float64) {
	for i := 1; i < len(x); i++ {
		for j := i; j > 0 && x[j] < x[j-1]; j-- {
			x[j], x[j-1] = x[j-1], x[j]
		}
	}
}

func TestSmallHelpers(t *testing.T) {
	x := []float64{1, 2, math.NaN(), 4}
	if s := Shift(x, 1); !math.IsNaN(s[0]) || s[1] != 1 || s[3] != 2 && !math.IsNaN(s[3]) {
		t.Error("shift", s)
	}
	if s := Shift(x, -1); s[0] != 2 || !math.IsNaN(s[3]) {
		t.Error("shift back", s)
	}
	if d := Diff(x); !math.IsNaN(d[0]) || d[1] != 1 || !math.IsNaN(d[2]) {
		t.Error("diff", d)
	}
	if w := Where(x, []bool{true, false, x[2] > 0, true}, 0.5); w[1] != 0.5 || w[2] != 0.5 || w[3] != 4 {
		t.Error("where", w) // a comparison with NaN is false, so NaN gets the other value
	}
	if c := Clip([]float64{-1, math.NaN(), 5}, 0, 2); c[0] != 0 || !math.IsNaN(c[1]) || c[2] != 2 {
		t.Error("clip", c)
	}
	if Sum(nil) != 0 || !math.IsNaN(Mean(nil)) || !math.IsNaN(Quantile(nil, 0.5)) {
		t.Error("empty")
	}
}
