package backtest

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"deltarisk/engine/cache"
	"deltarisk/engine/candles"
	"deltarisk/engine/setups"
	"deltarisk/engine/wedges"
)

const parity = "../testdata/parity"

func readJSON(t *testing.T, name string, v any) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(parity, name))
	if err != nil {
		t.Fatal(err)
	}
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.UseNumber()
	if err := dec.Decode(v); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
}

func fixtureBars(t *testing.T, interval string) *candles.Bars {
	t.Helper()
	c := &cache.Candles{Dir: filepath.Join(parity, "cache", "candles"), MaxRows: 120_000}
	b, span := c.Load("ETHUSDT", interval)
	if span == nil {
		t.Fatal("no fixture candles")
	}
	return &b
}

// plain is a value as decoded JSON (json.Number numbers), for comparing with Python's records.
func plain(t *testing.T, v any) any {
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.UseNumber()
	dec.Decode(&out)
	return out
}

// same compares decoded JSON: numbers to 1e-12 relative (exactly for integers), the rest exactly.
func same(t *testing.T, path string, got, want any) {
	t.Helper()
	switch w := want.(type) {
	case map[string]any:
		g, ok := got.(map[string]any)
		if !ok {
			t.Errorf("%s: %v, want %v", path, got, want)
			return
		}
		for k, wv := range w {
			gv, ok := g[k]
			if !ok {
				t.Errorf("%s.%s: missing (want %v)", path, k, wv)
				continue
			}
			same(t, path+"."+k, gv, wv)
		}
		for k := range g {
			if _, ok := w[k]; !ok {
				t.Errorf("%s.%s: unexpected", path, k)
			}
		}
	case []any:
		g, ok := got.([]any)
		if !ok || len(g) != len(w) {
			t.Errorf("%s: %d items, want %d", path, lenOf(got), len(w))
			return
		}
		for i := range w {
			same(t, fmt.Sprintf("%s[%d]", path, i), g[i], w[i])
		}
	case json.Number:
		g, ok := got.(json.Number)
		if !ok {
			t.Errorf("%s = %v, want %v", path, got, want)
			return
		}
		gf, _ := g.Float64()
		wf, _ := w.Float64()
		if gf != wf && math.Abs(gf-wf) > 1e-12*math.Abs(wf) {
			t.Errorf("%s = %v, want %v", path, g, w)
		}
	default:
		if got != want {
			t.Errorf("%s = %v, want %v", path, got, want)
		}
	}
}

func lenOf(v any) int {
	if l, ok := v.([]any); ok {
		return len(l)
	}
	return -1
}

// outcomeRecords are Outcomes as Python's columns (fill, fill_idx, order_end, outcome_T, win_T, r_T, end_T).
func outcomeRecords(t *testing.T, times []int64, out []setups.Outcome) any {
	var rows []map[string]any
	for i, o := range out {
		row := map[string]any{"time": times[i], "fill": o.Fill, "fill_idx": o.FillIdx, "order_end": o.OrderEnd}
		for key, tr := range o.Targets {
			row["outcome_"+key] = tr.Outcome
			row["win_"+key] = nanNull(tr.Win)
			row["r_"+key] = nanNull(tr.R)
			row["end_"+key] = tr.End
		}
		rows = append(rows, row)
	}
	return plain(t, rows)
}

func nanNull(v float64) any {
	if math.IsNaN(v) {
		return nil
	}
	return v
}

// summaryRecord is the summary with its stats, as Python's dict.
func summaryRecord(t *testing.T, s Summary) any {
	base := plain(t, s).(map[string]any)
	if s.Stats != nil {
		for k, v := range plain(t, s.Stats).(map[string]any) {
			base[k] = v
		}
	}
	return base
}

func setupRecords(t *testing.T, v any) any {
	rows := plain(t, v)
	if rows == nil {
		return []any{}
	}
	for _, r := range rows.([]any) {
		m := r.(map[string]any)
		for k, x := range m {
			if s, ok := x.(string); ok && s == "NaN" {
				m[k] = nil
			}
		}
	}
	return rows
}

func TestParityFVGBacktest(t *testing.T) {
	var want map[string]any
	readJSON(t, "backtest_fvg.json", &want)
	entry, context := fixtureBars(t, "15m"), fixtureBars(t, "1h")
	spec, _ := setups.ForCharts("15m", "1h")
	cfg := Config{Target: setups.R(3), Grades: []string{"A+", "B"}, RiskPct: 1}

	found := setups.FindSetups(entry, context, spec)
	var gotSetups []map[string]any
	for _, s := range found {
		m := plain(t, struct {
			setups.Setup
			ZoneTop    any `json:"zone_top"`
			ZoneBottom any `json:"zone_bottom"`
		}{s, nanNull(s.ZoneTop), nanNull(s.ZoneBottom)}).(map[string]any)
		gotSetups = append(gotSetups, m)
	}
	same(t, "setups", plain(t, gotSetups), want["setups"])
	outcomes := setups.Outcomes(found, entry, spec)
	times := make([]int64, len(found))
	for i, s := range found {
		times[i] = s.Time
	}
	same(t, "outcomes", outcomeRecords(t, times, outcomes), want["outcomes"])

	res := Run(FVGCandidates(found), outcomes, entry.Time, spec, cfg)
	same(t, "summary", summaryRecord(t, res.Summary), want["summary"])
	same(t, "trades", plain(t, res.Trades), want["trades"])
	same(t, "monthly", plain(t, res.Monthly), want["monthly"])
	same(t, "sides", plain(t, res.Sides), want["sides"])
	same(t, "shapes", plain(t, res.Shapes), want["shapes"])
	if len(res.Trades) < 20 {
		t.Fatalf("only %d trades", len(res.Trades))
	}
}

func TestParityWedgeBacktest(t *testing.T) {
	var want map[string]any
	readJSON(t, "backtest_wedge.json", &want)
	bars := fixtureBars(t, "15m")
	spec, _ := wedges.ForChart("15m")
	cfg := Config{Target: setups.R(2), Grades: []string{"A+", "B"}, RiskPct: 1, MaxLeverage: 5}

	found, funnel := wedges.Find(bars, spec)
	same(t, "setups", plain(t, found), want["setups"])
	same(t, "funnel", plain(t, funnel), want["funnel"])
	outcomes := wedges.Outcomes(found, bars, spec)
	times := make([]int64, len(found))
	for i, s := range found {
		times[i] = s.Time
	}
	same(t, "outcomes", outcomeRecords(t, times, outcomes), want["outcomes"])

	res := Wedge(bars, spec, cfg)
	same(t, "summary", summaryRecord(t, res.Summary), want["summary"])
	same(t, "trades", plain(t, res.Trades), want["trades"])
	same(t, "monthly", plain(t, res.Monthly), want["monthly"])
	same(t, "sides", plain(t, res.Sides), want["sides"])
	same(t, "shapes", plain(t, res.Shapes), want["shapes"])
	if len(res.Trades) < 5 {
		t.Fatalf("only %d trades", len(res.Trades))
	}
}
