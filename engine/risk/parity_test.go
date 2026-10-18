package risk

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"deltarisk/engine/candles"
	"deltarisk/engine/funding"
	"deltarisk/engine/jsonx"
	"deltarisk/engine/pq"
	"deltarisk/engine/regime"
)

const parity = "../testdata/parity"

type parityInputs struct {
	Symbol    string `json:"symbol"`
	Benchmark string `json:"benchmark"`
}

func readJSON(t *testing.T, name string, v any) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(parity, name))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
}

// parityCharts are Python's aligned charts.
func parityCharts(t *testing.T) (map[string]*candles.Chart, parityInputs) {
	t.Helper()
	var in parityInputs
	readJSON(t, "inputs.json", &in)
	charts := map[string]*candles.Chart{}
	for _, iv := range []string{"5m", "15m", "1h"} {
		tbl, err := pq.Read(filepath.Join(parity, "aligned", iv+".parquet"))
		if err != nil {
			t.Fatal(err)
		}
		c := &candles.Chart{StepMs: candles.MustIntervalMs(iv), OpenTime: tbl.Col("time").Int64, Columns: map[string][]float64{}}
		for _, col := range tbl.Columns {
			if col.Name != "time" && !strings.HasSuffix(col.Name, "_symbol") {
				c.Names = append(c.Names, col.Name)
				c.Columns[col.Name] = col.Floats()
			}
		}
		charts[iv] = c
	}
	return charts, in
}

func close(got, want, rel float64) bool {
	if math.IsNaN(got) || math.IsNaN(want) {
		return math.IsNaN(got) && math.IsNaN(want)
	}
	if math.IsInf(want, 0) {
		return got == want
	}
	return math.Abs(got-want) <= rel*math.Max(math.Abs(want), 1e-12)
}

func checkCol(t *testing.T, name string, got, want []float64, rel float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: %d rows, want %d", name, len(got), len(want))
		return
	}
	bad, first := 0, -1
	for i := range got {
		if !close(got[i], want[i], rel) {
			if first < 0 {
				first = i
			}
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("%s: %d of %d differ, first [%d] = %.17g, want %.17g", name, bad, len(got), first, got[first], want[first])
	}
}

func near(t *testing.T, name string, got, want, rel float64) {
	t.Helper()
	if !close(got, want, rel) {
		t.Errorf("%s = %.17g, want %.17g", name, got, want)
	}
}

func TestParityRegimeInputsAndTargets(t *testing.T) {
	charts, in := parityCharts(t)
	var meta struct {
		HarColumns []string `json:"har_columns"`
		Windows    map[string]struct {
			Rows, Usable, Trainable int
			Gap                     int      `json:"gap"`
			InputColumns            []string `json:"input_columns"`
		} `json:"windows"`
	}
	readJSON(t, "windows.json", &meta)
	if fmt.Sprint(meta.HarColumns) != fmt.Sprint(HARColumns) {
		t.Fatalf("HAR columns %v, want %v", HARColumns, meta.HarColumns)
	}
	for _, w := range regime.Windows {
		t.Run(w.Name, func(t *testing.T) {
			data := NewWindowData(charts, in.Symbol, w, in.Benchmark, regime.DefaultSpec)
			want, err := pq.Read(filepath.Join(parity, "windows", w.Name+".parquet"))
			if err != nil {
				t.Fatal(err)
			}
			m := meta.Windows[w.Name]
			if fmt.Sprint(data.Inputs.Names) != fmt.Sprint(m.InputColumns) {
				t.Errorf("inputs %v, want %v", data.Inputs.Names, m.InputColumns)
			}
			if len(data.Usable) != m.Usable || len(data.Trainable) != m.Trainable || DefaultSpec.Gap(w) != m.Gap {
				t.Errorf("usable %d/%d, trainable %d/%d, gap %d/%d", len(data.Usable), m.Usable, len(data.Trainable), m.Trainable, DefaultSpec.Gap(w), m.Gap)
			}
			for _, name := range data.Inputs.Names {
				checkCol(t, name, data.Inputs.Col(name), want.Col(name).Floats(), 1e-9)
			}
			checkCol(t, "adverse_long", data.Adverse[Long], want.Col("adverse_long").Floats(), 1e-9)
			checkCol(t, "adverse_short", data.Adverse[Short], want.Col("adverse_short").Floats(), 1e-9)
			checkCol(t, "future_vol", data.FutureVol, want.Col("future_vol").Floats(), 1e-9)
			checkCol(t, "regime_momentum", data.Regime.Momentum, want.Col("regime_momentum").Floats(), 1e-9)
			checkCol(t, "regime_efficiency", data.Regime.Efficiency, want.Col("regime_efficiency").Floats(), 1e-9)
			checkCol(t, "regime_vol_pct", data.Regime.VolPct, want.Col("regime_vol_pct").Floats(), 1e-9)
			for name, got := range map[string][]string{"momentum_label": data.Regime.MomentumLabel,
				"volatility_label": data.Regime.VolatilityLabel, "state": data.Regime.State} {
				col := want.Col("regime_" + name)
				bad := 0
				for i := range got {
					w := col.String[i]
					if col.IsNull(i) {
						w = ""
					}
					if got[i] != w {
						if bad == 0 {
							t.Errorf("regime_%s[%d] = %q, want %q", name, i, got[i], w)
						}
						bad++
					}
				}
				if bad > 0 {
					t.Errorf("regime_%s: %d differ", name, bad)
				}
			}
		})
	}
}

type pySideCheck struct {
	Har struct {
		Coverage             float64 `json:"coverage"`
		PinballPct           float64 `json:"pinball_pct"`
		StopHitPredicted     float64 `json:"stop_hit_predicted"`
		Brier                float64 `json:"brier"`
		LiquidationPredicted float64 `json:"liquidation_predicted"`
	} `json:"har"`
	Model           string   `json:"model"`
	Checked         bool     `json:"checked"`
	StopHitRate     float64  `json:"stop_hit_rate"`
	LiquidationRate float64  `json:"liquidation_rate"`
	Calibration     []Bucket `json:"calibration"`
}

func TestParityWalkForwardAndModel(t *testing.T) {
	charts, in := parityCharts(t)
	var meta struct {
		Windows map[string]struct {
			Model struct {
				HarCoef       []float64            `json:"har_coef"`
				TypicalSwing  float64              `json:"typical_swing"`
				PastQuantiles map[string][]float64 `json:"past_quantiles"`
			} `json:"model"`
		} `json:"windows"`
	}
	readJSON(t, "windows.json", &meta)
	for _, w := range regime.Windows {
		t.Run(w.Name, func(t *testing.T) {
			data := NewWindowData(charts, in.Symbol, w, in.Benchmark, regime.DefaultSpec)
			var want struct {
				Stop, Liquidation float64
				Tests, Folds      int
				TrainFrom         int `json:"train_from"`
				Long, Short       pySideCheck
			}
			readJSON(t, "walk_forward/"+w.Name+".json", &want)
			got, err := WalkForward(data, want.Stop, want.Liquidation, DefaultSpec)
			if err != nil {
				t.Fatal(err)
			}
			if got.Tests != want.Tests || got.Folds != want.Folds || got.TrainFrom != want.TrainFrom {
				t.Errorf("tests %d/%d folds %d/%d train_from %d/%d", got.Tests, want.Tests, got.Folds, want.Folds, got.TrainFrom, want.TrainFrom)
			}
			for side, ws := range []pySideCheck{want.Long, want.Short} {
				g, name := got.Sides[side], SideNames[side]
				near(t, name+" coverage", g.Coverage, ws.Har.Coverage, 1e-9)
				near(t, name+" pinball", g.PinballPct, ws.Har.PinballPct, 1e-6)
				near(t, name+" stop_hit_predicted", g.StopHitPredicted, ws.Har.StopHitPredicted, 1e-9)
				near(t, name+" brier", g.Brier, ws.Har.Brier, 1e-9)
				near(t, name+" liquidation_predicted", g.LiquidationPredicted, ws.Har.LiquidationPredicted, 1e-9)
				near(t, name+" stop_hit_rate", g.StopHitRate, ws.StopHitRate, 0)
				near(t, name+" liquidation_rate", g.LiquidationRate, ws.LiquidationRate, 0)
				if g.Checked != ws.Checked || ws.Model != "har" {
					t.Errorf("%s checked %v, want %v (model %s)", name, g.Checked, ws.Checked, ws.Model)
				}
				if len(g.Calibration) != len(ws.Calibration) {
					t.Errorf("%s calibration %v, want %v", name, g.Calibration, ws.Calibration)
					continue
				}
				for i, b := range ws.Calibration {
					gb := g.Calibration[i]
					if gb.From != b.From || gb.To != b.To || gb.N != b.N || !close(gb.Predicted, b.Predicted, 1e-9) || !close(gb.Actual, b.Actual, 0) {
						t.Errorf("%s bucket %d: %+v, want %+v", name, i, gb, b)
					}
				}
			}

			model, err := Fit(data, data.Trainable, DefaultSpec)
			if err != nil {
				t.Fatal(err)
			}
			m := meta.Windows[w.Name].Model
			for i, c := range m.HarCoef {
				near(t, fmt.Sprintf("har_coef[%d]", i), model.Har.Coef[i], c, 1e-6)
			}
			near(t, "typical_swing", model.TypicalSwing, m.TypicalSwing, 1e-6)
			for side, name := range SideNames {
				q := model.Quantiles(side, PageAlphas)
				for i, v := range m.PastQuantiles[name] {
					near(t, fmt.Sprintf("%s past quantile %g", name, PageAlphas[i]), q[i], v, 1e-6)
				}
			}

			var record regime.TrendRecord
			readJSON(t, "trend_record/"+w.Name+".json", &record)
			gotRecord := regime.ComputeTrendRecord(charts[w.Interval], in.Symbol, w, &data.Regime, RoundTripPct)
			if a, b := mustJSON(gotRecord), mustJSON(record); a != b {
				t.Errorf("trend record %s, want %s", a, b)
			}
		})
	}
}

func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func TestParityCoinViews(t *testing.T) {
	charts, in := parityCharts(t)
	var want struct {
		Benchmark        string  `json:"benchmark"`
		TrainStop        float64 `json:"train_stop"`
		TrainLiquidation float64 `json:"train_liquidation"`
		Views            []struct {
			Leverage, MMR float64
			StopPct       float64 `json:"stop_pct"`
			View          map[string]json.RawMessage
		} `json:"views"`
	}
	readJSON(t, "coin_risk_views.json", &want)
	coin, err := Train(in.Symbol, in.Benchmark, charts, want.TrainStop, want.TrainLiquidation, DefaultSpec, regime.DefaultSpec, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range want.Views {
		got := coin.View(v.Leverage, v.MMR, v.StopPct)
		if len(got) != len(v.View) {
			t.Errorf("%d windows, want %d", len(got), len(v.View))
		}
		for name, raw := range v.View {
			label := fmt.Sprintf("%gx -%g%% %s", v.Leverage, v.StopPct, name)
			var w, g map[string]any
			json.Unmarshal(raw, &w)
			json.Unmarshal([]byte(mustJSON(got[name])), &g)
			w["at"] = jsonx.Stamp(int64(w["at"].(float64))).String() // the fixture keeps times in ms
			delete(w["long"].(map[string]any), "model")
			delete(w["short"].(map[string]any), "model")
			compareJSON(t, label, g, w)
		}
	}

	var fv *funding.Reading
	readJSON(t, "funding_view.json", &fv)
	got := funding.View(charts, in.Symbol)
	if got == nil || !close(got.RatePct, fv.RatePct, 1e-12) || !close(got.MeanPct, fv.MeanPct, 1e-12) ||
		!close(*got.Percentile, *fv.Percentile, 1e-12) || got.Days != fv.Days {
		t.Errorf("funding view %+v, want %+v", got, fv)
	}
}

// compareJSON compares decoded JSON: same keys and nulls, strings and bools exactly, numbers
// to 1e-6 (the HAR fit) relative.
func compareJSON(t *testing.T, path string, got, want any) {
	t.Helper()
	switch w := want.(type) {
	case map[string]any:
		g, ok := got.(map[string]any)
		if !ok {
			t.Errorf("%s: %v, want an object", path, got)
			return
		}
		for k := range g {
			if _, ok := w[k]; !ok {
				t.Errorf("%s.%s: unexpected key", path, k)
			}
		}
		for k, wv := range w {
			gv, ok := g[k]
			if !ok {
				t.Errorf("%s.%s: missing", path, k)
				continue
			}
			compareJSON(t, path+"."+k, gv, wv)
		}
	case []any:
		g, ok := got.([]any)
		if !ok || len(g) != len(w) {
			t.Errorf("%s: %v, want %v", path, got, want)
			return
		}
		for i := range w {
			compareJSON(t, fmt.Sprintf("%s[%d]", path, i), g[i], w[i])
		}
	case float64:
		g, ok := got.(float64)
		if !ok || !close(g, w, 1e-6) {
			t.Errorf("%s = %v, want %v", path, got, want)
		}
	default:
		if got != want {
			t.Errorf("%s = %v, want %v", path, got, want)
		}
	}
}

func TestParityExceedChance(t *testing.T) {
	var want struct {
		Alphas     []float64   `json:"alphas"`
		PageAlphas []float64   `json:"page_alphas"`
		Quantiles  [][]float64 `json:"quantiles"`
		X          []float64   `json:"x"`
		Chance     [][]float64 `json:"chance"`
	}
	readJSON(t, "exceed_chance.json", &want)
	if fmt.Sprint(PageAlphas) != fmt.Sprint(want.PageAlphas) {
		t.Errorf("page alphas %v, want %v", PageAlphas, want.PageAlphas)
	}
	for i, q := range want.Quantiles {
		for j, x := range want.X {
			near(t, fmt.Sprintf("exceed_chance(%v, %g)", q, x), ExceedChance(q, want.Alphas, x), want.Chance[i][j], 1e-12)
		}
	}
}
