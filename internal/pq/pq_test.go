package pq

import (
	"math"
	"path/filepath"
	"reflect"
	"testing"
)

func TestWriteAndRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "x.parquet")
	cols := []Column{
		{Name: "time", Int64: []int64{1, 2, 3}},
		{Name: "v", Float64: []float64{1.5, math.NaN(), -2}},
	}
	if err := Write(path, cols, map[string]string{"covered_from": "7"}); err != nil {
		t.Fatal(err)
	}
	tbl, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if tbl.Rows() != 3 || tbl.Meta["covered_from"] != "7" || !reflect.DeepEqual(tbl.Col("time").Int64, []int64{1, 2, 3}) {
		t.Fatalf("%+v", tbl)
	}
	v := tbl.Col("v")
	if v.Float64[0] != 1.5 || !math.IsNaN(v.Float64[1]) || !v.IsNull(1) || v.IsNull(0) || v.Float64[2] != -2 {
		t.Fatalf("%+v", v)
	}
	if tbl.Col("nope") != nil {
		t.Fatal("missing column")
	}
	if err := Write(path, []Column{{Name: "a", Int64: []int64{1}}, {Name: "b", Int64: []int64{}}}, nil); err == nil {
		t.Fatal("ragged columns should fail")
	}
}

// pandas frames written by pyarrow: nullable floats, int64 and string columns.
func TestReadsPandasFrames(t *testing.T) {
	tbl, err := Read("../testdata/parity/windows/8h.parquet")
	if err != nil {
		t.Fatal(err)
	}
	state := tbl.Col("regime_state")
	if state == nil || state.String == nil || !state.IsNull(0) || state.String[tbl.Rows()-1] == "" {
		t.Fatalf("regime_state: %+v", state)
	}
	if f := tbl.Col("rv_96").Floats(); !math.IsNaN(f[0]) || math.IsNaN(f[len(f)-1]) {
		t.Fatal("rv_96 should be NaN during warm-up only")
	}
}
