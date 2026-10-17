package jsonx

import (
	"encoding/json"
	"math"
	"testing"
)

func TestNullsAndStamps(t *testing.T) {
	v := struct {
		A, B, C F
		L, M    Label
		At      Stamp
	}{F(1.5), F(math.NaN()), F(math.Inf(1)), "wild", "", Stamp(1_792_263_600_000)}
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"A":1.5,"B":null,"C":null,"L":"wild","M":null,"At":"2026-10-17 19:00:00+00:00"}`
	if string(b) != want {
		t.Fatalf("%s\nwant %s", b, want)
	}
	var back struct{ B F }
	if err := json.Unmarshal([]byte(`{"B":null}`), &back); err != nil || !math.IsNaN(float64(back.B)) {
		t.Fatal(back, err)
	}
}
