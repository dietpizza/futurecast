package binance

import (
	"encoding/json"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"deltarisk/internal/binance/binancetest"
	"deltarisk/internal/cache"
	"deltarisk/internal/candles"
	"deltarisk/internal/pq"
)

const parity = "../testdata/parity"

type parityInputs struct {
	Symbol    string         `json:"symbol"`
	Benchmark string         `json:"benchmark"`
	EndMs     int64          `json:"end_ms"`
	Days      map[string]int `json:"days"`
	Spans     map[string]struct {
		Rows int `json:"rows"`
	} `json:"spans"`
}

// copyTree copies the fixture caches, so nothing a test does can touch them.
func copyTree(t *testing.T, from, to string) {
	t.Helper()
	err := filepath.Walk(from, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(from, path)
		dst := filepath.Join(to, rel)
		os.MkdirAll(filepath.Dir(dst), 0o755)
		in, err := os.Open(path)
		if err != nil {
			return err
		}
		defer in.Close()
		out, err := os.Create(dst)
		if err != nil {
			return err
		}
		defer out.Close()
		_, err = io.Copy(out, in)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
}

// The fixture candles, read back through the client and aligned, are Python's aligned charts.
func TestParityCandleHistoryAndAlign(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(parity, "inputs.json"))
	if err != nil {
		t.Fatal(err)
	}
	var in parityInputs
	if err := json.Unmarshal(raw, &in); err != nil {
		t.Fatal(err)
	}
	home := t.TempDir()
	copyTree(t, filepath.Join(parity, "cache"), home)
	f := &binancetest.Fake{Clock: binancetest.NewClock(in.EndMs), Markets: []string{"ETHUSDT", "BTCUSDT"},
		Handler: func(w http.ResponseWriter, r *http.Request) bool {
			if r.URL.Path != "/fapi/v1/exchangeInfo" {
				t.Errorf("unexpected download %s", r.URL)
				w.WriteHeader(500)
				return true
			}
			return false
		}}
	c := client(t, f, Options{Candles: &cache.Candles{Dir: filepath.Join(home, "candles"), MaxRows: 120_000},
		Funding: &cache.Funding{Dir: filepath.Join(home, "funding")}})

	for iv := range in.Days {
		bars := in.Spans["ETHUSDT_"+iv].Rows
		target, _, err := c.CandleHistory(in.Symbol, iv, bars, in.EndMs, iv == "1h", nil)
		if err != nil {
			t.Fatal(err)
		}
		bench, _, err := c.CandleHistory(in.Benchmark, iv, bars, in.EndMs, false, nil)
		if err != nil {
			t.Fatal(err)
		}
		if iv == "1h" {
			want, err := pq.Read(filepath.Join(parity, "funding_known_at.parquet"))
			if err != nil {
				t.Fatal(err)
			}
			compare(t, "funding_known_at", target.Time, map[string][]float64{"funding_rate": target.FundingRate}, want)
		}
		chart, err := candles.Align(candles.MustIntervalMs(iv), 3, candles.Asset{Symbol: in.Symbol, Bars: target},
			candles.Asset{Symbol: in.Benchmark, Bars: bench})
		if err != nil {
			t.Fatal(err)
		}
		want, err := pq.Read(filepath.Join(parity, "aligned", iv+".parquet"))
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "aligned "+iv, chart.OpenTime, chart.Columns, want)
		if chart.Len() == 0 {
			t.Fatalf("%s: empty chart", iv)
		}
	}
}

// compare checks every column of want (except time and the string symbol columns) exactly.
func compare(t *testing.T, name string, times []int64, cols map[string][]float64, want *pq.Table) {
	t.Helper()
	wt := want.Col("time").Int64
	if len(wt) != len(times) {
		t.Fatalf("%s: %d rows, want %d", name, len(times), len(wt))
	}
	for i := range wt {
		if wt[i] != times[i] {
			t.Fatalf("%s: row %d at %d, want %d", name, i, times[i], wt[i])
		}
	}
	n := 0
	for _, wc := range want.Columns {
		if wc.Name == "time" || strings.HasSuffix(wc.Name, "_symbol") {
			continue
		}
		n++
		got, ok := cols[wc.Name]
		if !ok {
			t.Errorf("%s: missing column %s", name, wc.Name)
			continue
		}
		for i, w := range wc.Floats() {
			if g := got[i]; g != w && !(math.IsNaN(g) && math.IsNaN(w)) {
				t.Errorf("%s: %s[%d] = %v, want %v", name, wc.Name, i, g, w)
				break
			}
		}
	}
	if n != len(cols) {
		t.Errorf("%s: %d columns, want %d", name, len(cols), n)
	}
}
