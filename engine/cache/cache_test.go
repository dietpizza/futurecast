package cache

import (
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"deltarisk/engine/candles"
)

const step = int64(300_000)

func bars(from, n int64) candles.Bars {
	var b candles.Bars
	for i := int64(0); i < n; i++ {
		t := from + i*step
		p := 100 + float64(i%7)
		b.Append(candles.Bar{Time: t, Open: p, High: p + 1, Low: p - 1, Close: p + 0.5, Volume: 10, QuoteVolume: 1000,
			Count: 42, TakerBuyVolume: 6, TakerBuyQuoteVolume: 600})
	}
	return b
}

func TestCandlesRoundTrip(t *testing.T) {
	c := NewCandles(t.TempDir(), 100)
	if got, span := c.Load("SOLUSDT", "5m"); got.Len() != 0 || span != nil {
		t.Fatal("empty cache")
	}
	b := bars(1000*step, 50)
	c.Store("solusdt", "5m", step, b, 0, b.Time[49])
	got, span := c.Load("SOLUSDT", "5m")
	if !reflect.DeepEqual(got, b) || *span != (Span{0, b.Time[49]}) {
		t.Fatalf("span %+v", span)
	}
	if _, err := os.Stat(filepath.Join(c.Dir, "SOLUSDT_5m.parquet")); err != nil {
		t.Fatal(err)
	}
}

func TestCandlesJoinSpansAndKeepTheNewest(t *testing.T) {
	c := NewCandles(t.TempDir(), 100)
	b := bars(1000*step, 150)
	c.Store("X", "5m", step, b.Slice(0, 60), b.Time[0], b.Time[59])
	c.Store("X", "5m", step, b.Slice(60, 90), b.Time[60], b.Time[89]) // joins: one span
	got, span := c.Load("X", "5m")
	if got.Len() != 90 || *span != (Span{b.Time[0], b.Time[89]}) {
		t.Fatalf("%d candles, %+v", got.Len(), span)
	}

	// An older stretch that cannot join is ignored; a newer one replaces the file
	c.Store("X", "5m", step, bars(0, 10), 0, 9*step)
	if _, span := c.Load("X", "5m"); span.From != b.Time[0] {
		t.Fatal("older stretch replaced the newer one")
	}
	c.Store("X", "5m", step, b.Slice(120, 150), b.Time[120], b.Time[149])
	if got, span := c.Load("X", "5m"); got.Len() != 30 || span.From != b.Time[120] {
		t.Fatalf("%d candles, %+v", got.Len(), span)
	}

	// Past MaxRows the oldest go and the span starts at the first kept candle
	c.Store("X", "5m", step, bars(b.Time[149]+step, 90), b.Time[149]+step, b.Time[149]+90*step)
	got, span = c.Load("X", "5m")
	if got.Len() != 100 || span.From != got.Time[0] || got.Time[0] != b.Time[140] {
		t.Fatalf("%d candles, %+v", got.Len(), span)
	}
}

func TestUnreadableFilesAreIgnored(t *testing.T) {
	c := NewCandles(t.TempDir(), 100)
	os.MkdirAll(c.Dir, 0o755)
	os.WriteFile(filepath.Join(c.Dir, "X_5m.parquet"), []byte("not parquet"), 0o644)
	if got, span := c.Load("X", "5m"); got.Len() != 0 || span != nil {
		t.Fatal("garbage should read as empty")
	}
	f := NewFunding(t.TempDir())
	os.MkdirAll(f.Dir, 0o755)
	os.WriteFile(filepath.Join(f.Dir, "X.parquet"), []byte("not parquet"), 0o644)
	if f.Load("X") != nil {
		t.Fatal("garbage should read as nothing")
	}
}

func TestFundingRoundTrip(t *testing.T) {
	f := NewFunding(t.TempDir())
	if f.Load("SOLUSDT") != nil {
		t.Fatal("empty cache")
	}
	events := candles.Funding{Time: []int64{1, 2, 3}, Rate: []float64{1e-4, -2e-5, 3e-6}}
	f.Store("SOLUSDT", 1, 99, events)
	got := f.Load("SOLUSDT")
	if got == nil || got.CoveredFrom != 1 || got.FetchedAt != 99 || !reflect.DeepEqual(got.Events, events) {
		t.Fatalf("%+v", got)
	}
	f.Store("EMPTY", 5, 6, candles.Funding{})
	if got := f.Load("EMPTY"); got == nil || got.Events.Len() != 0 || got.FetchedAt != 6 {
		t.Fatalf("%+v", got)
	}
}

// The files the Python app wrote (pyarrow) load as they are.
func TestReadsPythonWrittenFiles(t *testing.T) {
	root := "../testdata/parity/cache"
	c := &Candles{Dir: filepath.Join(root, "candles"), MaxRows: 120_000}
	b, span := c.Load("ETHUSDT", "1h")
	if span == nil || b.Len() == 0 || b.Time[0] != span.From || b.Time[b.Len()-1] != span.To {
		t.Fatalf("%d candles, %+v", b.Len(), span)
	}
	for i := range b.Close {
		if math.IsNaN(b.Close[i]) || b.Count[i] <= 0 || b.High[i] < b.Low[i] {
			t.Fatalf("candle %d: %+v", i, b.At(i))
		}
	}
	f := (&Funding{Dir: filepath.Join(root, "funding")}).Load("ETHUSDT")
	if f == nil || f.Events.Len() == 0 || f.FetchedAt == 0 || f.Events.Time[0] < f.CoveredFrom {
		t.Fatalf("%+v", f)
	}
}

// Appending to a copy of a Python-written file keeps every candle it had.
func TestAppendsToPythonWrittenFiles(t *testing.T) {
	dir := t.TempDir()
	copyFile(t, "../testdata/parity/cache/candles/ETHUSDT_1h.parquet", filepath.Join(dir, "ETHUSDT_1h.parquet"))
	c := &Candles{Dir: dir, MaxRows: 120_000}
	before, span := c.Load("ETHUSDT", "1h")
	h := int64(3_600_000)
	next := before.Tail(3)
	for i := range next.Time {
		next.Time[i] += 3 * h
	}
	c.Store("ETHUSDT", "1h", h, next, span.To+h, span.To+3*h)
	after, span2 := c.Load("ETHUSDT", "1h")
	if after.Len() != before.Len()+3 || span2.From != span.From || span2.To != span.To+3*h {
		t.Fatalf("%d → %d candles, %+v → %+v", before.Len(), after.Len(), span, span2)
	}
	if head := after.Slice(0, before.Len()); !reflect.DeepEqual(head, before) {
		t.Fatal("old candles changed")
	}
}

// DELTARISK_CHECK_CACHE=<cache folder> (e.g. ~/.deltarisk/cache) reads every file there and
// appends to a copy of each: a one-off check against the user's real cache.
func TestRealCache(t *testing.T) {
	root := os.Getenv("DELTARISK_CHECK_CACHE")
	if root == "" {
		t.Skip("set DELTARISK_CHECK_CACHE to a cache folder to check it")
	}
	files, _ := filepath.Glob(filepath.Join(root, "candles", "*.parquet"))
	if len(files) == 0 {
		t.Fatal("no candle files")
	}
	tmp := t.TempDir()
	for _, path := range files {
		copyFile(t, path, filepath.Join(tmp, filepath.Base(path)))
		name := filepath.Base(path)
		name = name[:len(name)-len(".parquet")]
		i := len(name) - 1
		for name[i] != '_' {
			i--
		}
		market, interval := name[:i], name[i+1:]
		c := &Candles{Dir: tmp, MaxRows: 120_000}
		b, span := c.Load(market, interval)
		if span == nil || b.Len() == 0 {
			t.Errorf("%s: unreadable", path)
			continue
		}
		stepMs := candles.MustIntervalMs(interval)
		next := b.Tail(1)
		next.Time[0] += stepMs
		c.Store(market, interval, stepMs, next, span.To+stepMs, span.To+stepMs)
		after, span2 := c.Load(market, interval)
		if want := min(b.Len()+1, 120_000); after.Len() != want || span2.To != span.To+stepMs {
			t.Errorf("%s: %d → %d candles", path, b.Len(), after.Len())
		}
		t.Logf("%s: %d candles, %d..%d, appended one", filepath.Base(path), b.Len(), span.From, span.To)
	}
	fundingFiles, _ := filepath.Glob(filepath.Join(root, "funding", "*.parquet"))
	for _, path := range fundingFiles {
		name := filepath.Base(path)
		f := (&Funding{Dir: filepath.Dir(path)}).Load(name[:len(name)-len(".parquet")])
		if f == nil || f.Events.Len() == 0 {
			t.Errorf("%s: unreadable", path)
		}
	}
}

func copyFile(t *testing.T, from, to string) {
	t.Helper()
	in, err := os.Open(from)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	out, err := os.Create(to)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	if _, err := io.Copy(out, in); err != nil {
		t.Fatal(err)
	}
}
