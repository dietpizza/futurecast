package binance

import (
	"errors"
	"math"
	"net/http"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
	"time"

	"deltarisk/internal/binance/binancetest"
	"deltarisk/internal/cache"
	"deltarisk/internal/catalog"
)

const (
	step   = int64(300_000) // 5m
	minute = int64(60_000)
	hour   = int64(3_600_000)
	day    = int64(86_400_000)
)

var now = int64(1_700_000_000_000)/step*step + 120_000 // two minutes into a 5m candle

var exchangeInfo = []map[string]any{
	{"symbol": "SOLUSDT", "status": "TRADING", "contractType": "PERPETUAL"},
	{"symbol": "1000PEPEUSDT", "status": "TRADING", "contractType": "PERPETUAL"},
	{"symbol": "XAUUSDT", "status": "TRADING", "contractType": "TRADIFI_PERPETUAL"},
	{"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL"},
	{"symbol": "FTMUSDT", "status": "SETTLING", "contractType": "PERPETUAL"},
	{"symbol": "SOLUSDT_261225", "status": "TRADING", "contractType": "CURRENT_QUARTER"},
}

// fake serves exchangeInfo and a 5m series listed 4,000 candles before now, up to the forming candle.
func fake(nowMs int64) *binancetest.Fake {
	return &binancetest.Fake{
		Clock: binancetest.NewClock(nowMs), ExchangeInfo: exchangeInfo, ListedMs: now/step*step - 4000*step,
		Prices: map[string]string{"SOLUSDT": "150.5", "1000PEPEUSDT": "0.004"},
	}
}

func client(t *testing.T, f *binancetest.Fake, o Options) *Client {
	o.BaseURL = f.Start(t).URL
	o.Clock = f.Clock
	return New(o)
}

func limits(f *binancetest.Fake) []int {
	var out []int
	for _, p := range f.Calls("/fapi/v1/klines") {
		n, _ := strconv.Atoi(p.Get("limit"))
		out = append(out, n)
	}
	return out
}

func TestCandidates(t *testing.T) {
	if got := Candidates("B-SOL_USDT")[0]; got != (Market{"SOLUSDT", 1}) {
		t.Error(got)
	}
	if got := Candidates("B-1000PEPE_USDT")[0]; got != (Market{"1000PEPEUSDT", 1}) {
		t.Error(got)
	}
	if got := Candidates("B-PEPE_USDT")[1]; got != (Market{"1000PEPEUSDT", 1000}) {
		t.Error(got)
	}
	if got := Candidates("I-BTC_INR")[0]; got != (Market{"BTCUSDT", 1}) {
		t.Error(got)
	}
}

func TestResolveMapsCoinDCXPairsToTradingPerpetuals(t *testing.T) {
	f := fake(now)
	c := client(t, f, Options{})
	for sym, want := range map[string]Market{
		"B-SOL_USDT": {"SOLUSDT", 1}, "B-1000PEPE_USDT": {"1000PEPEUSDT", 1}, "B-PEPE_USDT": {"1000PEPEUSDT", 1000},
		"B-XAU_USDT": {"XAUUSDT", 1}, "I-BTC_INR": {"BTCUSDT", 1},
	} {
		if m, ok, err := c.Resolve(sym); err != nil || !ok || m != want {
			t.Errorf("Resolve(%s) = %v, %v, %v", sym, m, ok, err)
		}
	}
	for _, sym := range []string{"B-FTM_USDT", "B-NOPE_USDT"} { // delisted, unknown
		if _, ok, err := c.Resolve(sym); err != nil || ok {
			t.Errorf("Resolve(%s) should find nothing", sym)
		}
	}
	if n := f.Count("/fapi/v1/exchangeInfo"); n != 1 {
		t.Errorf("exchangeInfo fetched %d times", n)
	}
}

func TestCandleHistoryPagesBackFromLastClosedCandle(t *testing.T) {
	f := fake(now)
	bars, source, err := client(t, f, Options{}).CandleHistory("B-SOL_USDT", "5m", 2500, now, false, nil)
	if err != nil {
		t.Fatal(err)
	}
	if source != "binance:SOLUSDT" || bars.Len() != 2500 || f.Count("/fapi/v1/klines") != 3 {
		t.Fatalf("source %q, %d candles, %d requests", source, bars.Len(), f.Count("/fapi/v1/klines"))
	}
	for i := 1; i < bars.Len(); i++ {
		if bars.Time[i]-bars.Time[i-1] != step {
			t.Fatalf("candle %d out of order", i)
		}
	}
	if last := bars.Time[bars.Len()-1]; last != (now/step-1)*step {
		t.Errorf("last open %d", last)
	}
	if bars.TakerBuyVolume[0] != 6 || bars.Count[0] != 42 || bars.QuoteVolume[0] != 1000 || bars.FundingRate != nil {
		t.Errorf("columns: %+v", bars.At(0))
	}
}

func TestCandleHistoryStopsAtListingDate(t *testing.T) {
	f := fake(now)
	f.ListedMs = now/step*step - 300*step
	bars, _, _ := client(t, f, Options{}).CandleHistory("B-SOL_USDT", "5m", 2000, now, false, nil)
	if bars.Len() != 300 || f.Count("/fapi/v1/klines") != 1 { // every closed candle since listing, not the forming one
		t.Fatalf("%d candles, %d requests", bars.Len(), f.Count("/fapi/v1/klines"))
	}
}

func TestCachedCandlesAreNotDownloadedAgain(t *testing.T) {
	store := cache.NewCandles(filepath.Join(t.TempDir(), "candles"), 120_000)
	f := fake(now)
	first, _, _ := client(t, f, Options{Candles: store}).CandleHistory("B-SOL_USDT", "5m", 2500, now, false, nil)
	if f.Count("/fapi/v1/klines") != 3 {
		t.Fatal("first download")
	}

	// A new client (as each training run makes) reads the same candles from disk
	f = fake(now)
	again, _, _ := client(t, f, Options{Candles: store}).CandleHistory("B-SOL_USDT", "5m", 2500, now, false, nil)
	if f.Count("/fapi/v1/klines") != 0 || !reflect.DeepEqual(again, first) {
		t.Fatalf("%d requests; same candles: %v", f.Count("/fapi/v1/klines"), reflect.DeepEqual(again, first))
	}

	// Ten candles later only those ten are fetched, in one request
	f = fake(now + 10*step)
	later, _, _ := client(t, f, Options{Candles: store}).CandleHistory("B-SOL_USDT", "5m", 2500, now+10*step, false, nil)
	if got := limits(f); !reflect.DeepEqual(got, []int{10}) {
		t.Fatalf("limits %v", got)
	}
	if later.Len() != 2500 || later.Time[2499] != (now/step+9)*step {
		t.Fatalf("%d candles, last %d", later.Len(), later.Time[later.Len()-1])
	}

	// Deeper history fetches only the older stretch (the cache already holds 2,510)
	f = fake(now + 10*step)
	deeper, _, _ := client(t, f, Options{Candles: store}).CandleHistory("B-SOL_USDT", "5m", 3000, now+10*step, false, nil)
	if got := limits(f); !reflect.DeepEqual(got, []int{490}) {
		t.Fatalf("limits %v", got)
	}
	if tail := deeper.Tail(2500); deeper.Len() != 3000 || !reflect.DeepEqual(tail, later) {
		t.Fatalf("%d candles; tail matches: %v", deeper.Len(), reflect.DeepEqual(tail, later))
	}
}

func TestCacheRemembersTheListingDate(t *testing.T) {
	store := cache.NewCandles(filepath.Join(t.TempDir(), "candles"), 120_000)
	listed := now/step*step - 300*step
	f := fake(now)
	f.ListedMs = listed
	client(t, f, Options{Candles: store}).CandleHistory("B-SOL_USDT", "5m", 2000, now, false, nil)

	f = fake(now)
	f.ListedMs = listed
	bars, _, _ := client(t, f, Options{Candles: store}).CandleHistory("B-SOL_USDT", "5m", 2000, now, false, nil)
	if bars.Len() != 300 || f.Count("/fapi/v1/klines") != 0 { // nothing older exists, so nothing is asked for
		t.Fatalf("%d candles, %d requests", bars.Len(), f.Count("/fapi/v1/klines"))
	}
}

func TestInterruptedDownloadKeepsWhatItFetched(t *testing.T) {
	store := cache.NewCandles(filepath.Join(t.TempDir(), "candles"), 120_000)
	f := fake(now)
	f.Handler = func(w http.ResponseWriter, r *http.Request) bool {
		if r.URL.Path == "/fapi/v1/klines" && f.Count("/fapi/v1/klines") > 2 {
			w.WriteHeader(400)
			return true
		}
		return false
	}
	_, _, err := client(t, f, Options{Candles: store}).CandleHistory("B-SOL_USDT", "5m", 2500, now, false, nil)
	var status *StatusError
	if !errors.As(err, &status) || status.Code != 400 {
		t.Fatalf("err = %v", err)
	}

	f = fake(now)
	bars, _, _ := client(t, f, Options{Candles: store}).CandleHistory("B-SOL_USDT", "5m", 2500, now, false, nil)
	if got := limits(f); bars.Len() != 2500 || !reflect.DeepEqual(got, []int{500}) {
		t.Fatalf("%d candles, limits %v", bars.Len(), got)
	}
}

func TestDownloadProgressCountsCandles(t *testing.T) {
	var seen [][2]int
	store := cache.NewCandles(filepath.Join(t.TempDir(), "candles"), 120_000)
	client(t, fake(now), Options{Candles: store}).CandleHistory("B-SOL_USDT", "5m", 2500, now, false,
		func(done, total int) { seen = append(seen, [2]int{done, total}) })
	if want := [][2]int{{1000, 2500}, {2000, 2500}, {2500, 2500}}; !reflect.DeepEqual(seen, want) {
		t.Fatal(seen)
	}
}

func TestCandleHistoryWithoutBinanceMarket(t *testing.T) {
	f := fake(now)
	bars, source, err := client(t, f, Options{}).CandleHistory("B-NOPE_USDT", "5m", 100, 0, false, nil)
	if err != nil || bars.Len() != 0 || source != "" || f.Count("/fapi/v1/klines") != 0 {
		t.Fatalf("%d candles, %q, %v", bars.Len(), source, err)
	}
}

func TestLivePricesInPairUnits(t *testing.T) {
	prices := client(t, fake(now), Options{}).LivePrices([]string{"B-SOL_USDT", "B-1000PEPE_USDT", "B-PEPE_USDT", "B-NOPE_USDT"})
	if prices["B-SOL_USDT"] != 150.5 || prices["B-1000PEPE_USDT"] != 0.004 || math.Abs(prices["B-PEPE_USDT"]-0.000004) > 1e-12 {
		t.Fatal(prices)
	}
	if _, ok := prices["B-NOPE_USDT"]; ok {
		t.Fatal("B-NOPE_USDT has no market")
	}
}

func TestRateLimitIsRetried(t *testing.T) {
	f := fake(now)
	replies := 0
	f.Handler = func(w http.ResponseWriter, r *http.Request) bool {
		replies++
		if replies == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(429)
			return true
		}
		w.Header().Set("X-MBX-USED-WEIGHT-1M", "5")
		return false
	}
	c := client(t, f, Options{})
	if m, ok, err := c.Resolve("B-SOL_USDT"); err != nil || !ok || m != (Market{"SOLUSDT", 1}) {
		t.Fatal(m, ok, err)
	}
	if got := f.Clock.Slept(); !reflect.DeepEqual(got, []time.Duration{time.Second}) {
		t.Fatal(got)
	}
}

func TestWeightBudgetPausesUntilTheNextMinute(t *testing.T) {
	f := fake(now)
	f.Handler = func(w http.ResponseWriter, r *http.Request) bool {
		w.Header().Set("X-MBX-USED-WEIGHT-1M", "1920")
		return false
	}
	client(t, f, Options{}).Perpetuals()
	want := time.Duration((61 - float64(now%60_000)/1000) * float64(time.Second))
	if got := f.Clock.Slept(); len(got) != 1 || got[0] != want {
		t.Fatal(got, want)
	}
}

func TestPerpetualsKeepServingThroughAnOutage(t *testing.T) {
	f := fake(now)
	down := false
	f.Handler = func(w http.ResponseWriter, r *http.Request) bool {
		if down {
			w.WriteHeader(503)
		}
		return down
	}
	c := client(t, f, Options{})
	first, err := c.Perpetuals()
	if err != nil || len(first) != 4 {
		t.Fatal(first, err)
	}
	down = true
	f.Clock.Advance(2 * hour)
	if again, err := c.Perpetuals(); err != nil || len(again) != 4 {
		t.Fatal(again, err)
	}
	if _, err := client(t, f, Options{}).Perpetuals(); err == nil {
		t.Fatal("a client with no list yet should fail")
	}
	var _ catalog.PerpetualLister = c
}
