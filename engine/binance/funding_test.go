package binance

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
	"time"

	"deltarisk/engine/binance/binancetest"
	"deltarisk/engine/cache"
)

const (
	settle = int64(1_700_006_400_000) // a real 8h BTCUSDT funding settlement (16:00 UTC)
	symbol = "B-SOL_USDT"
)

var fundingNow = settle + 30_000 // the candles closing at the settlement have just closed

func phase(market string) int {
	s := 0
	for _, r := range market {
		s += int(r)
	}
	return s % 17
}

// rate is the funding settled at t: different every settlement and per coin, never zero; 8 decimals like Binance.
func rate(market string, t int64) float64 {
	v := 1e-4*math.Sin(float64(t)/float64(8*hour)*0.7+float64(phase(market))) + 2e-5
	r, _ := strconv.ParseFloat(strconv.FormatFloat(v, 'f', 8, 64), 64)
	return r
}

func fundingFake(fundingHours int64) *binancetest.Fake {
	return &binancetest.Fake{
		Clock: binancetest.NewClock(fundingNow), Markets: []string{"SOLUSDT", "BTCUSDT", "ETHUSDT"},
		ListedMs: fundingNow - 120*day, FundingStepMs: fundingHours * hour, Rate: rate,
	}
}

func near(a, b float64) bool { return math.Abs(a-b) <= 1e-6*math.Abs(b)+1e-12 }

func startTimes(f *binancetest.Fake) []int64 {
	var out []int64
	for _, p := range f.Calls("/fapi/v1/fundingRate") {
		v, _ := strconv.ParseInt(p.Get("startTime"), 10, 64)
		out = append(out, v)
	}
	return out
}

func TestFundingRatesPageForwardAndReuseDownloads(t *testing.T) {
	f := fundingFake(1)
	c := client(t, f, Options{})
	start := fundingNow - 50*day

	events, err := c.FundingRates(symbol, start, 0)
	if err != nil {
		t.Fatal(err)
	}
	first := (start/day*day + hour - 1) / hour * hour
	var expected []int64
	for ts := first; ts <= fundingNow; ts += hour {
		expected = append(expected, ts)
	}
	if len(expected) <= 1000 || f.Count("/fapi/v1/fundingRate") != 2 { // 1000-row pages walking forward
		t.Fatalf("%d settlements, %d requests", len(expected), f.Count("/fapi/v1/fundingRate"))
	}
	if !reflect.DeepEqual(events.Time, expected) || !near(events.Rate[len(expected)-1], rate("SOLUSDT", expected[len(expected)-1])) {
		t.Fatal("settlements differ")
	}

	// Nothing can have settled since the download, so a request up to now is served from it
	recent, _ := c.FundingRates(symbol, fundingNow-day, fundingNow) // start rounds down to the UTC day
	want := 0
	for _, ts := range expected {
		if ts >= (fundingNow-day)/day*day {
			want++
		}
	}
	if recent.Len() != want || f.Count("/fapi/v1/fundingRate") != 2 {
		t.Fatalf("%d recent (want %d), %d requests", recent.Len(), want, f.Count("/fapi/v1/fundingRate"))
	}

	// An hour later only the settlements after the newest one held are fetched
	f.Clock.Advance(hour)
	events, _ = c.FundingRates(symbol, start, 0)
	starts := startTimes(f)
	if len(starts) != 3 || starts[2] != expected[len(expected)-1]+1 || events.Time[events.Len()-1] != expected[len(expected)-1]+hour {
		t.Fatalf("starts %v", starts)
	}
}

func TestFundingDownloadsSurviveARestart(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "funding")
	f := fundingFake(1)
	start := fundingNow - 50*day
	first, _ := client(t, f, Options{Funding: cache.NewFunding(dir)}).FundingRates(symbol, start, 0)
	if _, err := os.Stat(filepath.Join(dir, "SOLUSDT.parquet")); err != nil || f.Count("/fapi/v1/fundingRate") != 2 {
		t.Fatal("first download should be on disk", err)
	}

	// A new client (a restarted app) at the same moment reads the file and downloads nothing
	again, _ := client(t, f, Options{Funding: cache.NewFunding(dir)}).FundingRates(symbol, start, 0)
	if !reflect.DeepEqual(again, first) || f.Count("/fapi/v1/fundingRate") != 2 {
		t.Fatal("restart should read the file")
	}

	// An hour later it fetches only the settlement since
	f.Clock.Advance(hour)
	later, _ := client(t, f, Options{Funding: cache.NewFunding(dir)}).FundingRates(symbol, start, 0)
	starts := startTimes(f)
	if len(starts) != 3 || later.Len() != first.Len()+1 || starts[2] != first.Time[first.Len()-1]+1 {
		t.Fatalf("starts %v, %d events", starts, later.Len())
	}

	// A request reaching further back than the file covers downloads again from there
	client(t, f, Options{Funding: cache.NewFunding(dir)}).FundingRates(symbol, start-30*day, 0)
	if starts := startTimes(f); starts[3] != (start-30*day)/day*day {
		t.Fatalf("starts %v", starts)
	}
}

func TestFundingRequestsStayWithinTheirOwnBudget(t *testing.T) {
	f := fundingFake(8)
	c := client(t, f, Options{FundingBudget: 2})
	for range 3 {
		c.fundingSlot()
	}
	// the third request waits for the first to leave the 5-minute window
	if got := f.Clock.Slept(); !reflect.DeepEqual(got, []time.Duration{5 * time.Minute}) {
		t.Fatal(got)
	}
}

func TestCandlesCarryTheRateSettledBeforeTheyClosed(t *testing.T) {
	f := fundingFake(8)
	bars, _, err := client(t, f, Options{}).CandleHistory(symbol, "5m", 400, 0, true, nil)
	if err != nil {
		t.Fatal(err)
	}
	at := -1
	for i, open := range bars.Time {
		close := open + step
		settled := (close - 1) / (8 * hour) * (8 * hour)
		if !near(bars.FundingRate[i], rate("SOLUSDT", settled)) {
			t.Fatalf("candle %d: %v, want %v", i, bars.FundingRate[i], rate("SOLUSDT", settled))
		}
		if close == settle-8*hour {
			at = i
		}
	}
	// The candle closing at a settlement still shows the previous rate; the next one the new rate
	if !near(bars.FundingRate[at], rate("SOLUSDT", settle-16*hour)) || !near(bars.FundingRate[at+1], rate("SOLUSDT", settle-8*hour)) {
		t.Fatal("rates around the settlement")
	}
	if last := bars.Time[bars.Len()-1] + step; last != settle {
		t.Fatalf("last close %d", last)
	}
}
