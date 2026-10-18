package manager

import (
	"errors"
	"math"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
	"time"

	"deltarisk/internal/binance"
	"deltarisk/internal/binance/binancetest"
	"deltarisk/internal/config"
	"deltarisk/internal/history"
	"deltarisk/internal/paths"
	"deltarisk/internal/regime"
)

func tempHome(t *testing.T) {
	t.Helper()
	tmp := t.TempDir()
	t.Setenv("DELTARISK_HOME", filepath.Join(tmp, "home"))
	t.Setenv("FUTURECAST_HOME", "")
	oldDefault, oldLegacy := paths.DefaultHome, paths.LegacyHome
	paths.DefaultHome, paths.LegacyHome = filepath.Join(tmp, "default"), filepath.Join(tmp, "legacy")
	t.Cleanup(func() { paths.DefaultHome, paths.LegacyHome = oldDefault, oldLegacy })
}

// swapSteps replaces the download and training steps for one test.
func swapSteps(t *testing.T, download func(*Worker, *binance.Client) error, prepare func(*Worker)) {
	oldD, oldP := Download, Prepare
	Download, Prepare = download, prepare
	t.Cleanup(func() { Download, Prepare = oldD, oldP })
}

func TestNextCoinDownloadsWhileThePreviousOneTrains(t *testing.T) {
	tempHome(t)
	var mu sync.Mutex
	at := map[string]time.Time{}
	log := func(what string, w *Worker) {
		mu.Lock()
		at[what+" "+w.Symbol] = time.Now()
		mu.Unlock()
	}
	swapSteps(t, func(w *Worker, _ *binance.Client) error {
		log("download start", w)
		time.Sleep(50 * time.Millisecond)
		w.setStatus("training")
		log("download end", w)
		return nil
	}, func(w *Worker) {
		log("train start", w)
		time.Sleep(200 * time.Millisecond)
		w.setStatus("ready")
		log("train end", w)
	})
	m := New(config.Default(), "", true)
	coins := []string{"B-AAA_USDT", "B-BBB_USDT", "B-CCC_USDT"}
	for _, s := range coins {
		m.Add(s, false)
	}
	var wg sync.WaitGroup
	for i, s := range coins {
		wg.Add(1)
		go func() {
			defer wg.Done()
			time.Sleep(time.Duration(i) * 5 * time.Millisecond) // queue in order
			m.LoadHistory(s)
		}()
	}
	wg.Wait()
	for _, s := range coins {
		if st := m.Worker(s).Status(); st != "ready" {
			t.Fatal(s, st)
		}
	}
	// B downloads while A trains, and C while B trains
	if !at["download start B-BBB_USDT"].Before(at["train end B-AAA_USDT"]) || !at["download start B-CCC_USDT"].Before(at["train end B-BBB_USDT"]) {
		t.Fatal("downloads should overlap training")
	}
	// Still one at a time within each queue
	for _, q := range []string{"download", "train"} {
		for i := 1; i < len(coins); i++ {
			if at[q+" start "+coins[i]].Before(at[q+" end "+coins[i-1]]) {
				t.Fatalf("%s queue ran two coins at once", q)
			}
		}
	}
}

func TestAFailedDownloadSkipsTraining(t *testing.T) {
	tempHome(t)
	trained := 0
	swapSteps(t, func(w *Worker, _ *binance.Client) error {
		w.setStatus("failed")
		return errors.New("no market")
	}, func(w *Worker) { trained++ })
	m := New(config.Default(), "", true)
	m.Add("B-NOPE_USDT", false)
	m.LoadHistory("B-NOPE_USDT")
	if trained != 0 || m.Worker("B-NOPE_USDT").Status() != "failed" {
		t.Fatal(trained)
	}
}

func TestScheduleLoadRunsOnceAtATime(t *testing.T) {
	tempHome(t)
	var mu sync.Mutex
	downloads := 0
	release := make(chan struct{})
	swapSteps(t, func(w *Worker, _ *binance.Client) error {
		mu.Lock()
		downloads++
		mu.Unlock()
		<-release
		return nil
	}, func(w *Worker) { w.setStatus("ready") })
	m := New(config.Default(), "", true)
	m.Add("B-AAA_USDT", false)
	m.ScheduleLoad("B-AAA_USDT")
	time.Sleep(20 * time.Millisecond)
	m.ScheduleLoad("b-aaa_usdt") // already queued: ignored
	close(release)
	m.Wait()
	if downloads != 1 || m.Worker("B-AAA_USDT").Status() != "ready" {
		t.Fatal(downloads)
	}
}

// ───────── a worker against a fake Binance ─────────

const (
	minute = int64(60_000)
	hour   = int64(3_600_000)
	day    = int64(86_400_000)
	settle = int64(1_700_006_400_000) // a real 8h funding settlement (16:00 UTC)
	symbol = "B-SOL_USDT"
)

func phase(market string) int {
	s := 0
	for _, r := range market {
		s += int(r)
	}
	return s % 17
}

func price(market string, t int64) float64 {
	x, p := float64(t)/float64(5*minute), float64(phase(market))
	return 100 * math.Exp(0.02*math.Sin(x/37+p)+0.01*math.Sin(x/11+2*p)+0.004*math.Sin(x/3.1+p))
}

func rate(market string, t int64) float64 {
	v := 1e-4*math.Sin(float64(t)/float64(8*hour)*0.7+float64(phase(market))) + 2e-5
	r, _ := strconv.ParseFloat(strconv.FormatFloat(v, 'f', 8, 64), 64)
	return r
}

func fakeBinance(t *testing.T) (*binancetest.Fake, *binance.Client) {
	now := settle + 30_000 // the candles closing at the settlement have just closed
	f := &binancetest.Fake{Clock: binancetest.NewClock(now), Markets: []string{"SOLUSDT", "BTCUSDT", "ETHUSDT"},
		ListedMs: now - 120*day, FundingStepMs: 8 * hour, Rate: rate,
		Kline: func(market string, open, step int64) []any {
			o, c := price(market, open), price(market, open+step)
			s := func(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }
			return []any{open, s(o), s(math.Max(o, c) * 1.001), s(math.Min(o, c) * 0.999), s(c), "1000", open + step - 1,
				s(1000 * c), 50, "500", s(500 * c), "0"}
		}}
	srv := f.Start(t)
	return f, binance.New(binance.Options{BaseURL: srv.URL, Clock: f.Clock})
}

func near(a, b float64) bool { return math.Abs(a-b) <= 1e-9*math.Abs(b)+1e-15 }

func TestHistoryAndLiveCandlesCarryTheSettledFundingRate(t *testing.T) {
	tempHome(t)
	f, client := fakeBinance(t)
	w := NewWorker(symbol, config.Default())
	if err := w.DownloadHistory(client); err != nil {
		t.Fatal(err)
	}
	if w.Status() != "training" || w.Snapshot().Progress == nil {
		t.Fatal(w.Status())
	}
	w.PrepareModels()
	snap := w.Snapshot()
	if snap.Status != "ready" || snap.Progress != nil || len(snap.Risk.Windows) == 0 {
		t.Fatalf("%s %+v, %d windows", snap.Status, snap.Progress, len(snap.Risk.Windows))
	}
	if _, ok := snap.Risk.Windows["8h"]; !ok {
		t.Fatal("8h trained after the download")
	}
	hourly := w.Chart("1h")
	if !hourly.Has(symbol, "funding_rate") || hourly.Has(w.Benchmark, "funding_rate") || w.Chart("15m").Has(symbol, "funding_rate") {
		t.Fatal("only the coin's funding, only on the 1h chart")
	}
	// The candle closing at the settlement still carries the previous rate
	if got := w.Funding().RatePct; !near(got, 100*rate("SOLUSDT", settle-8*hour)) {
		t.Fatal(got)
	}
	if fp := snap.Risk.Windows["1h"].Long.Reason; fp == "" {
		t.Fatal("reason")
	}

	// An hour later every chart gains a candle; the new hourly one carries the new rate, fetched incrementally
	before := f.Count("/fapi/v1/fundingRate")
	f.Clock.Advance(hour)
	updated := w.RefreshCandles(client, f.Clock.Now())
	if len(updated) != 3 {
		t.Fatal(updated)
	}
	if got := w.Funding().RatePct; !near(got, 100*rate("SOLUSDT", settle)) {
		t.Fatal(got)
	}
	calls := f.Calls("/fapi/v1/fundingRate")[before:]
	if start, _ := strconv.ParseInt(calls[0].Get("startTime"), 10, 64); len(calls) != 1 || start <= settle-8*hour {
		t.Fatal(len(calls), start)
	}
	if got := w.Chart("1h").OpenTime[w.Chart("1h").Len()-1]; got != settle {
		t.Fatal("the hour that closed at settle + 1h", got)
	}
	if w.Chart("5m").Len() > history.Bars["5m"] {
		t.Fatal("keeps at most history.Bars")
	}
	// Nothing new closed: nothing fetched
	if again := w.RefreshCandles(client, f.Clock.Now()); len(again) != 0 {
		t.Fatal(again)
	}
}

func TestAMissingMarketFailsTheDownload(t *testing.T) {
	tempHome(t)
	_, client := fakeBinance(t)
	w := NewWorker("B-NOPE_USDT", config.Default())
	if err := w.DownloadHistory(client); err == nil {
		t.Fatal("expected an error")
	}
	s := w.Snapshot()
	if s.Status != "failed" || s.Error == nil || s.Progress != nil {
		t.Fatalf("%+v", s)
	}
}

// ───────── offline ─────────

func TestOfflineCoinsNeedNoFundingAndHaveEveryChart(t *testing.T) {
	w := NewWorker("B-TEST_USDT", config.Default())
	w.LoadSynthetic(7)
	if w.Funding() != nil {
		t.Fatal("synthetic candles carry no funding")
	}
	for _, win := range regime.Windows {
		if c := w.Chart(win.Interval); c.Empty() {
			t.Fatal(win.Interval)
		}
	}
	s := w.Snapshot()
	if s.Risk.Windows["4h"].Long.Score == nil || s.Status != "ready" || s.CurrentPrice == nil || s.History["5m"].Candles != OfflineBars {
		t.Fatalf("%+v", s.History)
	}
}

func TestBTCRiskBuildsAgainstETH(t *testing.T) {
	w := NewWorker("B-BTC_USDT", config.Default())
	w.LoadSynthetic(1)
	// BTC benchmarked against itself duplicated its candle columns
	if w.Benchmark != "B-ETH_USDT" || !w.Chart("1h").Has("B-ETH_USDT", "close") || !w.Chart("1h").Has("B-BTC_USDT", "close") {
		t.Fatal(w.Benchmark)
	}
	if len(w.Chart("1h").Columns) != len(w.Chart("1h").Names) {
		t.Fatal("duplicate columns")
	}
}

func TestModelsRetrainAfterADay(t *testing.T) {
	w := NewWorker("B-TEST_USDT", config.Default())
	if w.NeedsTraining(time.Now()) { // nothing loaded yet
		t.Fatal("pending coins don't train")
	}
	w.LoadSynthetic(7)
	if w.NeedsTraining(time.Now()) || !w.NeedsTraining(w.TrainedAt().Add(25*time.Hour)) {
		t.Fatal("retrain after a day")
	}
}

func TestOverridesAreChecked(t *testing.T) {
	w := NewWorker("B-SOL_USDT", config.Default())
	w.LoadSynthetic(3)
	moon, lev := "moon", 500.0
	if w.SetOverrides(&moon, nil) == nil || w.SetOverrides(nil, &lev) == nil {
		t.Fatal("bad overrides accepted")
	}
	alt, five := "altcoin", 5.0
	if err := w.SetOverrides(&alt, &five); err != nil {
		t.Fatal(err)
	}
	s := w.Snapshot()
	if s.Tier != "altcoin" || s.TierAuto != "megacap" || s.Leverage != 5 || s.Risk.MMR != 0.005 {
		t.Fatalf("%+v", s)
	}
}

func TestRetrainingRunsOnTheTrainingQueue(t *testing.T) {
	tempHome(t)
	m := New(config.Default(), "", false)
	w := m.Add("B-TEST_USDT", false)
	first := w.TrainedAt()
	w.mu.Lock()
	w.risk.TrainedAt = first.Add(-25 * time.Hour) // a day old
	w.mu.Unlock()
	m.ScheduleTraining("B-TEST_USDT")
	m.Wait()
	if !w.TrainedAt().After(first.Add(-time.Hour)) || w.NeedsTraining(time.Now()) {
		t.Fatal("retrained", w.TrainedAt())
	}
}
