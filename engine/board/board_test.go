package board

import (
	"math"
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"

	"deltarisk/engine/binance"
	"deltarisk/engine/catalog"
)

const (
	minute = int64(60_000)
	now    = int64(1_800_000_000_000) // a whole minute
)

// fake has two perpetuals: SOL rose from 100 to 200 over four hours, 1000PEPE stayed flat.
type fake struct {
	mu    sync.Mutex
	calls []string
}

func (f *fake) Perpetuals() (map[string]catalog.Perpetual, error) {
	return map[string]catalog.Perpetual{"SOLUSDT": {}, "1000PEPEUSDT": {}}, nil
}

func (f *fake) Resolve(symbol string) (binance.Market, bool, error) {
	m, ok := map[string]binance.Market{"B-SOL_USDT": {Symbol: "SOLUSDT", Multiplier: 1},
		"B-PEPE_USDT": {Symbol: "1000PEPEUSDT", Multiplier: 1000}, "B-1000PEPE_USDT": {Symbol: "1000PEPEUSDT", Multiplier: 1}}[symbol]
	return m, ok, nil
}

func (f *fake) Tickers24h() ([]binance.Ticker24h, error) {
	return []binance.Ticker24h{
		{Symbol: "SOLUSDT", LastPrice: "200", PriceChangePercent: "5.5", QuoteVolume: "7.1e8"},
		{Symbol: "1000PEPEUSDT", LastPrice: "0.012", PriceChangePercent: "-2", QuoteVolume: "3e8"},
		{Symbol: "DELISTEDUSDT", LastPrice: "1", PriceChangePercent: "0", QuoteVolume: "1"},
	}, nil
}

func (f *fake) RecentCloses(market string, minutes int) ([]binance.TimedPrice, error) {
	f.mu.Lock()
	f.calls = append(f.calls, market)
	f.mu.Unlock()
	var out []binance.TimedPrice
	for k := minutes; k > 0; k-- {
		p := 0.012
		if market == "SOLUSDT" {
			p = 200 - 100*float64(k)/240
		}
		out = append(out, binance.TimedPrice{AtMs: now - int64(k)*minute, Close: p})
	}
	return out, nil
}

func newBoard() (*Board, *fake) {
	f := &fake{}
	b := New(func() Client { return f })
	b.Clock = func() time.Time { return time.UnixMilli(now) }
	b.SetClient(f)
	return b, f
}

func approx(p *float64, want float64) bool { return p != nil && math.Abs(*p-want) < 1e-9 }

func TestChangesOverEveryWindowFromBackfillAndTicker(t *testing.T) {
	b, f := newBoard()
	if err := b.PollOnce(f); err != nil {
		t.Fatal(err)
	}
	b.Backfill(f, time.Time{}, nil)
	if len(f.calls) != 0 { // nothing asked for yet
		t.Fatal(f.calls)
	}
	b.View([]string{"B-SOL_USDT", "B-PEPE_USDT"})
	b.Backfill(f, time.Time{}, nil)
	sort.Strings(f.calls)
	if !reflect.DeepEqual(f.calls, []string{"1000PEPEUSDT", "SOLUSDT"}) { // one klines call per market the page shows
		t.Fatal(f.calls)
	}
	v := b.View([]string{"B-SOL_USDT", "B-PEPE_USDT", "B-1000PEPE_USDT", "B-NOPE_USDT"})
	if v.Markets != 2 || v.Backfilled != 2 || v.UpdatedAt == nil || *v.UpdatedAt != now {
		t.Fatalf("%+v", v)
	}
	sol := v.Coins["B-SOL_USDT"]
	if sol.Price != 200 || sol.Volume24h != 7.1e8 || !approx(sol.Change["24h"], 5.5) || !approx(sol.Change["4h"], 100) ||
		!approx(sol.Change["1h"], 100*(200/(200-100*60.0/240)-1)) || !approx(sol.Change["5m"], 100*(200/(200-100*5.0/240)-1)) {
		t.Fatalf("%+v", sol)
	}
	pepe := v.Coins["B-PEPE_USDT"]
	if math.Abs(pepe.Price-0.012/1000) > 1e-15 || !approx(pepe.Change["15m"], 0) {
		t.Fatalf("%+v", pepe)
	}
	if v.Coins["B-1000PEPE_USDT"].Price != 0.012 { // the 1000x listing in its own units
		t.Fatal(v.Coins["B-1000PEPE_USDT"])
	}
	if _, ok := v.Coins["B-NOPE_USDT"]; ok {
		t.Fatal("unknown pair")
	}
}

func TestWindowsTheHistoryDoesNotReachAreNil(t *testing.T) {
	b, f := newBoard()
	b.PollOnce(f) // ticker only: one point, no backfill yet
	change := b.View([]string{"B-SOL_USDT"}).Coins["B-SOL_USDT"].Change
	for _, w := range []string{"5m", "15m", "1h", "4h"} {
		if change[w] != nil {
			t.Fatal(w, *change[w])
		}
	}
	if !approx(change["24h"], 5.5) {
		t.Fatal(change["24h"])
	}
}

func TestChangeNeedsAPointCloseToTheWindowStart(t *testing.T) {
	times, prices := []int64{now - 20*minute, now - 4*minute}, []float64{100, 110}
	if Change(times, prices, now, 5, 121) != nil { // nothing near 5 minutes ago
		t.Fatal("5m")
	}
	if !approx(Change(times, prices, now, 4, 121), 10) {
		t.Fatal("4m")
	}
	if Change(times, prices, now, 21, 121) != nil { // before the history starts
		t.Fatal("21m")
	}
	if HistoryMinutes < 240 {
		t.Fatal(HistoryMinutes)
	}
}

func TestTheBoardStartsOnTouchAndStops(t *testing.T) {
	f := &fake{}
	b := New(func() Client { return f })
	b.Poll = 50 * time.Millisecond
	b.Touch()
	if !b.WaitReady(2 * time.Second) {
		t.Fatal("first poll")
	}
	v := b.View([]string{"B-SOL_USDT"})
	if v.Coins["B-SOL_USDT"].Price != 200 {
		t.Fatalf("%+v", v)
	}
	deadline := time.Now().Add(2 * time.Second) // the view woke it to backfill SOL
	for {
		if b.View([]string{"B-SOL_USDT"}).Backfilled == 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("backfill")
		}
		time.Sleep(10 * time.Millisecond)
	}
	b.Stop()
	b.Idle = 0
	b.Touch() // restarts; with no idle time it stops after one round
	b.WaitReady(2 * time.Second)
	b.Stop()
}
