// Package board is the add-pair drawer's market board: price, 24h volume and price change over
// several windows for every Binance USD-M perpetual, keyed by CoinDCX pair.
//
//   - 24h change, 24h quote volume and the price come from one all-symbol ticker call
//     (/fapi/v1/ticker/24hr, weight 40), repeated every Poll while the board is in use.
//   - Shorter changes (ChangeWindows) are read off a price history kept per market: each market's
//     last HistoryMinutes of 1-minute closes (one klines call each, weight 2), then the ticker
//     price at every poll; markets are filled in a few at a time once the page has asked for them.
//     Change over w = price now / the latest price at or before (now - w) - 1, nil until the history
//     reaches back that far (within the tolerance in Change).
//   - The board starts on the first request and stops after Idle without one, so it costs nothing
//     while the drawer stays closed. A restart after a stop backfills again.
package board

import (
	"log/slog"
	"sort"
	"strconv"
	"sync"
	"time"

	"deltarisk/engine/binance"
	"deltarisk/engine/catalog"
)

// ChangeWindows are the short windows in minutes; 24h comes from the ticker.
var ChangeWindows = []struct {
	Name    string
	Minutes int
}{{"5m", 5}, {"15m", 15}, {"1h", 60}, {"4h", 240}}

// WindowNames are every window the board reports.
var WindowNames = []string{"5m", "15m", "1h", "4h", "24h"}

const (
	HistoryMinutes  = 245 // the longest window plus a few candles
	BackfillWorkers = 3   // klines calls in flight while filling in (weight 2 each)
)

// Client is the market data the board reads (binance.Client).
type Client interface {
	Tickers24h() ([]binance.Ticker24h, error)
	Perpetuals() (map[string]catalog.Perpetual, error)
	RecentCloses(market string, minutes int) ([]binance.TimedPrice, error)
	Resolve(symbol string) (binance.Market, bool, error)
}

type ticker struct{ price, change24h, volume24h float64 }

type points struct {
	times  []int64
	prices []float64
}

// Board is safe for concurrent use.
type Board struct {
	NewClient func() Client
	Poll      time.Duration
	Idle      time.Duration
	Clock     func() time.Time

	mu          sync.Mutex
	running     bool
	stop        chan struct{}
	ready       chan struct{} // closed once the first ticker poll is in
	wake        chan struct{} // the page asked about markets not filled in yet
	lastRequest time.Time
	tickers     map[string]ticker // Binance market -> latest
	history     map[string]*points
	backfilled  map[string]bool
	wanted      map[string]bool // markets behind the pairs the page asks about: only these are backfilled
	updatedAt   *int64
	client      Client
}

// New makes a board that polls every 30 s and stops after 10 idle minutes.
func New(newClient func() Client) *Board {
	return &Board{NewClient: newClient, Poll: 30 * time.Second, Idle: 10 * time.Minute, Clock: time.Now,
		ready: make(chan struct{}), wake: make(chan struct{}, 1)}
}

func (b *Board) reset() {
	b.tickers, b.history, b.backfilled, b.wanted, b.updatedAt = map[string]ticker{}, map[string]*points{}, map[string]bool{}, map[string]bool{}, nil
}

// Touch marks the board as in use, starting it if it isn't running.
func (b *Board) Touch() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.lastRequest = b.Clock()
	if b.running {
		return
	}
	b.running = true
	b.stop = make(chan struct{})
	b.ready = make(chan struct{})
	b.reset()
	b.client = b.NewClient()
	go b.run(b.stop, b.client)
}

// WaitReady waits up to timeout for the first poll after a start (so the first request has prices).
func (b *Board) WaitReady(timeout time.Duration) bool {
	b.mu.Lock()
	ready := b.ready
	b.mu.Unlock()
	select {
	case <-ready:
		return true
	case <-time.After(timeout):
		return false
	}
}

// Stop ends the board's loop.
func (b *Board) Stop() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.running {
		close(b.stop)
		b.running = false
	}
}

func (b *Board) idle() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.Clock().Sub(b.lastRequest) > b.Idle
}

func (b *Board) run(stop chan struct{}, client Client) {
	defer func() {
		b.mu.Lock()
		if b.stop == stop {
			b.running = false
		}
		b.mu.Unlock()
		slog.Info("market board stopped")
	}()
	for {
		select {
		case <-stop:
			return
		default:
		}
		if b.idle() {
			return
		}
		started := b.Clock()
		if err := b.PollOnce(client); err != nil { // a failed poll is retried at the next one
			slog.Warn("market board refresh failed", "err", err)
		} else {
			b.Backfill(client, started.Add(b.Poll), stop)
		}
		wait := max(time.Second, b.Poll-b.Clock().Sub(started))
		select {
		case <-stop:
			return
		case <-b.wake:
		case <-time.After(wait):
		}
	}
}

// PollOnce reads the all-symbol 24h ticker: prices, 24h change and volume, and a point in each history.
func (b *Board) PollOnce(client Client) error {
	rows, err := client.Tickers24h()
	if err != nil {
		return err
	}
	perps, err := client.Perpetuals()
	if err != nil {
		return err
	}
	now := b.Clock().UnixMilli()
	b.mu.Lock()
	for _, t := range rows {
		if _, ok := perps[t.Symbol]; !ok {
			continue
		}
		price, e1 := strconv.ParseFloat(t.LastPrice, 64)
		change, e2 := strconv.ParseFloat(t.PriceChangePercent, 64)
		volume, e3 := strconv.ParseFloat(t.QuoteVolume, 64)
		if e1 != nil || e2 != nil || e3 != nil {
			continue
		}
		b.tickers[t.Symbol] = ticker{price, change, volume}
		h := b.history[t.Symbol]
		if h == nil {
			h = &points{}
			b.history[t.Symbol] = h
		}
		h.times, h.prices = append(h.times, now), append(h.prices, price)
		trim(h, now)
	}
	b.updatedAt = &now
	ready := b.ready
	b.mu.Unlock()
	select {
	case <-ready:
	default:
		close(ready)
	}
	return nil
}

// Backfill seeds the history of the markets the page shows with their last 1-minute closes, a few
// at a time, until deadline (so the ticker keeps polling while hundreds of markets fill in).
func (b *Board) Backfill(client Client, deadline time.Time, stop <-chan struct{}) {
	b.mu.Lock()
	var todo []string
	for m := range b.tickers {
		if b.wanted[m] && !b.backfilled[m] {
			todo = append(todo, m)
		}
	}
	b.mu.Unlock()
	sort.Strings(todo)
	jobs := make(chan string)
	var wg sync.WaitGroup
	for range BackfillWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for m := range jobs {
				b.fill(client, m, deadline, stop)
			}
		}()
	}
	for _, m := range todo {
		jobs <- m
	}
	close(jobs)
	wg.Wait()
}

func (b *Board) fill(client Client, market string, deadline time.Time, stop <-chan struct{}) {
	if stop != nil {
		select {
		case <-stop:
			return
		default:
		}
	}
	if !deadline.IsZero() && !b.Clock().Before(deadline) {
		return
	}
	closes, err := client.RecentCloses(market, HistoryMinutes)
	if err != nil {
		slog.Debug("market board backfill failed", "market", market, "err", err)
		closes = nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.backfilled[market] = true
	h := b.history[market]
	if h == nil {
		h = &points{}
		b.history[market] = h
	}
	merged := map[int64]float64{}
	for _, c := range closes {
		merged[c.AtMs] = c.Close
	}
	for i, t := range h.times { // the ticker's own points win
		merged[t] = h.prices[i]
	}
	h.times, h.prices = h.times[:0], h.prices[:0]
	for t := range merged {
		h.times = append(h.times, t)
	}
	sort.Slice(h.times, func(i, j int) bool { return h.times[i] < h.times[j] })
	for _, t := range h.times {
		h.prices = append(h.prices, merged[t])
	}
}

func trim(h *points, now int64) {
	cut := sort.Search(len(h.times), func(i int) bool { return h.times[i] >= now-(HistoryMinutes+5)*60_000 })
	if cut > 0 {
		h.times, h.prices = h.times[cut:], h.prices[cut:]
	}
}

// Change is the % change from the latest price at or before (now - minutes) to price; nil when
// the history doesn't reach back that far (within max(90 s, a tenth of the window)).
func Change(times []int64, prices []float64, now int64, minutes int, price float64) *float64 {
	target := now - int64(minutes)*60_000
	i := sort.Search(len(times), func(k int) bool { return times[k] > target }) - 1
	if i < 0 || target-times[i] > max(90_000, int64(minutes)*6_000) || prices[i] <= 0 {
		return nil
	}
	c := 100.0 * (price/prices[i] - 1.0)
	return &c
}

// Coin is one pair on the board.
type Coin struct {
	Price     float64             `json:"price"`      // in the pair's units
	Volume24h float64             `json:"volume_24h"` // USDT
	Change    map[string]*float64 `json:"change"`     // %
}

// View is GET /api/market.
type View struct {
	Windows    []string        `json:"windows"`
	UpdatedAt  *int64          `json:"updated_at"` // ms
	Markets    int             `json:"markets"`
	Backfilled int             `json:"backfilled"`
	Coins      map[string]Coin `json:"coins"`
	Offline    bool            `json:"offline,omitempty"`
}

// Offline is the board without market data.
func Offline() View {
	return View{Windows: WindowNames, Coins: map[string]Coin{}, Offline: true}
}

// View is, per CoinDCX pair (the catalog's symbols), the price in the pair's units, 24h quote
// volume in USDT and the change over every window in %.
func (b *Board) View(symbols []string) View {
	b.mu.Lock()
	client := b.client
	tickers := make(map[string]ticker, len(b.tickers))
	for k, v := range b.tickers {
		tickers[k] = v
	}
	hist := make(map[string]points, len(b.history))
	for k, v := range b.history {
		hist[k] = points{append([]int64(nil), v.times...), append([]float64(nil), v.prices...)}
	}
	updatedAt := b.updatedAt
	backfilled := make(map[string]bool, len(b.backfilled))
	for k := range b.backfilled {
		backfilled[k] = true
	}
	b.mu.Unlock()

	coins := map[string]Coin{}
	wanted := map[string]bool{}
	if client != nil && len(tickers) > 0 {
		for _, sym := range symbols {
			market, ok, err := client.Resolve(sym)
			if err != nil || !ok {
				continue
			}
			t, ok := tickers[market.Symbol]
			if !ok {
				continue
			}
			wanted[market.Symbol] = true
			h := hist[market.Symbol]
			var now int64
			if len(h.times) > 0 {
				now = h.times[len(h.times)-1]
			}
			change := map[string]*float64{}
			for _, w := range ChangeWindows {
				change[w.Name] = Change(h.times, h.prices, now, w.Minutes, t.price)
			}
			c24 := t.change24h
			change["24h"] = &c24
			coins[sym] = Coin{Price: t.price / float64(market.Multiplier), Volume24h: t.volume24h, Change: change}
		}
	}
	filled := 0
	for m := range wanted {
		if backfilled[m] {
			filled++
		}
	}
	b.mu.Lock()
	added := false
	for m := range wanted {
		if !b.wanted[m] {
			b.wanted[m] = true
			added = true
		}
	}
	b.mu.Unlock()
	if added {
		select {
		case b.wake <- struct{}{}:
		default:
		}
	}
	return View{Windows: WindowNames, UpdatedAt: updatedAt, Markets: len(wanted), Backfilled: filled, Coins: coins}
}

// SetClient sets the client without starting the loop (tests).
func (b *Board) SetClient(c Client) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.client = c
	if b.tickers == nil {
		b.reset()
	}
}
