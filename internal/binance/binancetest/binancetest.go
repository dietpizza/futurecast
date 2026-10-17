// Package binancetest fakes the Binance endpoints deltarisk calls, on an httptest server, with
// a clock the client and the fake share. Tests never reach the network.
package binancetest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"sync"
	"testing"
	"time"
)

// Clock is a settable clock that records sleeps instead of sleeping.
type Clock struct {
	mu    sync.Mutex
	ms    int64
	slept []time.Duration
}

func NewClock(ms int64) *Clock { return &Clock{ms: ms} }

func (c *Clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return time.UnixMilli(c.ms)
}

func (c *Clock) Sleep(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.slept = append(c.slept, d)
}

// Ms is the time in milliseconds.
func (c *Clock) Ms() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.ms
}

// Advance moves the clock on.
func (c *Clock) Advance(ms int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ms += ms
}

// Slept is every sleep asked for.
func (c *Clock) Slept() []time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]time.Duration(nil), c.slept...)
}

// Fake serves exchangeInfo (Markets, TRADING perpetuals, or ExchangeInfo verbatim), klines from
// the listing up to the forming candle at the clock's time, funding settled every FundingStepMs,
// and last prices. Handler, when set, answers first; returning false falls through.
type Fake struct {
	Clock         *Clock
	Markets       []string
	ExchangeInfo  []map[string]any
	ListedMs      int64                                       // first candle / settlement (rounded up to the step)
	Kline         func(market string, open, step int64) []any // a kline row; default Row
	FundingStepMs int64                                       // default 8h
	Rate          func(market string, t int64) float64        // default 0.0001
	Prices        map[string]string
	Tickers24h    []map[string]any
	Handler       func(w http.ResponseWriter, r *http.Request) bool

	mu    sync.Mutex
	calls []Call
}

// Call is one request the fake answered.
type Call struct {
	Path   string
	Params url.Values
}

// Start serves the fake until the test ends; the URL is the client's BaseURL.
func (f *Fake) Start(t testing.TB) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(srv.Close)
	return srv
}

// Calls are the requests to path, in order.
func (f *Fake) Calls(path string) []url.Values {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []url.Values
	for _, c := range f.calls {
		if c.Path == path {
			out = append(out, c.Params)
		}
	}
	return out
}

// Count is the number of requests to path.
func (f *Fake) Count(path string) int { return len(f.Calls(path)) }

// Row is a plain kline row: price 100 + (open / step mod 7), as Binance formats it.
func Row(market string, open, step int64) []any {
	px := 100.0 + float64((open/step)%7)
	s := func(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }
	return []any{open, s(px), s(px + 1), s(px - 1), s(px + 0.5), "10", open + step - 1, "1000", 42, "6", "600", "0"}
}

func ceilTo(v, step int64) int64 {
	q := v / step
	if v%step != 0 && v > 0 {
		q++
	}
	return q * step
}

func (f *Fake) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.calls = append(f.calls, Call{r.URL.Path, r.URL.Query()})
	f.mu.Unlock()
	if f.Handler != nil && f.Handler(w, r) {
		return
	}
	q, now := r.URL.Query(), f.Clock.Ms()
	switch r.URL.Path {
	case "/fapi/v1/exchangeInfo":
		symbols := f.ExchangeInfo
		if symbols == nil {
			for _, m := range f.Markets {
				symbols = append(symbols, map[string]any{"symbol": m, "status": "TRADING", "contractType": "PERPETUAL"})
			}
		}
		writeJSON(w, map[string]any{"symbols": symbols})
	case "/fapi/v1/klines":
		step := intervalMs(q.Get("interval"))
		end, _ := strconv.ParseInt(q.Get("endTime"), 10, 64)
		if q.Get("endTime") == "" {
			end = now
		}
		limit, _ := strconv.ParseInt(q.Get("limit"), 10, 64)
		last := min(end/step*step, now/step*step)
		first := max(ceilTo(f.ListedMs, step), last-(limit-1)*step)
		row := f.Kline
		if row == nil {
			row = Row
		}
		rows := [][]any{}
		for t := first; t <= last; t += step {
			rows = append(rows, row(q.Get("symbol"), t, step))
		}
		writeJSON(w, rows)
	case "/fapi/v1/fundingRate":
		fstep := f.FundingStepMs
		if fstep == 0 {
			fstep = 8 * 3_600_000
		}
		start, _ := strconv.ParseInt(q.Get("startTime"), 10, 64)
		limit, _ := strconv.Atoi(q.Get("limit"))
		rows := []map[string]any{}
		for t := ceilTo(max(start, f.ListedMs), fstep); t <= now && len(rows) < limit; t += fstep {
			rate := 0.0001
			if f.Rate != nil {
				rate = f.Rate(q.Get("symbol"), t)
			}
			rows = append(rows, map[string]any{"symbol": q.Get("symbol"), "fundingTime": t,
				"fundingRate": fmt.Sprintf("%.8f", rate), "markPrice": "100.0", "rateType": "Regular"})
		}
		writeJSON(w, rows)
	case "/fapi/v2/ticker/price":
		rows := []map[string]string{}
		for s, p := range f.Prices {
			rows = append(rows, map[string]string{"symbol": s, "price": p})
		}
		writeJSON(w, rows)
	case "/fapi/v1/ticker/24hr":
		writeJSON(w, f.Tickers24h)
	default:
		http.NotFound(w, r)
	}
}

func intervalMs(interval string) int64 {
	n, _ := strconv.ParseInt(interval[:len(interval)-1], 10, 64)
	return n * map[byte]int64{'m': 60_000, 'h': 3_600_000, 'd': 86_400_000}[interval[len(interval)-1]]
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}
