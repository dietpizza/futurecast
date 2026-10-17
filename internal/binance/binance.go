// Package binance reads Binance USD-M futures public market data (public endpoints only, no
// API keys). Every candle, funding rate and live price for a CoinDCX pair comes from the Binance
// perpetual behind it: B-SOL_USDT -> SOLUSDT, B-1000PEPE_USDT -> 1000PEPEUSDT.
package binance

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"deltarisk/internal/cache"
	"deltarisk/internal/candles"
	"deltarisk/internal/catalog"
)

const (
	FapiURL        = "https://fapi.binance.com"
	klinePage      = 1000 // klines weight is 5 up to 1000 rows, 10 above: 1000-row pages are cheapest per candle
	WeightLimit1m  = 2400
	fundingPage    = 1000
	FundingLimit5m = 500        // fundingRate has its own budget, outside the weight count: 500 requests / 5 min / IP
	dayMs          = 86_400_000 // funding settles every 1h, 4h or 8h, so a day back always holds a settlement
)

var perpetualTypes = map[string]bool{"PERPETUAL": true, "TRADIFI_PERPETUAL": true} // TRADIFI: gold, oil, ... (B-XAU_USDT)

// Binance lists some low-priced coins per 1000 / 1M units (1000PEPEUSDT, 1MBABYDOGEUSDT)
var unitPrefixes = []struct {
	prefix string
	mult   int64
}{{"", 1}, {"1000", 1_000}, {"1000000", 1_000_000}, {"1M", 1_000_000}}

// Market is the Binance perpetual behind a pair; Multiplier coins per Binance price unit (1000
// for 1000PEPEUSDT standing in for B-PEPE_USDT, 1 when the names match).
type Market struct {
	Symbol     string
	Multiplier int64
}

// Candidates are the Binance perpetual names a pair could trade under, exact name first:
// B-SOL_USDT -> SOLUSDT, then 1000SOLUSDT, 1000000SOLUSDT, 1MSOLUSDT. INR pairs map to the USDT
// perpetual (Binance has no INR markets): I-BTC_INR -> BTCUSDT.
func Candidates(symbol string) []Market {
	base, quote := catalog.BaseQuote(symbol)
	if quote == "INR" {
		quote = "USDT"
	}
	out := make([]Market, len(unitPrefixes))
	for i, u := range unitPrefixes {
		out[i] = Market{u.prefix + base + quote, u.mult}
	}
	return out
}

// Clock is the time source, swapped in tests.
type Clock interface {
	Now() time.Time
	Sleep(time.Duration)
}

type realClock struct{}

func (realClock) Now() time.Time        { return time.Now() }
func (realClock) Sleep(d time.Duration) { time.Sleep(d) }

func nowMs(c Clock) int64 { return c.Now().UnixMilli() }

// StatusError is a non-2xx reply.
type StatusError struct {
	Code int
	Path string
}

func (e *StatusError) Error() string { return fmt.Sprintf("Binance %s: HTTP %d", e.Path, e.Code) }

// Options configure a Client; zero values take the defaults.
type Options struct {
	BaseURL       string
	Timeout       time.Duration // 15 s
	MaxRetries    int           // 3
	MarketsTTL    time.Duration // 1 h
	WeightBudget  int           // 80% of 2,400 a minute
	FundingBudget int           // 80% of 500 per 5 minutes
	Candles       *cache.Candles
	Funding       *cache.Funding
	Clock         Clock
}

type fundingHeld struct {
	coveredFrom, fetchedAt int64
	events                 candles.Funding
}

// Client is a public read-only client for Binance USD-M perpetuals, keyed by CoinDCX pair. Safe
// for concurrent use.
type Client struct {
	baseURL       string
	http          *http.Client
	maxRetries    int
	marketsTTL    time.Duration
	weightBudget  int
	fundingBudget int
	clock         Clock
	candleCache   *cache.Candles
	fundingCache  *cache.Funding

	marketsMu   sync.Mutex
	perpetuals  map[string]catalog.Perpetual
	perpsAt     time.Time
	fundingMu   sync.Mutex
	fundingCall []time.Time
	funding     map[string]fundingHeld // Binance symbol -> settlements held
}

// New makes a client.
func New(o Options) *Client {
	if o.BaseURL == "" {
		o.BaseURL = FapiURL
	}
	if o.Timeout == 0 {
		o.Timeout = 15 * time.Second
	}
	if o.MaxRetries == 0 {
		o.MaxRetries = 3
	}
	if o.MarketsTTL == 0 {
		o.MarketsTTL = time.Hour
	}
	if o.WeightBudget == 0 {
		o.WeightBudget = WeightLimit1m * 8 / 10
	}
	if o.FundingBudget == 0 {
		o.FundingBudget = FundingLimit5m * 8 / 10
	}
	if o.Clock == nil {
		o.Clock = realClock{}
	}
	return &Client{
		baseURL: strings.TrimRight(o.BaseURL, "/"), http: &http.Client{Timeout: o.Timeout},
		maxRetries: o.MaxRetries, marketsTTL: o.MarketsTTL, weightBudget: o.WeightBudget,
		fundingBudget: o.FundingBudget, clock: o.Clock, candleCache: o.Candles, fundingCache: o.Funding,
		funding: map[string]fundingHeld{},
	}
}

func seconds(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }

// get fetches path and decodes the JSON reply into out, with retries: it honours 429/418
// Retry-After and pauses before the per-minute weight cap.
func (c *Client) get(path string, params url.Values, out any) error {
	u := c.baseURL + path
	if len(params) > 0 {
		u += "?" + params.Encode()
	}
	for attempt := 1; attempt <= c.maxRetries; attempt++ {
		req, err := http.NewRequest(http.MethodGet, u, nil)
		if err != nil {
			return err
		}
		req.Header.Set("User-Agent", "deltarisk-research/1.0")
		resp, err := c.http.Do(req)
		if err != nil {
			if attempt == c.maxRetries {
				slog.Error("Binance request failed", "path", path, "attempts", attempt, "err", err)
				return err
			}
			c.clock.Sleep(seconds(float64(attempt) * 1.5))
			continue
		}
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			if attempt == c.maxRetries {
				return err
			}
			c.clock.Sleep(seconds(float64(attempt) * 1.5))
			continue
		}

		if resp.StatusCode == 418 || resp.StatusCode == 429 { // 418: IP ban for ignoring 429s
			wait := float64(5 * attempt)
			if v, err := strconv.ParseFloat(resp.Header.Get("Retry-After"), 64); err == nil {
				wait = v
			}
			slog.Warn("Binance rate limited", "status", resp.StatusCode, "path", path, "retry_after_s", wait)
			if attempt == c.maxRetries || wait > 120 {
				return &StatusError{resp.StatusCode, path}
			}
			c.clock.Sleep(seconds(wait))
			continue
		}
		if resp.StatusCode >= 500 && attempt < c.maxRetries {
			c.clock.Sleep(seconds(float64(attempt) * 1.5))
			continue
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return &StatusError{resp.StatusCode, path}
		}

		if used, _ := strconv.Atoi(resp.Header.Get("X-MBX-USED-WEIGHT-1M")); used >= c.weightBudget {
			now := float64(c.clock.Now().UnixNano()) / 1e9
			pause := 61 - math.Mod(now, 60) // weight resets each clock minute
			slog.Warn("Binance weight budget used up this minute; pausing", "used", used, "limit", WeightLimit1m, "pause_s", math.Round(pause))
			c.clock.Sleep(seconds(pause))
		}
		dec := json.NewDecoder(strings.NewReader(string(body)))
		dec.UseNumber()
		return dec.Decode(out)
	}
	return errors.New("unreachable retry loop exit")
}

// ───────── symbol mapping ─────────

// Perpetuals are the currently trading Binance perpetuals by symbol (exchangeInfo, refreshed
// hourly; the last good list keeps serving through an outage).
func (c *Client) Perpetuals() (map[string]catalog.Perpetual, error) {
	c.marketsMu.Lock()
	defer c.marketsMu.Unlock()
	if c.perpetuals != nil && c.clock.Now().Sub(c.perpsAt) < c.marketsTTL {
		return c.perpetuals, nil
	}
	var info struct {
		Symbols []catalog.Perpetual `json:"symbols"`
	}
	if err := c.get("/fapi/v1/exchangeInfo", nil, &info); err != nil {
		if c.perpetuals != nil {
			return c.perpetuals, nil
		}
		return nil, err
	}
	perps := map[string]catalog.Perpetual{}
	for _, s := range info.Symbols {
		if s.Status == "TRADING" && perpetualTypes[s.ContractType] {
			perps[s.Symbol] = s
		}
	}
	c.perpetuals, c.perpsAt = perps, c.clock.Now()
	return perps, nil
}

// Resolve is the Binance perpetual for a CoinDCX pair; ok is false when Binance has no such market.
func (c *Client) Resolve(symbol string) (m Market, ok bool, err error) {
	perps, err := c.Perpetuals()
	if err != nil {
		return Market{}, false, err
	}
	for _, m := range Candidates(symbol) {
		if _, ok := perps[m.Symbol]; ok {
			return m, true, nil
		}
	}
	return Market{}, false, nil
}

func (c *Client) require(symbol string) (Market, bool, error) {
	m, ok, err := c.Resolve(symbol)
	if err == nil && !ok {
		slog.Warn("no trading Binance perpetual found", "symbol", symbol)
	}
	return m, ok, err
}

// ───────── prices ─────────

// LivePrices is the last traded price per pair from one all-symbol ticker call, in the pair's
// own units; pairs without a market are left out, and a failed call gives an empty map.
func (c *Client) LivePrices(symbols []string) map[string]float64 {
	prices := map[string]float64{}
	var tickers []struct {
		Symbol string `json:"symbol"`
		Price  string `json:"price"`
	}
	if err := c.get("/fapi/v2/ticker/price", nil, &tickers); err != nil {
		slog.Debug("Binance live prices failed", "err", err)
		return prices
	}
	last := map[string]float64{}
	for _, t := range tickers {
		p, err := strconv.ParseFloat(t.Price, 64)
		if err != nil {
			slog.Debug("Binance live prices failed", "err", err)
			return map[string]float64{}
		}
		last[t.Symbol] = p
	}
	for _, sym := range symbols {
		m, ok, err := c.Resolve(sym)
		if err != nil {
			slog.Debug("Binance live prices failed", "err", err)
			return map[string]float64{}
		}
		if p, have := last[m.Symbol]; ok && have {
			prices[sym] = p / float64(m.Multiplier)
		}
	}
	return prices
}

// Ticker24h is one market's rolling 24h statistics as Binance sends them (prices in market units).
type Ticker24h struct {
	Symbol             string `json:"symbol"`
	LastPrice          string `json:"lastPrice"`
	PriceChangePercent string `json:"priceChangePercent"`
	QuoteVolume        string `json:"quoteVolume"`
}

// Tickers24h is every Binance USD-M symbol's 24h statistics in one call (weight 40).
func (c *Client) Tickers24h() ([]Ticker24h, error) {
	var out []Ticker24h
	err := c.get("/fapi/v1/ticker/24hr", nil, &out)
	return out, err
}

// TimedPrice is a close at its close time (ms).
type TimedPrice struct {
	AtMs  int64
	Close float64
}

// RecentCloses are a Binance market's last `minutes` closed 1-minute candles, oldest first
// (weight 1 under 100 candles, 2 under 500).
func (c *Client) RecentCloses(market string, minutes int) ([]TimedPrice, error) {
	var rows [][]any
	params := url.Values{"symbol": {market}, "interval": {"1m"}, "limit": {strconv.Itoa(minutes + 1)}}
	if err := c.get("/fapi/v1/klines", params, &rows); err != nil {
		return nil, err
	}
	now := nowMs(c.clock)
	var out []TimedPrice
	for _, r := range rows {
		k, err := parseKline(r)
		if err != nil {
			return nil, err
		}
		if at := k.closeMs + 1; at <= now {
			out = append(out, TimedPrice{at, k.bar.Close})
		}
	}
	return out, nil
}

// ───────── funding rates ─────────

// fundingSlot waits for room in the fundingRate budget (500 requests per 5 minutes, not
// reported in weight headers).
func (c *Client) fundingSlot() {
	c.fundingMu.Lock()
	now := c.clock.Now()
	calls := c.fundingCall
	for len(calls) > 0 && now.Sub(calls[0]) >= 5*time.Minute {
		calls = calls[1:]
	}
	at := now
	if len(calls) >= c.fundingBudget {
		at = calls[len(calls)-c.fundingBudget].Add(5 * time.Minute)
		if at.Before(now) {
			at = now
		}
	}
	c.fundingCall = append(calls, at)
	c.fundingMu.Unlock()
	if at.After(now) {
		slog.Warn("Binance funding-rate budget used up; pausing", "pause_s", at.Sub(now).Seconds())
		c.clock.Sleep(at.Sub(now))
	}
}

// FundingRates are the settled funding rates of the pair's Binance perpetual from startMs to
// endMs (endMs ≤ 0: now), oldest first; empty when Binance has no market for the pair. startMs
// is rounded down to a UTC day so nearby requests share a download. A download is reused while
// it was made at or after endMs, since no settlement before endMs can be missing from it;
// otherwise only the settlements after the newest one held are fetched. With a funding cache,
// downloads are kept on disk too, so a restarted app starts from them.
func (c *Client) FundingRates(symbol string, startMs, endMs int64) (candles.Funding, error) {
	m, ok, err := c.require(symbol)
	if err != nil || !ok {
		return candles.Funding{}, err
	}
	now := nowMs(c.clock)
	if endMs <= 0 {
		endMs = now
	}
	startMs = floorDiv(startMs, dayMs) * dayMs

	c.fundingMu.Lock()
	held, have := c.funding[m.Symbol]
	c.fundingMu.Unlock()
	if !have && c.fundingCache != nil {
		if f := c.fundingCache.Load(m.Symbol); f != nil {
			held, have = fundingHeld{f.CoveredFrom, f.FetchedAt, f.Events}, true
			c.fundingMu.Lock()
			if _, set := c.funding[m.Symbol]; !set {
				c.funding[m.Symbol] = held
			}
			c.fundingMu.Unlock()
		}
	}
	if !have || held.coveredFrom > startMs {
		held = fundingHeld{coveredFrom: startMs}
	}
	events := held.events
	if held.fetchedAt < endMs {
		cursor := held.coveredFrom
		if events.Len() > 0 {
			cursor = events.Time[events.Len()-1] + 1
		}
		var fresh candles.Funding
		for {
			c.fundingSlot()
			var page []struct {
				FundingTime int64  `json:"fundingTime"`
				FundingRate string `json:"fundingRate"`
			}
			params := url.Values{"symbol": {m.Symbol}, "startTime": {strconv.FormatInt(cursor, 10)}, "limit": {strconv.Itoa(fundingPage)}}
			if err := c.get("/fapi/v1/fundingRate", params, &page); err != nil {
				return candles.Funding{}, err
			}
			for _, r := range page {
				if rate, err := strconv.ParseFloat(r.FundingRate, 64); err == nil && !math.IsNaN(rate) {
					fresh.Time = append(fresh.Time, r.FundingTime)
					fresh.Rate = append(fresh.Rate, rate)
				}
			}
			if len(page) < fundingPage {
				break
			}
			cursor = page[len(page)-1].FundingTime + 1
		}
		if fresh.Len() > 0 {
			events = candles.MergeFunding(events, fresh)
		}
		c.fundingMu.Lock()
		c.funding[m.Symbol] = fundingHeld{held.coveredFrom, now, events}
		c.fundingMu.Unlock()
		if c.fundingCache != nil {
			c.fundingCache.Store(m.Symbol, held.coveredFrom, now, events)
		}
	}
	return events.Between(startMs, endMs), nil
}

func floorDiv(a, b int64) int64 {
	q := a / b
	if (a%b != 0) && ((a < 0) != (b < 0)) {
		q--
	}
	return q
}

// ───────── candle history (closed candles only) ─────────

// CandleHistory is the newest `bars` fully closed candles of a CoinDCX pair from its Binance
// perpetual, oldest first, as of endMs (≤ 0: now). Candles come from the disk cache when the
// client has one; only the ones it is missing are downloaded. The still-forming candle is never
// included. Prices are Binance's (a 1000x listing stays in 1000x units). With funding, FundingRate
// holds the latest rate settled before each candle closed (NaN before the first on record).
// progress(done, total) counts downloaded candles. source is "binance:<MARKET>", or "" (and no
// candles) when Binance has no market for the pair.
func (c *Client) CandleHistory(symbol, interval string, bars int, endMs int64, funding bool,
	progress func(done, total int)) (candles.Bars, string, error) {
	m, ok, err := c.require(symbol)
	if err != nil || !ok {
		return candles.Bars{}, "", err
	}
	step, err := candles.IntervalMs(interval)
	if err != nil {
		return candles.Bars{}, "", err
	}
	if endMs <= 0 {
		endMs = nowMs(c.clock)
	}
	lastOpen := endMs/step*step - step
	df, err := c.closedKlines(m.Symbol, interval, lastOpen-int64(bars-1)*step, lastOpen, progress)
	if err != nil {
		return candles.Bars{}, "", err
	}
	if df.Len() == 0 {
		return candles.Bars{}, "", nil
	}
	df = df.Tail(bars)
	if funding {
		closeMs := make([]int64, df.Len())
		for i, t := range df.Time {
			closeMs[i] = t + step
		}
		events, err := c.FundingRates(symbol, closeMs[0]-step-dayMs, closeMs[len(closeMs)-1])
		if err != nil {
			return candles.Bars{}, "", err
		}
		df.FundingRate = events.KnownAt(closeMs)
	}
	return df, "binance:" + m.Symbol, nil
}

type span struct{ lo, hi int64 }

// closedKlines are the closed candles opening from firstOpen to lastOpen (ms), oldest first.
//
// With a cache, only the stretches outside its span are downloaded: newer candles up to
// lastOpen, then older ones back to firstOpen. A new span that cannot join the cached one (a
// request far older, or a gap longer than the cache keeps) is downloaded whole. Every finished
// download is merged into the cache; an interrupted walk back is kept too, since what it fetched
// still joins the span.
func (c *Client) closedKlines(market, interval string, firstOpen, lastOpen int64, progress func(done, total int)) (candles.Bars, error) {
	step := candles.MustIntervalMs(interval)
	store := c.candleCache
	var have candles.Bars
	var cached *cache.Span
	if store != nil {
		have, cached = store.Load(market, interval)
		if cached != nil && (lastOpen < cached.From-step || (lastOpen-cached.To)/step > int64(store.MaxRows)) {
			have, cached = candles.Bars{}, nil
		}
	}

	var jobs []span
	if cached == nil {
		jobs = []span{{firstOpen, lastOpen}}
	} else {
		if lastOpen > cached.To {
			jobs = append(jobs, span{cached.To + step, lastOpen})
		}
		if firstOpen < cached.From {
			jobs = append(jobs, span{firstOpen, cached.From - step})
		}
	}
	total := 0
	for _, j := range jobs {
		total += int((j.hi-j.lo)/step + 1)
	}

	frames := []candles.Bars{have}
	done := 0
	for _, j := range jobs {
		var rows []kline
		var report func(int)
		if progress != nil {
			base := done
			report = func(n int) { progress(base+n, total) }
		}
		listed, werr := c.walkBack(market, interval, j.lo, j.hi, &rows, report)
		got := klineFrame(rows, j.lo, j.hi)
		frames = append(frames, got)
		done += int((j.hi-j.lo)/step + 1)
		if store != nil && werr == nil { // finished: the whole stretch is covered
			from := j.lo
			if listed {
				from = 0
			}
			store.Store(market, interval, step, got, from, j.hi)
		} else if store != nil && got.Len() > 0 && (cached == nil || j.hi < cached.From) {
			// interrupted, but a walk back from the span's start still joins it
			store.Store(market, interval, step, got, got.Time[0], j.hi)
		}
		if werr != nil {
			return candles.Bars{}, werr
		}
	}
	all := candles.Merge(true, frames...)
	return all.Between(firstOpen, lastOpen), nil
}

// walkBack appends klines opening in [firstOpen, lastOpen] to rows, newest page first. listed
// is true when Binance has no candles older than the ones fetched (the listing).
func (c *Client) walkBack(market, interval string, firstOpen, lastOpen int64, rows *[]kline, report func(int)) (listed bool, err error) {
	step := candles.MustIntervalMs(interval)
	currEnd := lastOpen + step - 1
	for currEnd >= firstOpen {
		want := min((currEnd-firstOpen)/step+1, klinePage)
		var page [][]any
		params := url.Values{"symbol": {market}, "interval": {interval}, "endTime": {strconv.FormatInt(currEnd, 10)},
			"limit": {strconv.FormatInt(want, 10)}}
		if err := c.get("/fapi/v1/klines", params, &page); err != nil {
			return false, err
		}
		if len(page) == 0 {
			return true, nil
		}
		oldest := int64(math.MaxInt64)
		for _, r := range page {
			k, err := parseKline(r)
			if err != nil {
				return false, err
			}
			*rows = append(*rows, k)
			oldest = min(oldest, k.bar.Time)
		}
		if report != nil {
			report(len(*rows))
		}
		currEnd = oldest - 1
		if int64(len(page)) < want { // reached the listing date
			return true, nil
		}
	}
	return false, nil
}

type kline struct {
	bar     candles.Bar
	closeMs int64
}

// parseKline reads [open time, "open", "high", "low", "close", "volume", close time, "quote
// volume", trades, "taker buy volume", "taker buy quote volume", ...].
func parseKline(r []any) (kline, error) {
	if len(r) < 11 {
		return kline{}, fmt.Errorf("short kline row %v", r)
	}
	var errs []error
	f := func(v any) float64 {
		x, err := number(v)
		errs = append(errs, err)
		return x
	}
	k := kline{
		bar: candles.Bar{
			Time: int64(f(r[0])), Open: f(r[1]), High: f(r[2]), Low: f(r[3]), Close: f(r[4]), Volume: f(r[5]),
			QuoteVolume: f(r[7]), Count: int64(f(r[8])), TakerBuyVolume: f(r[9]), TakerBuyQuoteVolume: f(r[10]),
		},
		closeMs: int64(f(r[6])),
	}
	return k, errors.Join(errs...)
}

func number(v any) (float64, error) {
	switch x := v.(type) {
	case json.Number:
		return x.Float64()
	case string:
		return strconv.ParseFloat(x, 64)
	case float64:
		return x, nil
	}
	return 0, fmt.Errorf("not a number: %v", v)
}

// klineFrame is raw kline rows as candles, limited to open times firstOpen..lastOpen (the first
// row per open time), sorted.
func klineFrame(rows []kline, firstOpen, lastOpen int64) candles.Bars {
	var b candles.Bars
	for _, r := range rows {
		if r.bar.Time >= firstOpen && r.bar.Time <= lastOpen {
			b.Append(r.bar)
		}
	}
	return candles.Merge(false, b)
}
