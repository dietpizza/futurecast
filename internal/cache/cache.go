// Package cache keeps downloaded closed candles and settled funding rates on disk, one Parquet
// file per Binance market (and interval), in the same format the Python app wrote: the files in
// ~/.deltarisk/cache from before the Go rewrite keep working.
package cache

import (
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"deltarisk/internal/candles"
	"deltarisk/internal/paths"
	"deltarisk/internal/pq"
)

// One lock per file for the whole process: the stream loop and history downloads each use their own client.
var (
	locksMu sync.Mutex
	locks   = map[string]*sync.Mutex{}
)

func lockFor(path string) *sync.Mutex {
	locksMu.Lock()
	defer locksMu.Unlock()
	if locks[path] == nil {
		locks[path] = &sync.Mutex{}
	}
	return locks[path]
}

// Span is the open times a candle file fully covers. From 0 means back to the listing.
type Span struct{ From, To int64 }

// Candles holds closed candles already downloaded, one file per Binance market and interval.
//
// Closed candles never change, so each file holds one span of open times that was fully
// downloaded: every candle Binance has between covered_from and covered_to is in the file (an
// open time missing inside the span is an exchange gap, not a hole in the cache). Downloads that
// join the span extend it; a newer download that cannot join replaces it. Files keep the newest
// MaxRows candles.
type Candles struct {
	Dir     string
	MaxRows int
}

// CandleColumns are the file's columns.
var CandleColumns = []string{"time", "open", "high", "low", "close", "volume", "quote_volume",
	"count", "taker_buy_volume", "taker_buy_quote_volume"}

// NewCandles keeps candles in dir (relative to the home folder unless absolute).
func NewCandles(dir string, maxRows int) *Candles {
	return &Candles{Dir: paths.Resolve(dir), MaxRows: maxRows}
}

func (c *Candles) path(market, interval string) string {
	return filepath.Join(c.Dir, strings.ToUpper(market)+"_"+interval+".parquet")
}

func (c *Candles) read(path string) (candles.Bars, *Span) {
	if _, err := os.Stat(path); err != nil {
		return candles.Bars{}, nil
	}
	tbl, err := pq.Read(path)
	if err == nil {
		var bars candles.Bars
		var span Span
		bars, err = barsFromTable(tbl)
		if err == nil {
			span.From, err = metaInt(tbl.Meta, "covered_from")
		}
		if err == nil {
			span.To, err = metaInt(tbl.Meta, "covered_to")
		}
		if err == nil {
			return bars, &span
		}
	}
	slog.Warn("ignoring unreadable candle cache", "path", path, "err", err)
	return candles.Bars{}, nil
}

// Load is the cached candles of a market and interval and the span they cover (nil without a file).
func (c *Candles) Load(market, interval string) (candles.Bars, *Span) {
	path := c.path(market, interval)
	mu := lockFor(path)
	mu.Lock()
	defer mu.Unlock()
	return c.read(path)
}

// Store merges candles downloaded for open times from..to into the file.
func (c *Candles) Store(market, interval string, stepMs int64, rows candles.Bars, from, to int64) {
	path := c.path(market, interval)
	mu := lockFor(path)
	mu.Lock()
	defer mu.Unlock()
	have, span := c.read(path)
	if span != nil && from <= span.To+stepMs && span.From <= to+stepMs {
		rows = candles.Merge(true, have, rows)
		from, to = min(from, span.From), max(to, span.To)
	} else if span != nil && span.To >= to {
		return // an older, separate stretch: keep the newer one on disk
	} else {
		rows = candles.Merge(true, rows)
	}
	rows = rows.Between(from, to)
	if c.MaxRows > 0 && rows.Len() > c.MaxRows {
		rows = rows.Tail(c.MaxRows)
		from = rows.Time[0]
	}
	cols := []pq.Column{
		{Name: "time", Int64: orEmpty(rows.Time)},
		{Name: "open", Float64: orEmptyF(rows.Open)},
		{Name: "high", Float64: orEmptyF(rows.High)},
		{Name: "low", Float64: orEmptyF(rows.Low)},
		{Name: "close", Float64: orEmptyF(rows.Close)},
		{Name: "volume", Float64: orEmptyF(rows.Volume)},
		{Name: "quote_volume", Float64: orEmptyF(rows.QuoteVolume)},
		{Name: "count", Int64: orEmpty(rows.Count)},
		{Name: "taker_buy_volume", Float64: orEmptyF(rows.TakerBuyVolume)},
		{Name: "taker_buy_quote_volume", Float64: orEmptyF(rows.TakerBuyQuoteVolume)},
	}
	meta := map[string]string{"covered_from": strconv.FormatInt(from, 10), "covered_to": strconv.FormatInt(to, 10)}
	if err := pq.Write(path, cols, meta); err != nil {
		slog.Warn("could not write candle cache", "path", path, "err", err)
	}
}

// Funding holds settled funding rates already downloaded, one file per Binance market.
//
// A file holds every settlement from covered_from (ms) on, as known at fetched_at (ms): no
// settlement before fetched_at can be missing, so after a restart only the ones since need
// downloading.
type Funding struct {
	Dir string
}

// NewFunding keeps funding rates in dir (relative to the home folder unless absolute).
func NewFunding(dir string) *Funding { return &Funding{Dir: paths.Resolve(dir)} }

func (f *Funding) path(market string) string {
	return filepath.Join(f.Dir, strings.ToUpper(market)+".parquet")
}

// FundingFile is what a funding file holds.
type FundingFile struct {
	CoveredFrom, FetchedAt int64
	Events                 candles.Funding
}

// Load is the market's file, or nil.
func (f *Funding) Load(market string) *FundingFile {
	path := f.path(market)
	mu := lockFor(path)
	mu.Lock()
	defer mu.Unlock()
	if _, err := os.Stat(path); err != nil {
		return nil
	}
	tbl, err := pq.Read(path)
	if err == nil {
		out := &FundingFile{}
		times, rates := tbl.Col("time"), tbl.Col("funding_rate")
		if times == nil || times.Int64 == nil && times.Len() > 0 || rates == nil {
			err = errMissing("time/funding_rate")
		}
		if err == nil {
			out.CoveredFrom, err = metaInt(tbl.Meta, "covered_from")
		}
		if err == nil {
			out.FetchedAt, err = metaInt(tbl.Meta, "fetched_at")
		}
		if err == nil {
			out.Events = candles.Funding{Time: times.Int64, Rate: rates.Floats()}
			return out
		}
	}
	slog.Warn("ignoring unreadable funding cache", "path", path, "err", err)
	return nil
}

// Store replaces the market's file.
func (f *Funding) Store(market string, coveredFrom, fetchedAt int64, events candles.Funding) {
	path := f.path(market)
	mu := lockFor(path)
	mu.Lock()
	defer mu.Unlock()
	cols := []pq.Column{{Name: "time", Int64: orEmpty(events.Time)}, {Name: "funding_rate", Float64: orEmptyF(events.Rate)}}
	meta := map[string]string{"covered_from": strconv.FormatInt(coveredFrom, 10), "fetched_at": strconv.FormatInt(fetchedAt, 10)}
	if err := pq.Write(path, cols, meta); err != nil {
		slog.Warn("could not write funding cache", "path", path, "err", err)
	}
}

type errMissing string

func (e errMissing) Error() string { return "missing columns " + string(e) }

func metaInt(meta map[string]string, key string) (int64, error) {
	v, ok := meta[key]
	if !ok {
		return 0, errMissing(key)
	}
	return strconv.ParseInt(v, 10, 64)
}

func barsFromTable(t *pq.Table) (candles.Bars, error) {
	get := func(name string) (*pq.Column, error) {
		c := t.Col(name)
		if c == nil {
			return nil, errMissing(name)
		}
		return c, nil
	}
	var b candles.Bars
	for _, name := range CandleColumns {
		c, err := get(name)
		if err != nil {
			return candles.Bars{}, err
		}
		switch name {
		case "time":
			b.Time = c.Int64
		case "count":
			if c.Int64 != nil {
				b.Count = c.Int64
			} else {
				for _, v := range c.Floats() {
					b.Count = append(b.Count, int64(v))
				}
			}
		default:
			col := c.Floats()
			switch name {
			case "open":
				b.Open = col
			case "high":
				b.High = col
			case "low":
				b.Low = col
			case "close":
				b.Close = col
			case "volume":
				b.Volume = col
			case "quote_volume":
				b.QuoteVolume = col
			case "taker_buy_volume":
				b.TakerBuyVolume = col
			case "taker_buy_quote_volume":
				b.TakerBuyQuoteVolume = col
			}
		}
	}
	if b.Len() == 0 {
		return candles.Bars{}, nil
	}
	return b, nil
}

// orEmpty keeps an empty column typed: pq.Write tells int from float by which slice is non-nil.
func orEmpty(v []int64) []int64 {
	if v == nil {
		return []int64{}
	}
	return v
}

func orEmptyF(v []float64) []float64 {
	if v == nil {
		return []float64{}
	}
	return v
}
