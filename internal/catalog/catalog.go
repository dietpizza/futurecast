// Package catalog knows CoinDCX pair names: their base and quote, sector and benchmark, and the
// pairs the add-coin search offers (a curated registry plus every trading Binance USDT perpetual
// under its CoinDCX name, 1000PEPEUSDT -> B-1000PEPE_USDT). CoinDCX itself is never queried.
package catalog

import (
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
)

// DetectExchange says "coindcx" for CoinDCX pair names (B-DOGE_USDT, I-BTC_INR, anything with
// an underscore) and "binance" otherwise (DOGEUSDT).
func DetectExchange(symbol string) string {
	s := strings.ToUpper(strings.TrimSpace(symbol))
	if strings.HasPrefix(s, "B-") || strings.HasPrefix(s, "I-") || strings.Contains(s, "_") {
		return "coindcx"
	}
	return "binance"
}

var quotes = []string{"USDT", "USDC", "FDUSD", "BUSD", "TUSD", "EUR", "TRY", "INR", "BTC", "ETH"}

// BaseQuote splits a pair name: B-DOGE_USDT -> DOGE, USDT; I-BTC_INR -> BTC, INR; DOGEUSDT -> DOGE, USDT.
func BaseQuote(symbol string) (base, quote string) {
	s := strings.ToUpper(strings.TrimSpace(symbol))
	if strings.Contains(s, "_") {
		parts := strings.Split(s, "_")
		base, quote = parts[0], parts[1]
		if strings.HasPrefix(base, "B-") || strings.HasPrefix(base, "I-") {
			base = base[2:]
		}
		return base, quote
	}
	for _, q := range quotes {
		if strings.HasSuffix(s, q) && len(s) > len(q) {
			return s[:len(s)-len(q)], q
		}
	}
	return s, "USDT"
}

// Sector is the market sector of a pair (MEME, L1, DEFI, NFT_GAMING, AI, INFRA, MAJOR), else GENERAL_ALT.
func Sector(symbol string) string {
	base, _ := BaseQuote(symbol)
	if s, ok := sectors[base]; ok {
		return s
	}
	return "GENERAL_ALT"
}

// Benchmark is the pair a coin's moves are measured against, by sector: meme coins -> DOGE,
// alt-L1s -> SOL, DeFi -> ETH, NFT/gaming -> BLUR, AI -> NEAR, infrastructure -> LINK; category
// leaders, majors and unlisted coins -> BTC. A coin is never its own benchmark (BTC -> ETH): its
// candle columns would collide with the benchmark's. INR pairs get the INR pair.
func Benchmark(symbol string) string {
	base, quote := BaseQuote(symbol)
	leader, ok := sectorLeaders[sectors[base]]
	if !ok {
		leader = "BTC"
	}
	if leader == base || base == "ETH" || base == "SOL" || base == "DOGE" || base == "BLUR" {
		leader = "BTC"
	}
	if base == "BTC" {
		leader = "ETH"
	}
	if quote == "INR" {
		return "I-" + leader + "_INR"
	}
	return "B-" + leader + "_" + quote
}

// RegistryEntry is a curated pair.
type RegistryEntry struct {
	Symbol, Name, Description string
}

// Instrument is one searchable pair, as /api/coindcx/futures sends it.
type Instrument struct {
	Symbol      string `json:"symbol"`
	Base        string `json:"base"`
	Name        string `json:"name"`
	Sector      string `json:"sector"`
	Description string `json:"description"`
	CoinDCXURL  string `json:"coindcx_url"`
}

func instrument(symbol, name, description string) Instrument {
	base, _ := BaseQuote(symbol)
	url := "https://coindcx.com/trade/" + symbol
	if strings.HasPrefix(symbol, "B-") {
		url = "https://coindcx.com/futures/" + symbol
	}
	return Instrument{Symbol: symbol, Base: base, Name: name, Sector: Sector(symbol), Description: description, CoinDCXURL: url}
}

// Perpetual is a Binance futures market as exchangeInfo lists it.
type Perpetual struct {
	Symbol         string `json:"symbol"`
	Status         string `json:"status"`
	ContractType   string `json:"contractType"`
	BaseAsset      string `json:"baseAsset"`
	QuoteAsset     string `json:"quoteAsset"`
	UnderlyingType string `json:"underlyingType"`
}

// PerpetualLister lists the trading Binance perpetuals by symbol (binance.Client does).
type PerpetualLister interface {
	Perpetuals() (map[string]Perpetual, error)
}

// CoinDCX lists crypto and commodity perpetuals, never Binance's stock / FX / index ones
var listedUnderlyings = map[string]bool{"COIN": true, "COMMODITY": true}

// TTL is how long a catalog with Binance's pairs is reused.
const TTL = time.Hour

// Catalog builds the search list and keeps it for TTL.
type Catalog struct {
	binance PerpetualLister
	mu      sync.Mutex
	items   []Instrument
	at      time.Time // zero after a Binance outage: retry on the next search
}

// New makes a catalog that lists Binance's pairs through binance.
func New(binance PerpetualLister) *Catalog { return &Catalog{binance: binance} }

// binancePairs are the CoinDCX names of the trading Binance USDT perpetuals; empty when Binance is unreachable.
func (c *Catalog) binancePairs() []string {
	perps, err := c.binance.Perpetuals()
	if err != nil {
		slog.Debug("Binance perpetuals fetch skipped", "err", err)
		return nil
	}
	var out []string
	for _, p := range perps {
		if p.QuoteAsset == "USDT" && listedUnderlyings[p.UnderlyingType] && isASCII(p.BaseAsset) { // CoinDCX has no CJK-named pairs
			out = append(out, "B-"+p.BaseAsset+"_USDT")
		}
	}
	sort.Strings(out)
	return out
}

func isASCII(s string) bool {
	for _, r := range s {
		if r > unicode.MaxASCII {
			return false
		}
	}
	return true
}

// Instruments is the curated registry, then every pair with a trading Binance perpetual.
func (c *Catalog) Instruments() []Instrument {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.items != nil && !c.at.IsZero() && time.Since(c.at) < TTL {
		return c.items
	}
	raw := c.binancePairs()
	var items []Instrument
	seen := map[string]bool{}
	for _, r := range Registry {
		sym := strings.ToUpper(strings.TrimSpace(r.Symbol))
		if !seen[sym] {
			seen[sym] = true
			items = append(items, instrument(sym, r.Name, r.Description))
		}
	}
	for _, sym := range raw {
		if !seen[sym] {
			seen[sym] = true
			base, quote := BaseQuote(sym)
			items = append(items, instrument(sym, base, "CoinDCX "+quote+"-Margined Perpetual Contract for "+base+"."))
		}
	}
	c.items = items
	c.at = time.Time{}
	if len(raw) > 0 {
		c.at = time.Now()
	}
	return items
}
