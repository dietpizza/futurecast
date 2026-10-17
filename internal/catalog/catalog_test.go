package catalog

import (
	"errors"
	"testing"
)

type fakeBinance struct {
	fail  bool
	calls int
}

func (f *fakeBinance) Perpetuals() (map[string]Perpetual, error) {
	f.calls++
	if f.fail {
		return nil, errors.New("offline")
	}
	out := map[string]Perpetual{}
	for _, p := range [][3]string{
		{"1000PEPE", "USDT", "COIN"}, {"ZRO", "USDT", "COIN"}, {"XAU", "USDT", "COMMODITY"},
		{"TSLA", "USDT", "EQUITY"}, {"币安人生", "USDT", "COIN"}, {"BTC", "USDC", "COIN"},
	} {
		out[p[0]+p[1]] = Perpetual{Symbol: p[0] + p[1], BaseAsset: p[0], QuoteAsset: p[1], UnderlyingType: p[2]}
	}
	return out, nil
}

func symbols(items []Instrument) map[string]bool {
	out := map[string]bool{}
	for _, it := range items {
		out[it.Symbol] = true
	}
	return out
}

func TestCatalogListsBinanceCryptoAndCommodityPerpetuals(t *testing.T) {
	got := symbols(New(&fakeBinance{}).Instruments())
	for _, s := range []string{"B-1000PEPE_USDT", "B-ZRO_USDT", "B-XAU_USDT", "B-DOGE_USDT"} { // DOGE: curated registry
		if !got[s] {
			t.Errorf("missing %s", s)
		}
	}
	for _, s := range []string{"B-TSLA_USDT", "B-币安人生_USDT", "B-BTC_USDC"} {
		if got[s] {
			t.Errorf("unexpected %s", s)
		}
	}
}

func TestCatalogFallsBackToRegistryAndRetriesAfterOutage(t *testing.T) {
	offline := &fakeBinance{fail: true}
	c := New(offline)
	got := symbols(c.Instruments())
	if len(got) != len(Registry) {
		t.Fatalf("got %d pairs, want the %d registry pairs", len(got), len(Registry))
	}
	for _, r := range Registry {
		if !got[r.Symbol] {
			t.Errorf("missing %s", r.Symbol)
		}
	}
	c.Instruments()
	if offline.calls != 2 { // a failed fetch is not cached for the hour
		t.Fatalf("calls = %d", offline.calls)
	}

	online := &fakeBinance{}
	c = New(online)
	c.Instruments()
	c.Instruments()
	if online.calls != 1 {
		t.Fatalf("a good list should be reused, calls = %d", online.calls)
	}
}

func TestInstrumentFields(t *testing.T) {
	items := New(&fakeBinance{}).Instruments()
	byName := map[string]Instrument{}
	for _, it := range items {
		byName[it.Symbol] = it
	}
	if got := byName["B-1000PEPE_USDT"]; got != (Instrument{"B-1000PEPE_USDT", "1000PEPE", "1000PEPE", "MEME",
		"CoinDCX USDT-Margined Perpetual Contract for 1000PEPE.", "https://coindcx.com/futures/B-1000PEPE_USDT"}) {
		t.Errorf("1000PEPE: %+v", got)
	}
	if got := byName["I-BTC_INR"]; got.CoinDCXURL != "https://coindcx.com/trade/I-BTC_INR" || got.Base != "BTC" || got.Sector != "MAJOR" {
		t.Errorf("I-BTC_INR: %+v", got)
	}
}

func TestDetectExchange(t *testing.T) {
	for sym, want := range map[string]string{
		"B-DOGE_USDT": "coindcx", "B-SOL_USDT": "coindcx", "I-BTC_INR": "coindcx", "DOGEUSDT": "binance", "BTCUSDT": "binance",
	} {
		if got := DetectExchange(sym); got != want {
			t.Errorf("DetectExchange(%s) = %s", sym, got)
		}
	}
}

func TestBaseQuote(t *testing.T) {
	for sym, want := range map[string][2]string{
		"B-DOGE_USDT": {"DOGE", "USDT"}, "I-BTC_INR": {"BTC", "INR"}, "DOGEUSDT": {"DOGE", "USDT"},
		"BTCUSDT": {"BTC", "USDT"}, "b-1000pepe_usdt": {"1000PEPE", "USDT"}, "XYZ": {"XYZ", "USDT"},
	} {
		if b, q := BaseQuote(sym); b != want[0] || q != want[1] {
			t.Errorf("BaseQuote(%s) = %s, %s", sym, b, q)
		}
	}
}

func TestSectorBenchmarks(t *testing.T) {
	if Sector("B-PEPE_USDT") != "MEME" || Sector("B-AVAX_USDT") != "L1" || Sector("B-WAL_USDT") != "GENERAL_ALT" {
		t.Error("sectors")
	}
	for sym, want := range map[string]string{
		"B-PEPE_USDT": "B-DOGE_USDT", "B-SHIB_USDT": "B-DOGE_USDT", // meme -> DOGE
		"B-SUI_USDT":  "B-SOL_USDT", // alt-L1 -> SOL
		"B-AAVE_USDT": "B-ETH_USDT", "B-SAND_USDT": "B-BLUR_USDT", "B-RENDER_USDT": "B-NEAR_USDT", "B-PYTH_USDT": "B-LINK_USDT",
		// sector leaders, majors and unlisted coins -> BTC
		"B-DOGE_USDT": "B-BTC_USDT", "B-SOL_USDT": "B-BTC_USDT", "B-BLUR_USDT": "B-BTC_USDT", "B-LINK_USDT": "B-BTC_USDT",
		"B-ETH_USDT": "B-BTC_USDT", "B-WAL_USDT": "B-BTC_USDT",
		"I-ETH_INR": "I-BTC_INR", // INR pairs get the INR pair
	} {
		if got := Benchmark(sym); got != want {
			t.Errorf("Benchmark(%s) = %s, want %s", sym, got, want)
		}
	}
}

func TestCoinIsNeverItsOwnBenchmark(t *testing.T) {
	// BTC benchmarked against itself duplicated its candle columns
	if Benchmark("B-BTC_USDT") != "B-ETH_USDT" || Benchmark("I-BTC_INR") != "I-ETH_INR" {
		t.Error("BTC should benchmark against ETH")
	}
	for _, sym := range []string{"B-BTC_USDT", "B-ETH_USDT", "B-SOL_USDT", "B-DOGE_USDT", "B-BLUR_USDT", "B-NEAR_USDT", "B-LINK_USDT"} {
		if Benchmark(sym) == sym {
			t.Errorf("%s is its own benchmark", sym)
		}
	}
}
