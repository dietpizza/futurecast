package candles

import (
	"math"
	"math/rand/v2"
	"reflect"
	"testing"
)

const minute = int64(60_000)

var start = int64(1_735_689_600_000) // 2025-01-01 00:00 UTC

func walk(seed uint64, n int, price float64) Bars {
	rng := rand.New(rand.NewPCG(seed, 1))
	var b Bars
	p := price
	for i := 0; i < n; i++ {
		p *= math.Exp(rng.NormFloat64() * 0.001)
		b.Append(Bar{Time: start + int64(i)*minute, Open: p, High: p * 1.001, Low: p * 0.999, Close: p, Volume: 1,
			QuoteVolume: p, Count: 3, TakerBuyVolume: 0.5, TakerBuyQuoteVolume: p / 2})
	}
	return b
}

func drop(b Bars, from, to int) Bars {
	var idx []int
	for i := 0; i < b.Len(); i++ {
		if i < from || i >= to {
			idx = append(idx, i)
		}
	}
	return b.Take(idx)
}

func TestCrossAssetAlignmentWithGapFill(t *testing.T) {
	target, btc := walk(8, 60, 0.15), walk(7, 60, 65000)
	gapped := drop(btc, 10, 12) // a 2-minute gap

	c, err := Align(minute, 3, Asset{"DOGEUSDT", target}, Asset{"BTCUSDT", gapped})
	if err != nil {
		t.Fatal(err)
	}
	// 2 minutes ≤ the 3-minute fill limit: every minute stays
	if c.Len() != 60 || !c.Has("DOGEUSDT", "close") || !c.Has("BTCUSDT", "close") {
		t.Fatalf("%d candles, columns %v", c.Len(), c.Names)
	}
	closes, vols := c.Col("BTCUSDT", "close"), c.Col("BTCUSDT", "volume")
	for i := range closes {
		if math.IsNaN(closes[i]) {
			t.Fatalf("NaN close at %d", i)
		}
	}
	if closes[10] != btc.Close[9] || closes[11] != btc.Close[9] || vols[10] != 0 || vols[11] != 0 || c.Col("BTCUSDT", "count")[10] != 0 {
		t.Error("filled candles carry the last price and trade nothing")
	}
	if c.Has("DOGEUSDT", "funding_rate") {
		t.Error("no funding column without funding")
	}
}

func TestAGapLongerThanTheFillLimitDropsItsCandles(t *testing.T) {
	target, btc := walk(8, 60, 0.15), drop(walk(7, 60, 65000), 20, 25)
	c, err := Align(minute, 3, Asset{"DOGEUSDT", target}, Asset{"BTCUSDT", btc})
	if err != nil {
		t.Fatal(err)
	}
	for _, ts := range c.OpenTime {
		if ts >= start+20*minute && ts < start+25*minute {
			t.Fatalf("gap candle %d kept", (ts-start)/minute)
		}
		// pandas groups the gap with the candle before it, so that one goes too
		if ts == start+19*minute {
			t.Fatal("the candle before a long gap should be dropped")
		}
	}
	if c.Len() != 54 || c.OpenTime[c.Len()-1] != start+59*minute {
		t.Fatalf("%d candles, last %d", c.Len(), c.OpenTime[c.Len()-1])
	}
}

func TestAlignSpansTheCommonRange(t *testing.T) {
	target, btc := walk(8, 60, 0.15), walk(7, 60, 65000)
	c, err := Align(minute, 3, Asset{"DOGEUSDT", target.Slice(5, 60)}, Asset{"BTCUSDT", btc.Slice(0, 50)})
	if err != nil {
		t.Fatal(err)
	}
	if c.Len() != 45 || c.OpenTime[0] != start+5*minute || c.OpenTime[44] != start+49*minute {
		t.Fatalf("%d candles from %d", c.Len(), c.OpenTime[0])
	}
	if _, err := Align(minute, 3, Asset{"A", target.Slice(0, 10)}, Asset{"B", btc.Slice(20, 30)}); err == nil {
		t.Error("no overlap should be an error")
	}
	if _, err := Align(minute, 3, Asset{"A", Bars{}}); err == nil {
		t.Error("empty target should be an error")
	}
}

func TestFundingIsCarriedForwardLikePrices(t *testing.T) {
	target, btc := walk(8, 20, 0.15), walk(7, 20, 65000)
	target.FundingRate = make([]float64, 20)
	for i := range target.FundingRate {
		target.FundingRate[i] = math.NaN()
		if i >= 4 {
			target.FundingRate[i] = float64(i)
		}
	}
	target = drop(target, 8, 10)
	c, _ := Align(minute, 3, Asset{"X", target}, Asset{"B", btc})
	f := c.Col("X", "funding_rate")
	if !math.IsNaN(f[0]) || f[8] != 7 || f[9] != 7 || f[10] != 10 || c.Has("B", "funding_rate") {
		t.Fatal(f)
	}
}

func TestMergeKeepsOneCandlePerTime(t *testing.T) {
	a, b := walk(1, 5, 10), walk(2, 5, 20)
	merged := Merge(true, a.Slice(0, 3), b.Slice(2, 5))
	if merged.Len() != 5 || merged.Close[2] != b.Close[2] || merged.Close[1] != a.Close[1] {
		t.Fatal("keep last")
	}
	merged = Merge(false, a.Slice(0, 3), b.Slice(2, 5))
	if merged.Close[2] != a.Close[2] {
		t.Fatal("keep first")
	}
	backwards := Merge(true, b.Slice(3, 5), a.Slice(0, 3))
	if !reflect.DeepEqual(backwards.Time, a.Time) {
		t.Fatal("sorted")
	}
}

func TestFundingKnownAtIsStrictlyBefore(t *testing.T) {
	f := Funding{Time: []int64{100, 200}, Rate: []float64{1, 2}}
	got := f.KnownAt([]int64{50, 100, 101, 200, 250})
	if !math.IsNaN(got[0]) || got[1] != 1 && !math.IsNaN(got[1]) || got[2] != 1 || got[3] != 1 || got[4] != 2 {
		t.Fatal(got)
	}
	if !math.IsNaN(got[1]) {
		t.Fatal("a candle closing at a settlement still has the previous rate", got[1])
	}
}

func TestIntervalMs(t *testing.T) {
	for in, want := range map[string]int64{"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000} {
		if got, err := IntervalMs(in); err != nil || got != want {
			t.Errorf("IntervalMs(%s) = %d, %v", in, got, err)
		}
	}
	for _, bad := range []string{"", "m", "5x", "h1"} {
		if _, err := IntervalMs(bad); err == nil {
			t.Errorf("IntervalMs(%q) should fail", bad)
		}
	}
}
