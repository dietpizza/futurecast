package risk

import (
	"math"
	"math/rand/v2"
	"reflect"
	"sort"
	"testing"

	"deltarisk/engine/candles"
	"deltarisk/engine/regime"
)

const (
	sym   = "B-TEST_USDT"
	bench = "B-BTC_USDT"
)

var start = int64(1_767_225_600_000) // 2026-01-01

// randomWalk adds random-walk candles of sym to the chart, highs and lows from the path inside
// each candle.
func randomWalk(c *candles.Chart, sym string, n int, seed uint64, sigma float64) {
	rng := rand.New(rand.NewPCG(seed, 9))
	if c.Columns == nil {
		c.Columns = map[string][]float64{}
	}
	if c.OpenTime == nil {
		for i := 0; i < n; i++ {
			c.OpenTime = append(c.OpenTime, start+int64(i)*c.StepMs)
		}
	}
	col := func(name string, v float64) { c.Columns[sym+"_"+name] = append(c.Columns[sym+"_"+name], v) }
	p, open := 100.0, 100.0
	for i := 0; i < n; i++ {
		hi, lo := open, open
		for k := 0; k < 5; k++ {
			p *= math.Exp(rng.NormFloat64() * sigma / math.Sqrt(5))
			hi, lo = math.Max(hi, p), math.Min(lo, p)
		}
		vol := 1e5 + 1e5*rng.Float64()
		col("open", open)
		col("high", hi)
		col("low", lo)
		col("close", p)
		col("quote_volume", vol)
		col("taker_buy_quote_volume", vol*(0.4+0.2*rng.Float64()))
		open = p
	}
}

func chartOf(minutes int64, n int, seed uint64) *candles.Chart {
	c := &candles.Chart{StepMs: minutes * 60_000}
	randomWalk(c, sym, n, seed, 0.002)
	return c
}

func TestAdverseMoveUsesTheNextHCandlesHighsAndLows(t *testing.T) {
	c := &candles.Chart{StepMs: 300_000, OpenTime: make([]int64, 5), Columns: map[string][]float64{
		sym + "_close": {100, 101, 99, 100, 100}, sym + "_high": {100, 103, 100, 101, 150}, sym + "_low": {99, 98, 95, 99, 50}}}
	long, short := AdverseMoves(c, sym, 2)
	approx := func(got, want float64) bool { return math.Abs(got-want) < 1e-12 }
	if !approx(long[0], 0.05) || !approx(short[0], 0.03) { // lowest low of candles 1-2 is 95, highest high 103
		t.Fatal(long[0], short[0])
	}
	if !approx(long[1], 6.0/101) || short[1] != 0 { // candle 4's low of 50 is outside the window
		t.Fatal(long[1], short[1])
	}
	for _, i := range []int{3, 4} {
		if !math.IsNaN(long[i]) || !math.IsNaN(short[i]) {
			t.Fatal("the last h candles have no target")
		}
	}
}

func TestInputsNeverUseLaterCandles(t *testing.T) {
	frame := chartOf(5, 3000, 0)
	randomWalk(frame, bench, 3000, 1, 0.002)
	hourly := chartOf(60, 400, 2)
	rng := rand.New(rand.NewPCG(3, 3))
	for range hourly.OpenTime {
		hourly.Columns[sym+"_funding_rate"] = append(hourly.Columns[sym+"_funding_rate"], 1e-4+5e-5*rng.NormFloat64())
	}
	hourly.Names = append(hourly.Names, sym+"_funding_rate")
	w, _ := regime.ByName("1h")
	inputs := func(c, h *candles.Chart) Frame {
		reg := regime.Compute(c, sym, w, regime.DefaultSpec)
		return Inputs(c, sym, w, bench, h, &reg)
	}
	before := inputs(frame, hourly)

	cut := 2500
	changed := &candles.Chart{StepMs: frame.StepMs, OpenTime: frame.OpenTime, Columns: map[string][]float64{}}
	for name, col := range frame.Columns {
		cp := append([]float64(nil), col...)
		for i := cut + 1; i < len(cp); i++ {
			cp[i] *= 1.7
		}
		changed.Columns[name] = cp
	}
	hourlyChanged := &candles.Chart{StepMs: hourly.StepMs, OpenTime: hourly.OpenTime, Columns: map[string][]float64{}}
	for name, col := range hourly.Columns {
		hourlyChanged.Columns[name] = append([]float64(nil), col...)
	}
	cutClose := frame.OpenTime[cut] + 5*60_000
	for i, open := range hourly.OpenTime {
		if open+3_600_000 > cutClose {
			hourlyChanged.Columns[sym+"_funding_rate"][i] = 0.01
		}
	}
	after := inputs(changed, hourlyChanged)

	for _, name := range []string{"funding", "funding_pct", "market_rv", "taker_1", "to_funding"} {
		if before.Col(name) == nil {
			t.Fatalf("missing input %s", name)
		}
	}
	later := false
	for _, name := range before.Names {
		b, a := before.Col(name), after.Col(name)
		for i := 0; i <= cut; i++ {
			if b[i] != a[i] && !(math.IsNaN(b[i]) && math.IsNaN(a[i])) {
				t.Fatalf("%s[%d] changed: %v → %v", name, i, b[i], a[i])
			}
		}
		for i := cut + 1; i < len(b); i++ {
			if b[i] != a[i] && !(math.IsNaN(b[i]) && math.IsNaN(a[i])) {
				later = true
			}
		}
	}
	if !later {
		t.Fatal("later candles should change later inputs")
	}
	last := len(frame.OpenTime) - 1
	for _, name := range HARColumns {
		if math.IsNaN(before.Col(name)[last]) {
			t.Fatalf("%s is NaN at the end", name)
		}
	}
	if !(before.Col("funding")[last] > -1) {
		t.Fatal("funding at the end")
	}
}

func TestSplitLeavesAGapAndTestsAfterTheTrainingShare(t *testing.T) {
	positions := make([]int, 10_000)
	for i := range positions {
		positions[i] = i
	}
	folds := SplitPositions(positions, DefaultSpec, 100)
	if len(folds) != 4 || folds[0][1][0] != 6000 {
		t.Fatalf("%d folds", len(folds))
	}
	for _, f := range folds {
		train, test := f[0], f[1]
		if train[len(train)-1] >= test[0]-100 || train[0] != 0 {
			t.Fatal("gap")
		}
	}
	h1, _ := regime.ByName("1h")
	h8, _ := regime.ByName("8h")
	if DefaultSpec.Gap(h1) != 12+96 || DefaultSpec.Gap(h8) != 8+8 {
		t.Fatal(DefaultSpec.Gap(h1), DefaultSpec.Gap(h8))
	}
}

func TestBadCaseCoversOneInTenOnARandomWalk(t *testing.T) {
	frame := chartOf(5, 9000, 0)
	w, _ := regime.ByName("30m")
	data := NewWindowData(map[string]*candles.Chart{"5m": frame}, sym, w, "", regime.DefaultSpec)
	spec := DefaultSpec
	spec.Folds = 2
	out, err := WalkForward(data, StopMove(5, 10), LiquidationMove(10, 0.005), spec)
	if err != nil {
		t.Fatal(err)
	}
	for side, c := range out.Sides {
		if c.Coverage < 0.07 || c.Coverage > 0.13 || math.Abs(c.StopHitPredicted-c.StopHitRate) >= 0.05 {
			t.Errorf("%s: %+v", SideNames[side], c)
		}
	}
	model, _ := Fit(data, data.Trainable, spec)
	at, _, today := Latest(model, data, StopMove(5, 10), 0.095, 10)
	if at != frame.OpenTime[len(frame.OpenTime)-1] {
		t.Fatal("latest reading at the last candle")
	}
	if c := today[Short].StopHitChance; c < 0 || c > 1 || math.Abs(today[Long].BadCaseMarginPct-10*today[Long].BadCasePct) > 1e-12 {
		t.Fatalf("%+v", today)
	}
}

func TestExceedChanceReadsTheQuantiles(t *testing.T) {
	alphas := []float64{0.5, 0.9, 0.97}
	q := []float64{1, 2, 3}
	if c := ExceedChance(q, alphas, 2); math.Abs(c-0.1) > 1e-12 {
		t.Fatal(c)
	}
	if c := ExceedChance(q, alphas, 0.5); math.Abs(c-0.75) > 1e-12 {
		t.Fatal(c)
	}
	a, b, c := ExceedChance(q, alphas, 3), ExceedChance(q, alphas, 4), ExceedChance(q, alphas, 10)
	if math.Abs(a-0.03) > 1e-12 || !(a > b && b > c && c > 0) {
		t.Fatal(a, b, c)
	}
}

func TestExceedChanceMatchesTheRowByRowReading(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 4))
	alphas := []float64{0.25, 0.5, 0.75, 0.9, 0.97}
	interp := func(x float64, xs, ys []float64) float64 { // np.interp
		for i := 1; i < len(xs); i++ {
			if x <= xs[i] {
				return ys[i-1] + (ys[i]-ys[i-1])*(x-xs[i-1])/(xs[i]-xs[i-1])
			}
		}
		return ys[len(ys)-1]
	}
	for range 500 {
		q := make([]float64, 5)
		for k := range q {
			q[k] = 0.01 + 2.99*rng.Float64()
		}
		sort.Float64s(q)
		x := -0.5 + 5.5*rng.Float64()
		var want float64
		if x >= q[4] {
			scale := (q[4] - q[3]) / math.Log((1-alphas[3])/(1-alphas[4]))
			want = (1 - alphas[4]) * math.Exp(-(x-q[4])/scale)
		} else if x <= 0 {
			want = 1
		} else {
			want = 1 - interp(x, append([]float64{0}, q...), append([]float64{0}, alphas...))
		}
		if got := ExceedChance(q, alphas, x); math.Abs(got-want) > 1e-12 {
			t.Fatalf("q %v x %g: %g, want %g", q, x, got, want)
		}
	}
	if ExceedChance([]float64{0.1, 0.2, 0.3, 0.4, 0.5}, alphas, 0) != 1 {
		t.Fatal("x = 0")
	}
}

func TestStopIsOnMarginAndScoresAgainstIt(t *testing.T) {
	approx := func(got, want float64) bool { return math.Abs(got-want) < 1e-15 }
	if !approx(StopMove(5, 20), 0.0025) || !approx(StopMove(-10, 3), 0.1/3) || !approx(LiquidationMove(20, 0.004), 0.046) {
		t.Fatal("stop and liquidation moves")
	}
	if *Score(0.426) != 43 || *Score(1.2) != 100 || *Score(-0.1) != 0 || Score(math.NaN()) != nil {
		t.Fatal("score")
	}
}

func TestTiers(t *testing.T) {
	v := func(x float64) *float64 { return &x }
	cases := []struct {
		symbol string
		volume *float64
		want   string
	}{
		{"B-BTC_USDT", nil, "megacap"}, {"B-XRP_USDT", v(1e3), "megacap"}, {"B-1000PEPE_USDT", v(5e9), "shitcoin"},
		{"B-ZK_USDT", v(20e6), "shitcoin"}, {"B-INJ_USDT", v(200e6), "altcoin"}, {"B-INJ_USDT", nil, "altcoin"},
	}
	for _, c := range cases {
		if got := CoinTier(c.symbol, c.volume); got != c.want {
			t.Errorf("CoinTier(%s) = %s, want %s", c.symbol, got, c.want)
		}
	}
}

func TestCoinTrainsScoresAndViews(t *testing.T) {
	charts := map[string]*candles.Chart{}
	for i, iv := range []string{"5m", "15m", "1h"} {
		c := chartOf(candles.MustIntervalMs(iv)/60_000, 3000, uint64(10+i))
		randomWalk(c, bench, 3000, uint64(20+i), 0.002)
		charts[iv] = c
	}
	var steps []string
	coin, err := Train(sym, bench, charts, StopMove(5, 10), LiquidationMove(10, 0.005), DefaultSpec, regime.DefaultSpec,
		func(w string, step, total int) { steps = append(steps, w) })
	if err != nil {
		t.Fatal(err)
	}
	if len(steps) != 8 || steps[0] != "5m" || coin.TrainedAt.IsZero() {
		t.Fatal(steps)
	}
	// 3,000 candles: enough for every window but 8h (96 windows of warm-up are 768 candles of 1h... and 4h's 1,536)
	view := coin.View(10, 0.005, 5)
	if _, ok := view["1h"]; !ok {
		t.Fatalf("windows %v", reflect.ValueOf(view).MapKeys())
	}
	v := view["1h"]
	if v.Long.Score == nil || v.Long.Coverage == nil || len(v.Long.Quantiles.MovesPct) != len(PageAlphas) || v.Long.Reason == "" {
		t.Fatalf("%+v", v.Long)
	}
	if !sort.SliceIsSorted(v.Long.Quantiles.MovesPct, func(i, j int) bool { return v.Long.Quantiles.MovesPct[i] < v.Long.Quantiles.MovesPct[j] }) {
		t.Fatal("quantiles ascend")
	}
	before := v.At
	coin.Score(charts, []string{"15m"})
	if coin.View(10, 0.005, 5)["1h"].At != before {
		t.Fatal("scoring 15m leaves 1h alone")
	}
}

func TestReasons(t *testing.T) {
	calm := regime.Row{State: "choppy", MomentumLabel: "neutral"}
	if r := Reason(Long, 1, math.NaN(), calm, math.NaN()); r != "normal conditions" {
		t.Fatal(r)
	}
	wild := regime.Row{State: "wild", MomentumLabel: "bearish"}
	if r := Reason(Long, 1.34, 95, wild, 1.25); r != "volatility 1.3× normal, market wild, funding crowded long, momentum bearish, 1.2× the short side's bad case" {
		t.Fatal(r)
	}
	if r := Reason(Short, 0.7, 5, regime.Row{MomentumLabel: "bullish"}, 1); r != "volatility 0.7× normal, funding crowded short, momentum bullish" {
		t.Fatal(r)
	}
}
