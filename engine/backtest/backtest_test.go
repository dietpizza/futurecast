package backtest

import (
	"bytes"
	"encoding/csv"
	"math"
	"reflect"
	"strings"
	"testing"

	"deltarisk/engine/candles"
	"deltarisk/engine/setups"
	"deltarisk/engine/wedges"
)

// Round trip on a 1% stop: a win 0.04R (maker both ways), a stop 0.10R (maker, then taker + slippage)
func testSpec() setups.Spec {
	s := setups.DefaultSpec()
	s.LtfMinutes, s.HtfMinutes, s.Targets, s.MakerBps, s.TakerBps, s.SlippageBps = 1, 15, []setups.Target{setups.R(3)}, 2, 5, 3
	return s
}

var t0 = int64(1_767_225_600_000) // 2026-01-01

// history is eight A+/B setups with known results, on 100 one-minute candles.
func history() ([]Candidate, []setups.Outcome, []int64) {
	open := make([]int64, 100)
	for i := range open {
		open[i] = t0 + int64(i)*60_000
	}
	type row struct {
		idx               int
		grade, fill       string
		fillIdx, orderEnd int
		outcome           string
		r                 float64
		end               int
	}
	rows := []row{
		{0, "A+", "filled", 2, -1, "won", 3, 10},          // taken
		{5, "A+", "filled", 6, -1, "won", 3, 9},           // skipped: the first trade is still open
		{12, "A+", "expired", -1, 20, "", math.NaN(), -1}, // an order that never fills: blocks until 20
		{15, "A+", "filled", 16, -1, "won", 3, 18},        // skipped: that order is still waiting
		{25, "B", "filled", 26, -1, "won", 3, 28},         // skipped: wrong grade
		{30, "A+", "filled", 31, -1, "lost", -1, 35},      // taken
		{40, "A+", "filled", 41, -1, "lost", -1, 45},      // taken
		{50, "A+", "pending", -1, -1, "", math.NaN(), -1}, // left out: not resolved yet
	}
	var cands []Candidate
	var outs []setups.Outcome
	for _, r := range rows {
		cands = append(cands, Candidate{Idx: r.idx, Side: 1, Grade: r.grade, Entry: 100, Stop: 99, Risk: 0.01, MeasuredR: math.NaN()})
		win := math.NaN()
		if r.outcome == "won" {
			win = 1
		} else if r.outcome != "" {
			win = 0
		}
		outs = append(outs, setups.Outcome{Fill: r.fill, FillIdx: r.fillIdx, OrderEnd: r.orderEnd,
			Targets: map[string]setups.TradeResult{"3": {Outcome: r.outcome, Win: win, R: r.r, End: r.end}}})
	}
	return cands, outs, open
}

func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestOneSetupAtATimeWithFixedRisk(t *testing.T) {
	cands, outs, open := history()
	res := Run(cands, outs, open, testSpec(), Config{Target: setups.R(3), Grades: []string{"A+"}, RiskPct: 1})
	s, st, trades := res.Summary, res.Summary.Stats, res.Trades
	if s.Orders != 4 || s.Trades != 3 || *st.FillRate != 0.75 {
		t.Fatalf("%+v", s)
	}
	var at []string
	var net []float64
	for _, tr := range trades {
		at = append(at, tr.SetupAt[11:16])
		net = append(net, Round(tr.NetR, 4))
	}
	if !reflect.DeepEqual(at, []string{"00:01", "00:31", "00:41"}) || !reflect.DeepEqual(net, []float64{2.96, -1.1, -1.1}) {
		t.Fatal(at, net) // setup candles 0, 30, 40 (close times)
	}
	if st.Wins != 1 || st.Losses != 2 || st.WinRate != Round(1.0/3, 4) || !approx(st.TotalNetR, 0.76, 1e-9) {
		t.Fatalf("%+v", st)
	}
	if !approx(*st.ProfitFactor, 2.96/2.2, 1e-3) || !approx(st.BreakEvenWinRate, 1.1/(3-0.04+1.1), 1e-4) {
		t.Fatalf("%+v", st)
	}
	// 1% risk per trade, compounding: +2.96%, then -1.1% twice
	if !approx(st.TotalReturnPct, (1.0296*0.989*0.989-1)*100, 0.01) || !approx(st.MaxDrawdownPct, (1-0.989*0.989)*100, 0.01) {
		t.Fatalf("%+v", st)
	}
	if !approx(st.MaxDrawdownR, 2.2, 1e-9) || st.LongestLosingStreak != 2 || !approx(trades[0].HoldHours, 9.0/60, 1e-12) {
		t.Fatalf("%+v", st) // fill candle 2 through candle 10
	}
	for i, want := range []float64{102.96, 102.96 * 0.989, 102.96 * 0.989 * 0.989} {
		if !approx(trades[i].EquityAfter, want, 1e-9) {
			t.Fatal(trades[i].EquityAfter, want)
		}
	}
	if res.Sides["long"].Trades != 3 || len(res.Shapes) != 0 {
		t.Fatalf("%+v", res.Sides)
	}
	if want := []Month{{"2026-01", 3, Round(1.0/3, 4), 0.76, Round((1.0296*0.989*0.989-1)*100, 2)}}; !reflect.DeepEqual(res.Monthly, want) {
		t.Fatalf("%+v", res.Monthly)
	}

	// Taking every grade adds the B setup, which is free by then
	if every := Run(cands, outs, open, testSpec(), Config{Target: setups.R(3), Grades: []string{"A+", "B"}, RiskPct: 1}); every.Summary.Trades != 4 {
		t.Fatal(every.Summary.Trades)
	}

	report := FormatReport("B-TEST_USDT", res)
	for _, want := range []string{"Win rate          33.3%", "Only 3 trades",
		"fees 0.02% maker / 0.05% taker, 0.03% stop slippage (0.04% round trip on a win, 0.1% on a stop)"} {
		if !strings.Contains(report, want) {
			t.Errorf("report lacks %q:\n%s", want, report)
		}
	}
}

func TestEachTradePaysTheCostsOfHowItEnded(t *testing.T) {
	cands, outs, open := history()
	// The last taken trade times out at +0.5R instead: closed with a market order, so taker but no slippage
	outs[6].Targets["3"] = setups.TradeResult{Outcome: "timeout", Win: 0, R: 0.5, End: 45}
	res := Run(cands, outs, open, testSpec(), Config{Target: setups.R(3), Grades: []string{"A+"}, RiskPct: 1})
	var outcomes []string
	var costs, net []float64
	for _, tr := range res.Trades {
		outcomes = append(outcomes, tr.Outcome)
		costs = append(costs, Round(tr.CostR, 4))
		net = append(net, Round(tr.NetR, 4))
	}
	if !reflect.DeepEqual(outcomes, []string{"won", "lost", "timeout"}) || !reflect.DeepEqual(costs, []float64{0.04, 0.1, 0.07}) ||
		!reflect.DeepEqual(net, []float64{2.96, -1.1, 0.43}) {
		t.Fatal(outcomes, costs, net)
	}
	st := res.Summary.Stats
	// Break-even still counts every non-win as a stop
	if !approx(st.AvgCostR, 0.07, 1e-4) || !approx(st.BreakEvenWinRate, 1.1/4.06, 1e-4) {
		t.Fatalf("%+v", st)
	}
}

func TestLeverageCapShrinksTheRiskOfTightStops(t *testing.T) {
	cands, outs, open := history()
	cands[5].Risk = 0.001 // the second taken trade has a 0.1% stop: 1% risk needs 10x
	free := testSpec()
	free.MakerBps, free.TakerBps, free.SlippageBps = 0, 0, 0
	res := Run(cands, outs, open, free, Config{Target: setups.R(3), Grades: []string{"A+"}, RiskPct: 1, MaxLeverage: 5})
	var lev, risk, net []float64
	for _, tr := range res.Trades {
		lev, risk, net = append(lev, tr.Leverage), append(risk, Round(tr.AccountRiskPct, 9)), append(net, tr.NetR)
	}
	if !reflect.DeepEqual(lev, []float64{1, 5, 1}) || !reflect.DeepEqual(risk, []float64{1, 0.5, 1}) || !reflect.DeepEqual(net, []float64{3, -1, -1}) {
		t.Fatal(lev, risk, net) // results in R are unchanged ...
	}
	st := res.Summary.Stats
	if st.CappedTrades != 1 || !approx(st.TotalReturnPct, (1.03*0.995*0.99-1)*100, 0.01) { // ... the capped loss costs half
		t.Fatalf("%+v", st)
	}
	if r := FormatReport("B-TEST_USDT", res); !strings.Contains(r, "1 of 3 trades risked less than 1%: their stop was too tight for 5x") {
		t.Fatal(r)
	}
	uncapped := Run(cands, outs, open, free, Config{Target: setups.R(3), Grades: []string{"A+"}, RiskPct: 1}).Summary.Stats
	if uncapped.CappedTrades != 0 || !approx(uncapped.TotalReturnPct, (1.03*0.99*0.99-1)*100, 0.01) {
		t.Fatalf("%+v", uncapped)
	}
}

func TestNoTradesIsReportedNotInvented(t *testing.T) {
	_, _, open := history()
	res := Run(nil, nil, open, testSpec(), DefaultConfig())
	if res.Summary.Trades != 0 || res.Monthly != nil || res.Summary.Stats != nil {
		t.Fatalf("%+v", res.Summary)
	}
	if !strings.Contains(FormatReport("B-TEST_USDT", res), "No filled trades.") {
		t.Fatal("report")
	}
}

func TestWedgeBacktestAndReport(t *testing.T) {
	// The wedge of the wedges tests: a falling wedge broken on candle 23, retested and triggered on 24
	turns := [][2]float64{{0, 110}, {2, 119}, {8, 99.2}, {14, 113}, {20, 98}, {22, 102}, {23, 110}}
	interp := func(x float64) float64 {
		for i := 1; i < len(turns); i++ {
			if x <= turns[i][0] {
				return turns[i-1][1] + (turns[i][1]-turns[i-1][1])*(x-turns[i-1][0])/(turns[i][0]-turns[i-1][0])
			}
		}
		return turns[len(turns)-1][1]
	}
	var close []float64
	for i := 0; i < 24; i++ {
		close = append(close, interp(float64(i)))
	}
	close = append(close, 110.4, 113.0, 116.0, 119.0, 112.0, 105.0, 103.0)
	var b candles.Bars
	for i, c := range close {
		open := close[max(i-1, 0)]
		h, l, v := math.Max(open, c), math.Min(open, c), 100.0
		if i == 24 {
			h, l = 110.6, 108.1
		}
		if i == 23 {
			v = 400
		}
		b.Append(candles.Bar{Time: 1_772_409_600_000 + int64(i)*3_600_000, Open: open, High: h, Low: l, Close: c, Volume: v})
	}
	spec := wedges.DefaultSpec()
	spec.SwingLen, spec.Window, spec.ApexMin, spec.ApexMax, spec.ATRLen, spec.VolumeWindow = 2, 40, 3, 40, 3, 5
	spec.RetestWindow, spec.Targets, spec.BreakevenAt, spec.MaxHold = 3, []setups.Target{setups.R(2), setups.Measured}, nil, 30
	spec.MakerBps, spec.TakerBps, spec.SlippageBps = 0, 0, 0

	won := Wedge(&b, spec, Config{Target: setups.R(2), Grades: []string{"A+"}, RiskPct: 1, MaxLeverage: 5})
	s := won.Summary
	if s.Strategy != "wedge" || s.Funnel.Triggered != 1 || s.Trades != 1 || s.Stats.Wins != 1 || s.Stats.TotalNetR != 2 {
		t.Fatalf("%+v %+v", s, s.Stats)
	}
	tr := won.Trades[0]
	if tr.Shape != "falling wedge" || tr.HoldHours != 3 || !approx(tr.Leverage, 0.01/tr.StopPct*100, 1e-12) {
		t.Fatalf("%+v", tr) // entered at 24's close, won on 27; a 3.8% stop needs no leverage
	}
	report := FormatReport("B-TEST_USDT", won)
	if !strings.HasPrefix(report, "B-TEST_USDT · wedge breakouts on 1h · A+ setups · 2R target · 1% risk per trade, "+
		"at most 5x leverage\nmarket entries, stop never moved") ||
		!strings.Contains(report, "1 wedges, 1 broke out: 0 closed back inside, 0 found no retest in time, 1 triggered") ||
		!strings.Contains(report, "falling wedge") {
		t.Fatal(report)
	}

	measured := Wedge(&b, spec, Config{Target: setups.Measured, Grades: []string{"A+"}, RiskPct: 1}).Summary
	if measured.Stats.Losses != 1 || measured.Stats.TotalNetR != -1 || measured.Stats.MedianTargetR != Round(19.2/(110.4-tr.Stop), 2) {
		t.Fatalf("%+v", measured.Stats)
	}

	var buf bytes.Buffer
	if err := WriteCSV(&buf, won.Trades); err != nil {
		t.Fatal(err)
	}
	rows, err := csv.NewReader(&buf).ReadAll()
	if err != nil || len(rows) != 2 || rows[0][2] != "shape" || rows[1][2] != "falling wedge" || rows[1][len(rows[1])-6] != "won" {
		t.Fatal(rows, err)
	}
}

func TestPythonFormatting(t *testing.T) {
	for v, want := range map[float64]string{1: "1.0", 0.0295: "0.0295", 102.96: "102.96", 1e-5: "1e-05", -3: "-3.0", 1234567.5: "1234567.5"} {
		if got := pyRepr(v); got != want {
			t.Errorf("pyRepr(%v) = %s, want %s", v, got, want)
		}
	}
	for v, want := range map[float64]string{0.0295: "0.0295", 0.029500000000000002: "0.0295", 3: "3", 2.5: "2.5", 0.177: "0.177"} {
		if got := setups.PyG(v); got != want {
			t.Errorf("PyG(%v) = %s, want %s", v, got, want)
		}
	}
	if Round(0.125, 2) != 0.12 || Round(2.675, 2) != 2.67 || Round(1.0/3, 4) != 0.3333 {
		t.Error("Round should be Python's round")
	}
	if ISO(t0) != "2026-01-01T00:00:00+00:00" {
		t.Error(ISO(t0))
	}
}
