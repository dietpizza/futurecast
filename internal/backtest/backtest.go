// Package backtest replays the setup rules on historical candles (rules only, no models).
//
// Every setup the rules find (fair-value-gap setups or wedge breakouts) is replayed with what its
// orders really did, like a trader running one account:
//
//   - Only setups of the chosen grades are taken (A+ by default), one at a time: a setup is
//     skipped while an earlier order is still waiting or its trade is still open.
//   - Each filled trade risks RiskPct of the account, so a result of r R (after its fees and
//     slippage, also counted in R) moves equity by r × RiskPct percent, compounding. With
//     MaxLeverage the position is at most that multiple of the account, so a trade whose stop
//     is too tight for it risks less. Costs follow how the orders are placed (the rules'
//     RoundTrip).
//   - Orders that never fill cost nothing. Setups whose outcome the data does not cover yet are
//     left out.
//
// Reports win rate, average and total R, profit factor, drawdown, losing streaks, monthly,
// per-side and (for wedges) per-shape results, and the trade list.
package backtest

import (
	"math"
	"sort"
	"strconv"
	"time"

	"deltarisk/internal/candles"
	"deltarisk/internal/series"
	"deltarisk/internal/setups"
	"deltarisk/internal/wedges"
)

// Rules is what the backtest needs from a strategy's spec.
type Rules interface {
	Strategy() string
	MarketEntry() bool
	EntryMinutes() int
	EntryInterval() string
	ContextInterval() string // "" for one chart
	BreakevenR() *float64
	RoundTrip(outcome string) float64
	Costs() setups.Costs
}

// Candidate is a setup as the backtest sees it.
type Candidate struct {
	Idx, Side         int
	Grade, Shape      string // Shape "" for fair-value-gap setups
	Entry, Stop, Risk float64
	MeasuredR         float64
}

// Config is how the account trades.
type Config struct {
	Target      setups.Target // one of the spec's targets
	Grades      []string
	RiskPct     float64 // account % risked per trade
	MaxLeverage float64 // position value per account at most this (0: no cap); tighter stops then risk less
}

// DefaultConfig is 3R, A+ only, 1% risk, no cap.
func DefaultConfig() Config {
	return Config{Target: setups.R(3), Grades: []string{"A+"}, RiskPct: 1}
}

// Trade is one taken trade.
type Trade struct {
	SetupAt        string  `json:"setup_at"`
	Side           string  `json:"side"`
	Shape          string  `json:"shape,omitempty"`
	Grade          string  `json:"grade"`
	Entry          float64 `json:"entry"`
	Stop           float64 `json:"stop"`
	TakeProfit     float64 `json:"take_profit"`
	TargetR        float64 `json:"target_r"`
	StopPct        float64 `json:"stop_pct"`
	Leverage       float64 `json:"leverage"`
	AccountRiskPct float64 `json:"account_risk_pct"`
	FilledAt       string  `json:"filled_at"`
	ClosedAt       string  `json:"closed_at"`
	Outcome        string  `json:"outcome"`
	GrossR         float64 `json:"gross_r"`
	CostR          float64 `json:"cost_r"`
	NetR           float64 `json:"net_r"`
	HoldHours      float64 `json:"hold_hours"`
	EquityAfter    float64 `json:"equity_after"`
}

// Summary is a backtest's headline numbers; Stats is nil without trades.
type Summary struct {
	Strategy        string         `json:"strategy"`
	EntryInterval   string         `json:"entry_interval"`
	ContextInterval *string        `json:"context_interval"`
	TargetR         setups.Target  `json:"target_r"`
	BreakevenR      *float64       `json:"breakeven_r"`
	Grades          []string       `json:"grades"`
	RiskPct         float64        `json:"risk_pct"`
	MaxLeverage     *float64       `json:"max_leverage"`
	Costs           setups.Costs   `json:"costs"`
	Start           *string        `json:"start"`
	End             *string        `json:"end"`
	Days            float64        `json:"days"`
	Setups          int            `json:"setups"`
	SetupsInGrades  int            `json:"setups_in_grades"`
	Orders          int            `json:"orders"`
	Trades          int            `json:"trades"`
	Funnel          *wedges.Funnel `json:"funnel,omitempty"`
	Stats           *Stats         `json:"-"`
}

// Stats are the results of the trades.
type Stats struct {
	FillRate            *float64 `json:"fill_rate"`
	Wins                int      `json:"wins"`
	Losses              int      `json:"losses"`
	Breakevens          int      `json:"breakevens"`
	Timeouts            int      `json:"timeouts"`
	WinRate             float64  `json:"win_rate"`
	BreakEvenWinRate    float64  `json:"break_even_win_rate"`
	MedianTargetR       float64  `json:"median_target_r"`
	AvgGrossR           float64  `json:"avg_gross_r"`
	AvgCostR            float64  `json:"avg_cost_r"`
	AvgNetR             float64  `json:"avg_net_r"`
	TotalNetR           float64  `json:"total_net_r"`
	ProfitFactor        *float64 `json:"profit_factor"`
	TotalReturnPct      float64  `json:"total_return_pct"`
	MaxDrawdownPct      float64  `json:"max_drawdown_pct"`
	MaxDrawdownR        float64  `json:"max_drawdown_r"`
	LongestLosingStreak int      `json:"longest_losing_streak"`
	AvgHoldHours        float64  `json:"avg_hold_hours"`
	TradesPerMonth      float64  `json:"trades_per_month"`
	MedianStopPct       float64  `json:"median_stop_pct"`
	CappedTrades        int      `json:"capped_trades"`
}

// Month is one calendar month's trades (by close).
type Month struct {
	Month     string  `json:"month"`
	Trades    int     `json:"trades"`
	WinRate   float64 `json:"win_rate"`
	NetR      float64 `json:"net_r"`
	ReturnPct float64 `json:"return_pct"`
}

// Group is the trades of one side or shape.
type Group struct {
	Trades    int     `json:"trades"`
	WinRate   float64 `json:"win_rate"`
	AvgNetR   float64 `json:"avg_net_r"`
	TotalNetR float64 `json:"total_net_r"`
}

// Result is one backtest.
type Result struct {
	Summary Summary
	Trades  []Trade
	Monthly []Month
	Sides   map[string]Group
	Shapes  map[string]Group
}

// Round is Python's round(x, digits): correctly rounded, ties to even.
func Round(x float64, digits int) float64 {
	v, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'f', digits, 64), 64)
	return v
}

// ISO is a time in ms as pandas' isoformat ("2026-09-08T00:15:00+00:00").
func ISO(ms int64) string { return time.UnixMilli(ms).UTC().Format("2006-01-02T15:04:05+00:00") }

func longestRun(hit []bool) int {
	best, run := 0, 0
	for _, h := range hit {
		if h {
			run++
		} else {
			run = 0
		}
		best = max(best, run)
	}
	return best
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// TakeTrades are the trades the rules would have taken, oldest first, and how many orders were
// placed. openTime is the entry chart the setups and outcomes were computed on.
func TakeTrades(cands []Candidate, outcomes []setups.Outcome, openTime []int64, rules Rules, cfg Config) ([]Trade, int) {
	key := cfg.Target.Key()
	ends := setups.CloseTimes(openTime, rules.EntryMinutes())
	var trades []Trade
	freeFrom, orders := -1, 0
	for i, s := range cands {
		o := outcomes[i]
		tr := o.Targets[key]
		filled := o.Fill == "filled"
		resolved := o.Fill != "pending" && !(filled && tr.Outcome == "open")
		if !contains(cfg.Grades, s.Grade) || !resolved || s.Idx <= freeFrom {
			continue
		}
		orders++
		freeFrom = o.OrderEnd
		if filled {
			freeFrom = tr.End
		}
		if !filled {
			continue
		}
		d := float64(s.Side)
		R := math.Abs(s.Entry - s.Stop)
		T := cfg.Target.Multiple
		if cfg.Target.Measured {
			T = s.MeasuredR
		}
		costR := rules.RoundTrip(tr.Outcome) / s.Risk
		leverage, accountRisk := cfg.RiskPct/100.0/s.Risk, cfg.RiskPct
		if cfg.MaxLeverage > 0 && leverage > cfg.MaxLeverage {
			leverage, accountRisk = cfg.MaxLeverage, cfg.MaxLeverage*s.Risk*100.0
		}
		held := tr.End - o.FillIdx
		if !rules.MarketEntry() {
			held++
		}
		side := "long"
		if s.Side < 0 {
			side = "short"
		}
		trades = append(trades, Trade{
			SetupAt: ISO(ends[s.Idx]), Side: side, Shape: s.Shape, Grade: s.Grade, Entry: s.Entry, Stop: s.Stop,
			TakeProfit: s.Entry + float64(d*T*R), TargetR: T, StopPct: s.Risk * 100, Leverage: leverage,
			AccountRiskPct: accountRisk, FilledAt: ISO(ends[o.FillIdx]), ClosedAt: ISO(ends[tr.End]), Outcome: tr.Outcome,
			GrossR: tr.R, CostR: costR, NetR: tr.R - costR, HoldHours: float64(held) * float64(rules.EntryMinutes()) / 60.0,
		})
	}
	return trades, orders
}

func groupBy(trades []Trade, key func(Trade) string) map[string]Group {
	byKey := map[string][]Trade{}
	for _, t := range trades {
		byKey[key(t)] = append(byKey[key(t)], t)
	}
	out := map[string]Group{}
	for name, g := range byKey {
		won, net := make([]float64, len(g)), make([]float64, len(g))
		for i, t := range g {
			if t.Outcome == "won" {
				won[i] = 1
			}
			net[i] = t.NetR
		}
		out[name] = Group{Trades: len(g), WinRate: Round(series.Mean(won), 4), AvgNetR: Round(series.Mean(net), 4),
			TotalNetR: Round(series.Sum(net), 2)}
	}
	return out
}

// Run backtests found setups and their outcomes on the entry chart's open times.
func Run(cands []Candidate, outcomes []setups.Outcome, openTime []int64, rules Rules, cfg Config) Result {
	trades, orders := TakeTrades(cands, outcomes, openTime, rules, cfg)
	ends := setups.CloseTimes(openTime, rules.EntryMinutes())
	days := 0.0
	if len(ends) > 0 {
		days = float64(ends[len(ends)-1]-ends[0]) / 86_400e3
	}
	s := Summary{Strategy: rules.Strategy(), EntryInterval: rules.EntryInterval(), TargetR: cfg.Target,
		BreakevenR: rules.BreakevenR(), Grades: cfg.Grades, RiskPct: cfg.RiskPct, Costs: rules.Costs(),
		Days: Round(days, 1), Setups: len(cands), Orders: orders, Trades: len(trades)}
	if ctx := rules.ContextInterval(); ctx != "" {
		s.ContextInterval = &ctx
	}
	if cfg.MaxLeverage > 0 {
		m := cfg.MaxLeverage
		s.MaxLeverage = &m
	}
	if len(ends) > 0 {
		start, end := ISO(ends[0]), ISO(ends[len(ends)-1])
		s.Start, s.End = &start, &end
	}
	for _, c := range cands {
		if contains(cfg.Grades, c.Grade) {
			s.SetupsInGrades++
		}
	}
	res := Result{Summary: s, Trades: trades, Sides: map[string]Group{}, Shapes: map[string]Group{}}
	if len(trades) == 0 {
		return res
	}

	n := len(trades)
	net, won, gross, cost, hold := make([]float64, n), make([]float64, n), make([]float64, n), make([]float64, n), make([]float64, n)
	targets, stops, breakEven := make([]float64, n), make([]float64, n), make([]float64, n)
	equity := make([]float64, n)
	st := &Stats{}
	var gains, losses []float64
	losing := make([]bool, n)
	for i, t := range trades {
		net[i], gross[i], cost[i], hold[i], targets[i], stops[i] = t.NetR, t.GrossR, t.CostR, t.HoldHours, t.TargetR, t.StopPct
		switch t.Outcome {
		case "won":
			won[i] = 1
			st.Wins++
		case "lost":
			st.Losses++
		case "breakeven":
			st.Breakevens++
		case "timeout":
			st.Timeouts++
		}
		risk := t.StopPct / 100.0
		// a win nets target minus its costs, a non-win counts as a stop: -1 minus the stop's costs
		breakEven[i] = setups.BreakEven(t.TargetR, 1.0, rules.RoundTrip("won")/risk, rules.RoundTrip("lost")/risk)
		if t.NetR > 0 {
			gains = append(gains, t.NetR)
		} else if t.NetR < 0 {
			losses = append(losses, -t.NetR)
		}
		losing[i] = t.NetR < 0
		if t.AccountRiskPct < cfg.RiskPct*(1-1e-9) {
			st.CappedTrades++
		}
	}
	// equity = 100 * cumprod(1 + account_risk * net), drawdowns against the running peak
	prod, cumR := 1.0, 0.0
	peak, peakR, maxDD, maxDDR := 100.0, 0.0, 0.0, 0.0
	for i, t := range trades {
		prod *= 1.0 + float64(t.AccountRiskPct/100.0*t.NetR) // float64: no FMA, like numpy
		equity[i] = 100.0 * prod
		trades[i].EquityAfter = equity[i]
		peak = math.Max(peak, equity[i])
		maxDD = math.Max(maxDD, 1.0-equity[i]/peak)
		cumR += t.NetR
		peakR = math.Max(peakR, cumR)
		maxDDR = math.Max(maxDDR, peakR-cumR)
	}
	res.Trades = trades
	if orders > 0 {
		fr := Round(float64(n)/float64(orders), 4)
		st.FillRate = &fr
	}
	st.WinRate = Round(series.Mean(won), 4)
	st.BreakEvenWinRate = Round(series.Mean(breakEven), 4)
	st.MedianTargetR = Round(series.Median(targets), 2)
	st.AvgGrossR = Round(series.Mean(gross), 4)
	st.AvgCostR = Round(series.Mean(cost), 4)
	st.AvgNetR = Round(series.Mean(net), 4)
	st.TotalNetR = Round(series.Sum(net), 2)
	if l := series.Sum(losses); l > 0 {
		pf := Round(series.Sum(gains)/l, 3)
		st.ProfitFactor = &pf
	}
	st.TotalReturnPct = Round(equity[n-1]-100.0, 2)
	st.MaxDrawdownPct = Round(maxDD*100, 2)
	st.MaxDrawdownR = Round(maxDDR, 2)
	st.LongestLosingStreak = longestRun(losing)
	st.AvgHoldHours = Round(series.Mean(hold), 2)
	st.TradesPerMonth = Round(float64(n)/math.Max(days/30.44, 1e-9), 2)
	st.MedianStopPct = Round(series.Median(stops), 4)
	res.Summary.Stats = st

	byMonth := map[string][]Trade{}
	for _, t := range trades {
		byMonth[t.ClosedAt[:7]] = append(byMonth[t.ClosedAt[:7]], t)
	}
	months := make([]string, 0, len(byMonth))
	for m := range byMonth {
		months = append(months, m)
	}
	sort.Strings(months)
	for _, m := range months {
		g := byMonth[m]
		r, w := make([]float64, len(g)), make([]float64, len(g))
		ret := 1.0
		for i, t := range g {
			r[i] = t.NetR
			if t.Outcome == "won" {
				w[i] = 1
			}
			ret *= 1.0 + float64(t.AccountRiskPct/100.0*t.NetR)
		}
		res.Monthly = append(res.Monthly, Month{Month: m, Trades: len(g), WinRate: Round(series.Mean(w), 4),
			NetR: Round(series.Sum(r), 2), ReturnPct: Round((ret-1.0)*100, 2)})
	}
	res.Sides = groupBy(trades, func(t Trade) string { return t.Side })
	if trades[0].Shape != "" {
		res.Shapes = groupBy(trades, func(t Trade) string { return t.Shape })
	}
	return res
}

// FVGCandidates turns fair-value-gap setups into candidates.
func FVGCandidates(found []setups.Setup) []Candidate {
	out := make([]Candidate, len(found))
	for i, s := range found {
		out[i] = Candidate{Idx: s.Idx, Side: s.Side, Grade: s.Grade, Entry: s.Entry, Stop: s.Stop, Risk: s.Risk, MeasuredR: math.NaN()}
	}
	return out
}

// WedgeCandidates turns wedge setups into candidates.
func WedgeCandidates(found []wedges.Setup) []Candidate {
	out := make([]Candidate, len(found))
	for i, s := range found {
		out[i] = Candidate{Idx: s.Idx, Side: s.Side, Grade: s.Grade, Shape: s.Shape, Entry: s.Entry, Stop: s.Stop,
			Risk: s.Risk, MeasuredR: s.MeasuredR}
	}
	return out
}

// FVG finds the setups in raw candles (entry and context charts) and backtests them.
func FVG(entry, context *candles.Bars, spec setups.Spec, cfg Config) Result {
	found := setups.FindSetups(entry, context, spec)
	return Run(FVGCandidates(found), setups.Outcomes(found, entry, spec), entry.Time, spec, cfg)
}

// Wedge finds the wedge setups in raw candles and backtests them; the summary carries the funnel.
func Wedge(bars *candles.Bars, spec wedges.Spec, cfg Config) Result {
	found, funnel := wedges.Find(bars, spec)
	res := Run(WedgeCandidates(found), wedges.Outcomes(found, bars, spec), bars.Time, spec, cfg)
	res.Summary.Funnel = &funnel
	return res
}
