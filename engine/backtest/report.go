package backtest

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"

	"deltarisk/engine/setups"
)

func fmtNum(v *float64, suffix string, digits int, signed bool) string {
	if v == nil || math.IsNaN(*v) || math.IsInf(*v, 0) {
		return "—"
	}
	if signed {
		return fmt.Sprintf("%+.*f%s", digits, *v, suffix)
	}
	return fmt.Sprintf("%.*f%s", digits, *v, suffix)
}

func f(v float64) *float64 { return &v }

// FeesText is the costs in words: the per-leg rates and what a win and a stop cost round trip.
func FeesText(c setups.Costs) string {
	g := setups.PyG
	return fmt.Sprintf("fees %s%% maker / %s%% taker, %s%% stop slippage (%s%% round trip on a win, %s%% on a stop)",
		g(c.MakerPct), g(c.TakerPct), g(c.SlippagePct), g(c.WonPct), g(c.LostPct))
}

func targetText(t setups.Target) string {
	if t.Measured {
		return "measured-move"
	}
	return setups.PyG(t.Multiple) + "R"
}

// FormatReport is a plain-text report of one backtest.
func FormatReport(symbol string, r Result) string {
	s := r.Summary
	charts := s.EntryInterval + " entries in " + deref(s.ContextInterval) + " context"
	if s.Strategy == "wedge" {
		charts = "wedge breakouts on " + s.EntryInterval
	}
	head := fmt.Sprintf("%s · %s · %s setups · %s target · %s%% risk per trade", symbol, charts,
		strings.Join(s.Grades, "/"), targetText(s.TargetR), setups.PyG(s.RiskPct))
	if s.MaxLeverage != nil {
		head += fmt.Sprintf(", at most %sx leverage", setups.PyG(*s.MaxLeverage))
	}
	lines := []string{head}
	if s.Strategy == "wedge" {
		be := "stop never moved"
		if s.BreakevenR != nil {
			be = "stop to break-even at " + setups.PyG(*s.BreakevenR) + "R"
		}
		lines = append(lines, "market entries, "+be)
	}
	taken := fmt.Sprintf("%d orders placed one at a time, %d filled", s.Orders, s.Trades)
	if s.Strategy == "wedge" {
		taken = fmt.Sprintf("%d taken one at a time", s.Trades)
	}
	day := func(v *string) string {
		if v == nil {
			return "?"
		}
		return (*v)[:10]
	}
	lines = append(lines, FeesText(s.Costs), fmt.Sprintf("%s to %s (%.0f days): %d setups found, %d in grade, %s",
		day(s.Start), day(s.End), s.Days, s.Setups, s.SetupsInGrades, taken))
	if fu := s.Funnel; fu != nil {
		lines = append(lines, fmt.Sprintf("%d wedges, %d broke out: %d closed back inside, %d found no retest in time, %d triggered",
			fu.Wedges, fu.Breakouts, fu.Failed, fu.Expired, fu.Triggered))
	}
	st := s.Stats
	if s.Trades == 0 || st == nil {
		lines = append(lines, "No filled trades.")
		return strings.Join(lines, "\n")
	}
	ended := fmt.Sprintf("%d won, %d stopped", st.Wins, st.Losses)
	if st.Breakevens > 0 {
		ended += fmt.Sprintf(", %d out at break-even", st.Breakevens)
	}
	ended += fmt.Sprintf(", %d timed out", st.Timeouts)
	winLine := fmt.Sprintf("  Win rate          %.1f%%  (%s)   needs %.1f%% to break even after fees", st.WinRate*100, ended, st.BreakEvenWinRate*100)
	if st.Breakevens > 0 {
		winLine += " if every non-win were a full stop"
	}
	pace := fmt.Sprintf("  Pace              %s, held %s on average", fmtNum(f(st.TradesPerMonth), " trades a month", 1, false),
		fmtNum(f(st.AvgHoldHours), " h", 1, false))
	if s.Strategy != "wedge" && st.FillRate != nil {
		pace += fmt.Sprintf(", %.0f%% of orders filled", *st.FillRate*100)
	}
	lines = append(lines, "", winLine,
		fmt.Sprintf("  Average trade     %s after fees  (%s before; fees cost %s a trade, median stop %s)",
			fmtNum(f(st.AvgNetR), "R", 2, true), fmtNum(f(st.AvgGrossR), "R", 2, true), fmtNum(f(st.AvgCostR), "R", 2, false),
			fmtNum(f(st.MedianStopPct), "%", 2, false)),
		fmt.Sprintf("  Total             %s  =  %s on the account, compounding", fmtNum(f(st.TotalNetR), "R", 2, true),
			fmtNum(f(st.TotalReturnPct), "%", 2, true)),
		fmt.Sprintf("  Profit factor     %s", fmtNum(st.ProfitFactor, "", 2, false)),
		fmt.Sprintf("  Max drawdown      %s  (%s), longest losing streak %d", fmtNum(f(st.MaxDrawdownPct), "%", 2, false),
			fmtNum(f(st.MaxDrawdownR), "R", 2, false), st.LongestLosingStreak),
		pace)
	if s.TargetR.Measured {
		lines = append(lines, fmt.Sprintf("  Target            the wedge's height, %s for the median trade", fmtNum(f(st.MedianTargetR), "R", 2, false)))
	}
	if st.CappedTrades > 0 {
		lines = append(lines, fmt.Sprintf("  Leverage          %d of %d trades risked less than %s%%: their stop was too tight for %sx",
			st.CappedTrades, s.Trades, setups.PyG(s.RiskPct), setups.PyG(deref64(s.MaxLeverage))))
	}
	if s.Trades < 100 {
		lines = append(lines, fmt.Sprintf("  Only %d trades: too few to trust the rates above (aim for 100+).", s.Trades))
	}
	for _, block := range []struct {
		title  string
		groups map[string]Group
	}{{"Side", r.Sides}, {"Shape", r.Shapes}} {
		if len(block.groups) == 0 {
			continue
		}
		names := make([]string, 0, len(block.groups))
		width := 6
		for name := range block.groups {
			names = append(names, name)
			width = max(width, len(name))
		}
		sort.Strings(names)
		lines = append(lines, "", fmt.Sprintf("  %-*s  trades  win rate  avg R    total R", width, block.title))
		for _, name := range names {
			v := block.groups[name]
			lines = append(lines, fmt.Sprintf("  %-*s  %6d  %7.1f%%  %+6.2f  %+8.2f", width, name, v.Trades, v.WinRate*100, v.AvgNetR, v.TotalNetR))
		}
	}
	if len(r.Monthly) > 0 {
		lines = append(lines, "", "  Month    trades  win rate   net R   account")
		for _, m := range r.Monthly {
			lines = append(lines, fmt.Sprintf("  %s  %6d  %7.1f%%  %+6.2f  %+7.2f%%", m.Month, m.Trades, m.WinRate*100, m.NetR, m.ReturnPct))
		}
	}
	return strings.Join(lines, "\n")
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func deref64(v *float64) float64 {
	if v == nil {
		return 0
	}
	return *v
}

// pyRepr formats a float like Python's repr (what pandas writes to CSV).
func pyRepr(v float64) string {
	if math.IsNaN(v) {
		return ""
	}
	if a := math.Abs(v); a != 0 && (a < 1e-4 || a >= 1e16) {
		s := strconv.FormatFloat(v, 'e', -1, 64) // 1e-05, 1.5e+16
		return s
	}
	s := strconv.FormatFloat(v, 'f', -1, 64)
	if !strings.ContainsAny(s, ".") {
		s += ".0"
	}
	return s
}

// WriteCSV writes the trade list as pandas' to_csv(index=False) did.
func WriteCSV(w io.Writer, trades []Trade) error {
	cw := csv.NewWriter(w)
	wedge := len(trades) > 0 && trades[0].Shape != ""
	head := []string{"setup_at", "side"}
	if wedge {
		head = append(head, "shape")
	}
	head = append(head, "grade", "entry", "stop", "take_profit", "target_r", "stop_pct", "leverage", "account_risk_pct",
		"filled_at", "closed_at", "outcome", "gross_r", "cost_r", "net_r", "hold_hours", "equity_after")
	if err := cw.Write(head); err != nil {
		return err
	}
	for _, t := range trades {
		row := []string{t.SetupAt, t.Side}
		if wedge {
			row = append(row, t.Shape)
		}
		row = append(row, t.Grade, pyRepr(t.Entry), pyRepr(t.Stop), pyRepr(t.TakeProfit), pyRepr(t.TargetR), pyRepr(t.StopPct),
			pyRepr(t.Leverage), pyRepr(t.AccountRiskPct), t.FilledAt, t.ClosedAt, t.Outcome, pyRepr(t.GrossR), pyRepr(t.CostR),
			pyRepr(t.NetR), pyRepr(t.HoldHours), pyRepr(t.EquityAfter))
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}
