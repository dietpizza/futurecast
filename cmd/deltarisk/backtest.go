package main

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"deltarisk/engine/backtest"
	"deltarisk/engine/candles"
	"deltarisk/engine/config"
	"deltarisk/engine/history"
	"deltarisk/engine/setups"
	"deltarisk/engine/wedges"
)

type backtestFlags struct {
	strategy, entry, context, target, grade, csv string
	days                                         int
	risk, maxLeverage, maker, taker, slippage    float64
	breakeven, noBreakeven                       bool
}

func backtestCommand() *cobra.Command {
	var o backtestFlags
	cmd := &cobra.Command{
		Use:   "backtest SYMBOL",
		Short: "Backtest the setup rules on Binance history (downloads go to the candle cache)",
		Long: "Backtest the setup rules on Binance history (downloads go to the candle cache).\n\n" +
			"SYMBOL is a CoinDCX pair, e.g. B-BTC_USDT.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			fl := cmd.Flags()
			opt := func(name string, v float64) *float64 {
				if fl.Changed(name) {
					return &v
				}
				return nil
			}
			return runBacktest(cmd.OutOrStdout(), cmd.ErrOrStderr(), strings.ToUpper(args[0]), o,
				opt("max-leverage", o.maxLeverage), opt("maker", o.maker), opt("taker", o.taker), opt("slippage", o.slippage))
		},
	}
	f := cmd.Flags()
	f.StringVar(&o.strategy, "strategy", "fvg", "fvg (fair-value-gap setups on an entry chart in a context chart) or wedge "+
		"(wedge breakouts with a retest, on one chart)")
	f.StringVar(&o.entry, "entry", "", "Entry chart interval (default 15m; 1h for wedges)")
	f.StringVar(&o.context, "context", "1h", "Context chart interval (structure and fair value gaps; fvg only)")
	f.IntVar(&o.days, "days", 730, "Days of history to replay")
	f.StringVar(&o.target, "target", "", "Take-profit in multiples of the risk (R), or 'measured' for a wedge's height "+
		"(default 3; 2 for wedges)")
	f.StringVar(&o.grade, "grade", "A+", "Setups to take: A+ or all")
	f.Float64Var(&o.risk, "risk", 1, "Account % risked per trade")
	f.Float64Var(&o.maxLeverage, "max-leverage", 0, "Largest position as a multiple of the account; tighter stops then "+
		"risk less (default 5 for wedges, none for fvg; 0 for none)")
	f.BoolVar(&o.breakeven, "breakeven", true, "Wedges: move the stop to break-even plus fees once price gets 1.5R")
	f.BoolVar(&o.noBreakeven, "no-breakeven", false, "Wedges: never move the stop")
	f.Float64Var(&o.maker, "maker", 0, "Maker fee in basis points, paid by limit orders (default: CoinDCX USDT-M futures, GST included)")
	f.Float64Var(&o.taker, "taker", 0, "Taker fee in basis points, paid by market orders and stops (default: CoinDCX USDT-M futures, GST included)")
	f.Float64Var(&o.slippage, "slippage", 0, "How far past its price a stop fills, in basis points")
	f.StringVar(&o.csv, "csv", "", "Write the trade list to this CSV file")
	return cmd
}

// parseTarget reads --target: a multiple of R ("3", "2.5") or "measured".
func parseTarget(value string, def float64) (setups.Target, error) {
	if value == "" {
		return setups.R(def), nil
	}
	if strings.EqualFold(value, "measured") {
		return setups.Measured, nil
	}
	v, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return setups.Target{}, fmt.Errorf("--target must be a number of R or 'measured', not %q", value)
	}
	return setups.R(v), nil
}

func runBacktest(out, log io.Writer, symbol string, o backtestFlags, maxLeverage, maker, taker, slippage *float64) error {
	strategy := strings.ToLower(o.strategy)
	if strategy != "fvg" && strategy != "wedge" {
		return fmt.Errorf("--strategy must be fvg or wedge, not %q", o.strategy)
	}
	wedge := strategy == "wedge"
	entry := o.entry
	if entry == "" {
		entry = "15m"
		if wedge {
			entry = "1h"
		}
	}
	def := 3.0
	if wedge {
		def = 2.0
	}
	target, err := parseTarget(o.target, def)
	if err != nil {
		return err
	}
	fees := func(maker0, taker0, slip0 *float64) {
		if maker != nil {
			*maker0 = *maker
		}
		if taker != nil {
			*taker0 = *taker
		}
		if slippage != nil {
			*slip0 = *slippage
		}
	}

	type chart struct {
		interval      string
		minutes, warm int
	}
	var charts []chart
	var fvgSpec setups.Spec
	var wedgeSpec wedges.Spec
	if wedge {
		if wedgeSpec, err = wedges.ForChart(entry); err != nil {
			return err
		}
		targets := map[float64]bool{2: true, 2.5: true}
		if !target.Measured {
			targets[target.Multiple] = true
		}
		wedgeSpec.Targets = append([]setups.Target{setups.Measured}, sortedTargets(targets)...)
		if o.noBreakeven || !o.breakeven {
			wedgeSpec.BreakevenAt = nil
		}
		fees(&wedgeSpec.MakerBps, &wedgeSpec.TakerBps, &wedgeSpec.SlippageBps)
		charts = []chart{{entry, wedgeSpec.Minutes, 0}}
		if maxLeverage == nil {
			five := 5.0
			maxLeverage = &five
		}
	} else {
		if target.Measured {
			return fmt.Errorf("a measured-move target needs --strategy wedge")
		}
		if fvgSpec, err = setups.ForCharts(entry, o.context); err != nil {
			return err
		}
		fvgSpec.Targets = sortedTargets(map[float64]bool{2: true, 3: true, 4: true, target.Multiple: true})
		fees(&fvgSpec.MakerBps, &fvgSpec.TakerBps, &fvgSpec.SlippageBps)
		charts = []chart{{entry, fvgSpec.LtfMinutes, 0}, {o.context, fvgSpec.HtfMinutes, fvgSpec.HtfZoneMaxAge}}
	}
	grades := []string{o.grade}
	if strings.EqualFold(o.grade, "all") {
		grades = []string{"A+", "B"}
	}
	cfg := backtest.Config{Target: target, Grades: grades, RiskPct: o.risk}
	if maxLeverage != nil {
		cfg.MaxLeverage = *maxLeverage
	}

	appCfg, err := config.Load("config")
	if err != nil {
		return err
	}
	client := history.Client(appCfg)
	loaded := map[string]*candles.Bars{}
	for _, c := range charts {
		bars := o.days*1440/c.minutes + c.warm
		fmt.Fprintf(log, "Loading %s %s candles of %s…\n", thousands(bars), c.interval, symbol)
		b, source, err := client.CandleHistory(symbol, c.interval, bars, 0, false, nil)
		if err != nil {
			return err
		}
		if source == "" || b.Len() == 0 {
			return fmt.Errorf("no Binance perpetual found for %s", symbol)
		}
		loaded[c.interval] = &b
	}

	var res backtest.Result
	if wedge {
		res = backtest.Wedge(loaded[entry], wedgeSpec, cfg)
	} else {
		res = backtest.FVG(loaded[entry], loaded[o.context], fvgSpec, cfg)
	}
	fmt.Fprintln(out, backtest.FormatReport(symbol, res))
	if o.csv != "" {
		f, err := os.Create(o.csv)
		if err != nil {
			return err
		}
		if err := backtest.WriteCSV(f, res.Trades); err != nil {
			f.Close()
			return err
		}
		if err := f.Close(); err != nil {
			return err
		}
		fmt.Fprintf(out, "\nTrades written to %s\n", o.csv)
	}
	return nil
}

func sortedTargets(set map[float64]bool) []setups.Target {
	var vals []float64
	for v := range set {
		vals = append(vals, v)
	}
	sort.Float64s(vals)
	out := make([]setups.Target, len(vals))
	for i, v := range vals {
		out[i] = setups.R(v)
	}
	return out
}
