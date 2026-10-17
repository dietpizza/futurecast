package main

import (
	"fmt"
	"io"
	"math"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"deltarisk/internal/candles"
	"deltarisk/internal/catalog"
	"deltarisk/internal/config"
	"deltarisk/internal/history"
	"deltarisk/internal/regime"
	"deltarisk/internal/risk"
	"deltarisk/internal/series"
)

func pct(x float64, digits int) string { return fmt.Sprintf("%.*f%%", digits, 100.0*x) }

func day(ms int64, layout string) string { return time.UnixMilli(ms).UTC().Format(layout) }

func riskCommand() *cobra.Command {
	var (
		window, tier    string
		stop, leverage  float64
		showCalibration bool
	)
	cmd := &cobra.Command{
		Use:   "risk SYMBOL",
		Short: "Walk-forward check of the risk models on Binance history, and today's readings",
		Long: "Walk-forward check of the risk models on Binance history, and today's readings.\n\n" +
			"SYMBOL is a CoinDCX pair, e.g. B-SOL_USDT. Candles go to the cache in the home folder.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRisk(cmd.OutOrStdout(), cmd.ErrOrStderr(), strings.ToUpper(args[0]), window, stop, leverage, tier, showCalibration)
		},
	}
	f := cmd.Flags()
	f.StringVar(&window, "window", "", "One window (5m, 15m, 30m, 1h, 2h, 3h, 4h, 8h); default all")
	f.Float64Var(&stop, "stop", 5, "Stop loss in % of margin (the PnL % CoinDCX shows)")
	f.Float64Var(&leverage, "leverage", 0, "Leverage (default by tier: megacap 20, altcoin 10, shitcoin 3)")
	f.StringVar(&tier, "tier", "", "megacap, altcoin or shitcoin (default from the coin)")
	f.BoolVar(&showCalibration, "calibration", false, "Also print the stop-hit calibration buckets")
	return cmd
}

func runRisk(out, log io.Writer, symbol, window string, stop, leverage float64, tier string, showCalibration bool) error {
	windows := regime.Windows
	if window != "" {
		w, ok := regime.ByName(window)
		if !ok {
			names := make([]string, len(regime.Windows))
			for i, w := range regime.Windows {
				names[i] = w.Name
			}
			return fmt.Errorf("--window must be one of %s", strings.Join(names, ", "))
		}
		windows = []regime.Window{w}
	}
	var charts []string
	for _, iv := range history.Charts {
		use := iv == "1h" // funding
		for _, w := range windows {
			use = use || w.Interval == iv
		}
		if use {
			charts = append(charts, iv)
		}
	}
	cfg, err := config.Load("config")
	if err != nil {
		return err
	}
	benchmark := catalog.Benchmark(symbol)
	client := history.Client(cfg)
	if _, ok, err := client.Resolve(symbol); err != nil {
		return err
	} else if !ok {
		return fmt.Errorf("no Binance perpetual found for %s", symbol)
	}
	loaded := map[string]*candles.Chart{}
	for _, iv := range charts {
		fmt.Fprintf(log, "Loading %s %s candles of %s and %s…\n", thousands(history.Bars[iv]), iv, symbol, benchmark)
		chart, _, err := history.Aligned(client, symbol, benchmark, iv, history.Bars[iv], 0, nil)
		if err != nil {
			return err
		}
		if chart.Empty() {
			return fmt.Errorf("no closed %s candles for %s", iv, symbol)
		}
		loaded[iv] = chart
	}

	qv := loaded["1h"].Col(symbol, "quote_volume")
	volume24h := series.Sum(qv[max(0, len(qv)-24):])
	tierName := strings.ToLower(tier)
	if tierName == "" {
		tierName = risk.CoinTier(symbol, &volume24h)
	}
	t, ok := risk.Tiers[tierName]
	if !ok {
		return fmt.Errorf("--tier must be one of %s", strings.Join(risk.TierNames, ", "))
	}
	lev := leverage
	if lev == 0 {
		lev = t.Leverage
	}
	stopPx, liqPx := risk.StopMove(stop, lev), risk.LiquidationMove(lev, t.MMR)
	spec, regimeSpec := risk.DefaultSpec, regime.DefaultSpec

	fmt.Fprintf(out, "\n%s · %s · %gx · 24h volume $%sM · benchmark %s\n", symbol, tierName, lev, thousands(int(math.RoundToEven(volume24h/1e6))), benchmark)
	fmt.Fprintf(out, "Stop −%g%% of margin = %s price · liquidation ≈ %s price\n", stop, pct(stopPx, 2), pct(liqPx, 2))
	for _, iv := range charts {
		c := loaded[iv]
		fmt.Fprintf(out, "  %s: %s candles, %s → %s UTC\n", iv, thousands(c.Len()), day(c.OpenTime[0], "2006-01-02"),
			day(c.OpenTime[c.Len()-1], "2006-01-02 15:04"))
	}

	type today struct {
		at       int64
		readings [2]risk.Reading
	}
	checks := map[string]risk.Check{}
	todays := map[string]today{}
	frames := map[string]regime.Frame{}
	for _, w := range windows {
		fmt.Fprintf(log, "Checking %s…\n", w.Name)
		data := risk.NewWindowData(loaded, symbol, w, benchmark, regimeSpec)
		frames[w.Name] = data.Regime
		if len(data.Trainable) == 0 {
			continue
		}
		check, err := risk.WalkForward(data, stopPx, liqPx, spec)
		if err != nil {
			return err
		}
		checks[w.Name] = check
		model, err := risk.Fit(data, data.Trainable, spec)
		if err != nil {
			return err
		}
		at, _, readings := risk.Latest(model, data, stopPx, liqPx, lev)
		todays[w.Name] = today{at, readings}
	}

	fmt.Fprintf(out, "\nWalk-forward check: train on the first %.0f%% (expanding), %d test blocks after it, %gh gap. "+
		"Coverage = share of test windows worse than the predicted 1-in-10 bad case (want %.0f%%–%.0f%%). "+
		"Pinball loss in price %%.\n", 100*spec.TrainShare, spec.Folds, spec.GapHours, 100*spec.CoverageBand[0], 100*spec.CoverageBand[1])
	head := fmt.Sprintf("%-7s%-6s%7s  %8s  %8s  %13s %7s  %5s  check", "window", "side", "tests", "coverage", "pinball", "stop-hit pred", "actual", "brier")
	fmt.Fprintln(out, head)
	fmt.Fprintln(out, strings.Repeat("─", len([]rune(head))))
	for _, w := range windows {
		c := checks[w.Name]
		for side, name := range risk.SideNames {
			s := c.Sides[side]
			if s == nil {
				fmt.Fprintf(out, "%-7s%-6s  not enough history\n", w.Name, name)
				continue
			}
			mark := "✗ —"
			if s.Checked {
				mark = "✓"
			}
			fmt.Fprintf(out, "%-7s%-6s%7s  %8s  %8.3f  %13s %7s  %5.3f  %s\n", w.Name, name, thousands(c.Tests),
				pct(s.Coverage, 1), s.PinballPct, pct(s.StopHitPredicted, 1), pct(s.StopHitRate, 1), s.Brier, mark)
		}
	}
	if showCalibration {
		fmt.Fprintln(out, "\nStop-hit calibration: bucket of predicted chance → mean predicted / actual (test windows)")
		for _, w := range windows {
			for side, name := range risk.SideNames {
				s := checks[w.Name].Sides[side]
				if s == nil {
					continue
				}
				var cells []string
				for _, b := range s.Calibration {
					cells = append(cells, fmt.Sprintf("%.0f%%–%.0f%% %.0f%%/%.0f%% (%s)", 100*b.From, 100*b.To, 100*b.Predicted, 100*b.Actual, thousands(b.N)))
				}
				fmt.Fprintf(out, "%-7s%-6s%s\n", w.Name, name, strings.Join(cells, " · "))
			}
		}
	}

	fmt.Fprintln(out, "\nLabels over the history (share of candles)")
	fmt.Fprintf(out, "%-7s%8s%8s%8s   %8s%8s%8s%8s\n", "window", "bullish", "neutral", "bearish", "trending", "choppy", "quiet", "wild")
	for _, w := range windows {
		f := frames[w.Name]
		line := fmt.Sprintf("%-7s", w.Name)
		for _, k := range []string{"bullish", "neutral", "bearish"} {
			line += fmt.Sprintf("%8s", pct(share(f.MomentumLabel, k), 0))
		}
		line += "   "
		for _, k := range []string{"trending", "choppy", "quiet", "wild"} {
			line += fmt.Sprintf("%8s", pct(share(f.State, k), 0))
		}
		fmt.Fprintln(out, line)
	}

	var last int64
	for _, td := range todays {
		last = max(last, td.at)
	}
	fmt.Fprintf(out, "\nToday (models trained on all history; each chart's last closed candle, the newest opened %s UTC)\n", day(last, "2006-01-02 15:04"))
	head = fmt.Sprintf("%-7s%13s %-9s%-7s  %-6s%9s %8s %6s %6s", "window", "momentum", "state", "vol", "side", "bad case", "margin", "liq", "score")
	fmt.Fprintln(out, head)
	fmt.Fprintln(out, strings.Repeat("─", len([]rune(head))))
	for _, w := range windows {
		td, ok := todays[w.Name]
		if !ok {
			continue
		}
		f := frames[w.Name]
		i := len(f.State) - 1
		reg := fmt.Sprintf("%13s %-9s%-7s", "—", "—", "—")
		if f.State[i] != "" {
			reg = fmt.Sprintf("%+5.0f %-7s %-9s%-7s", f.Momentum[i], f.MomentumLabel[i], f.State[i], f.VolatilityLabel[i])
		}
		for side, name := range risk.SideNames {
			r := td.readings[side]
			score := "—"
			if r.Score != nil {
				score = fmt.Sprint(*r.Score)
				if s := checks[w.Name].Sides[side]; s == nil || !s.Checked {
					score = "(" + score + ")"
				}
			}
			label, regCol := w.Name, reg
			if side == risk.Short {
				label, regCol = "", strings.Repeat(" ", len([]rune(reg)))
			}
			fmt.Fprintf(out, "%-7s%s  %-6s%8.2f%% %7.1f%% %6s %6s\n", label, regCol, name, r.BadCasePct, r.BadCaseMarginPct,
				pct(r.LiquidationChance, 2), score)
		}
	}
	fmt.Fprintln(out, "Score = the chance in % that the stop is hit within the window; in brackets where the check failed (the page would show —).")
	return nil
}

// share is the labels' share equal to k among the labelled candles (pandas value_counts(normalize=True)).
func share(labels []string, k string) float64 {
	n, hit := 0, 0
	for _, l := range labels {
		if l != "" {
			n++
			if l == k {
				hit++
			}
		}
	}
	if n == 0 {
		return 0
	}
	return float64(hit) / float64(n)
}

// thousands formats an integer with commas, like Python's {:,}.
func thousands(n int) string {
	s := fmt.Sprint(n)
	neg := strings.HasPrefix(s, "-")
	s = strings.TrimPrefix(s, "-")
	var out []string
	for len(s) > 3 {
		out = append([]string{s[len(s)-3:]}, out...)
		s = s[:len(s)-3]
	}
	out = append([]string{s}, out...)
	r := strings.Join(out, ",")
	if neg {
		r = "-" + r
	}
	return r
}
