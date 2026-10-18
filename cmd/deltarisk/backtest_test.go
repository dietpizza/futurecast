package main

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"deltarisk/internal/binance/binancetest"
	"deltarisk/internal/paths"
)

// price is a wavy path with structure on every scale, so the rules find setups.
func price(t int64) float64 {
	x := float64(t) / 3_600_000
	return 100 * math.Exp(0.05*math.Sin(x/97)+0.02*math.Sin(x/13.3+1)+0.008*math.Sin(x/3.1+2)+0.003*math.Sin(x*1.7))
}

// fakeHome points the home folder at a temp one whose config sends Binance requests to a fake.
func fakeHome(t *testing.T) {
	t.Helper()
	now := time.Now().UnixMilli() // the command uses the real clock
	f := &binancetest.Fake{Clock: binancetest.NewClock(now), Markets: []string{"TESTUSDT"},
		Kline: func(market string, open, step int64) []any {
			o, c := price(open), price(open+step)
			s := func(v float64) string { return fmt.Sprint(v) }
			return []any{open, s(o), s(math.Max(o, c) * 1.0007), s(math.Min(o, c) * 0.9993), s(c), "1000", open + step - 1,
				s(1000 * c), 10, "500", s(500 * c), "0"}
		}}
	srv := f.Start(t)
	tmp := t.TempDir()
	home := filepath.Join(tmp, "home")
	t.Setenv("DELTARISK_HOME", home)
	t.Setenv("FUTURECAST_HOME", "")
	oldDefault, oldLegacy := paths.DefaultHome, paths.LegacyHome
	paths.DefaultHome, paths.LegacyHome = filepath.Join(tmp, "default"), filepath.Join(tmp, "legacy")
	t.Cleanup(func() { paths.DefaultHome, paths.LegacyHome = oldDefault, oldLegacy })
	os.MkdirAll(filepath.Join(home, "config"), 0o755)
	os.WriteFile(filepath.Join(home, "config", "assets.yaml"), []byte("data_source:\n  base_url: "+srv.URL+"\n"), 0o644)
}

func runCmd(t *testing.T, args ...string) (string, error) {
	t.Helper()
	cmd := backtestCommand()
	var out, errOut bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&errOut)
	cmd.SetArgs(args)
	cmd.SilenceUsage, cmd.SilenceErrors = true, true
	err := cmd.Execute()
	return out.String(), err
}

func TestBacktestCommand(t *testing.T) {
	fakeHome(t)
	path := filepath.Join(t.TempDir(), "trades.csv")
	out, err := runCmd(t, "b-test_usdt", "--entry", "15m", "--context", "1h", "--days", "120", "--grade", "all",
		"--maker", "1", "--taker", "4", "--csv", path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out, "B-TEST_USDT · 15m entries in 1h context · A+/B setups · 3R target") ||
		!strings.Contains(out, "fees 0.01% maker / 0.04% taker, 0.02% stop slippage") || !strings.Contains(out, "setups found") {
		t.Fatal(out) // fees not given keep the defaults
	}
	if !strings.Contains(out, "No filled trades.") {
		f, _ := os.Open(path)
		rows, err := csv.NewReader(f).ReadAll()
		f.Close()
		if err != nil || len(rows) < 2 {
			t.Fatal(rows, err)
		}
		col := -1
		for i, name := range rows[0] {
			if name == "outcome" {
				col = i
			}
		}
		for _, r := range rows[1:] {
			if o := r[col]; o != "won" && o != "lost" && o != "timeout" {
				t.Fatal(r)
			}
		}
	}
}

func TestWedgeBacktestCommand(t *testing.T) {
	fakeHome(t)
	out, err := runCmd(t, "b-test_usdt", "--strategy", "wedge", "--days", "400", "--grade", "all")
	if err != nil {
		t.Fatal(err)
	}
	// One chart (1h by default), a 2R target, leverage capped at 5x and the stop moved at 1.5R unless told otherwise
	if !strings.HasPrefix(out, "B-TEST_USDT · wedge breakouts on 1h · A+/B setups · 2R target · 1% risk per trade, "+
		"at most 5x leverage\nmarket entries, stop to break-even at 1.5R") || !strings.Contains(out, " wedges, ") {
		t.Fatal(out)
	}
	out, err = runCmd(t, "b-test_usdt", "--strategy", "wedge", "--days", "100", "--no-breakeven", "--max-leverage", "0", "--target", "measured")
	if err != nil || !strings.Contains(out, "measured-move target · 1% risk per trade\nmarket entries, stop never moved") {
		t.Fatal(out, err)
	}
	if _, err := runCmd(t, "b-test_usdt", "--target", "measured", "--days", "30"); err == nil || !strings.Contains(err.Error(), "needs --strategy wedge") {
		t.Fatal(err)
	}
	if _, err := runCmd(t, "b-test_usdt", "--target", "lots"); err == nil || !strings.Contains(err.Error(), "--target must be") {
		t.Fatal(err)
	}
	if _, err := runCmd(t, "b-nope_usdt", "--days", "30"); err == nil || !strings.Contains(err.Error(), "no Binance perpetual") {
		t.Fatal(err)
	}
}
