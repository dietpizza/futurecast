package main

import "testing"

func TestThousands(t *testing.T) {
	for n, want := range map[int]string{0: "0", 999: "999", 1000: "1,000", 25920: "25,920", 1234567: "1,234,567", -4500: "-4,500"} {
		if got := thousands(n); got != want {
			t.Errorf("thousands(%d) = %s, want %s", n, got, want)
		}
	}
}

func TestShareCountsLabelledCandlesOnly(t *testing.T) {
	if got := share([]string{"", "a", "b", "a", ""}, "a"); got != 2.0/3 {
		t.Fatal(got)
	}
	if share([]string{"", ""}, "a") != 0 {
		t.Fatal("no labels")
	}
}

func TestRiskRejectsBadFlags(t *testing.T) {
	cmd := riskCommand()
	cmd.SetArgs([]string{"B-ETH_USDT", "--window", "2d"})
	cmd.SilenceUsage, cmd.SilenceErrors = true, true
	if err := cmd.Execute(); err == nil || err.Error() != "--window must be one of 5m, 15m, 30m, 1h, 2h, 3h, 4h, 8h" {
		t.Fatal(err)
	}
}
