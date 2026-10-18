// Command deltarisk: the web dashboard (the default command), the risk models' walk-forward
// check (risk) and a backtest of the setup rules (backtest).
package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

func main() {
	root := &cobra.Command{
		Use:           "deltarisk",
		Short:         "Risk readings for CoinDCX futures pairs, from Binance market data",
		SilenceUsage:  true,
		SilenceErrors: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			return errors.New("the dashboard isn't ported to Go yet (GO-REWRITE-PLAN.md phase 5): run it with `uv run deltarisk`")
		},
	}
	root.AddCommand(riskCommand(), backtestCommand())
	if err := root.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}
