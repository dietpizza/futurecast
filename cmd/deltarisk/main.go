// Command deltarisk: the web dashboard (the default command; `deltarisk web` works too), the risk
// models' walk-forward check (risk) and a backtest of the setup rules (backtest).
package main

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"deltarisk"
	"deltarisk/engine/config"
	"deltarisk/engine/server"
)

func main() {
	root := dashboardCommand("deltarisk")
	root.Short = "Risk readings for CoinDCX futures pairs, from Binance market data"
	root.SilenceUsage, root.SilenceErrors = true, true
	web := dashboardCommand("web")
	web.Short = "Start the web dashboard (the same as running deltarisk with no command)"
	root.AddCommand(web, riskCommand(), backtestCommand())
	if err := root.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

// pageToServe is the page the dashboard serves and where it comes from: --web-dist when given,
// else the page compiled into the binary, else web/dist on disk (in the working folder, or next
// to the binary: bin/deltarisk -> web/dist). nil when there is none.
func pageToServe(webDist string) (fs.FS, string) {
	if webDist != "" {
		return os.DirFS(webDist), webDist
	}
	if page, ok := deltarisk.Page(); ok {
		return page, "compiled into the binary"
	}
	candidates := []string{filepath.Join("web", "dist")}
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(exe), "..", "web", "dist"))
	}
	for _, dir := range candidates {
		if info, err := os.Stat(filepath.Join(dir, "index.html")); err == nil && !info.IsDir() {
			return os.DirFS(dir), dir
		}
	}
	return nil, "not built"
}

func dashboardCommand(use string) *cobra.Command {
	var (
		host, webDist string
		port          int
		offline       bool
	)
	cmd := &cobra.Command{
		Use:   use,
		Short: "Start the DeltaRisk web dashboard",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runDashboard(host, port, !offline, webDist)
		},
	}
	f := cmd.Flags()
	f.StringVar(&host, "host", "0.0.0.0", "Interface to bind the dashboard to (default: every interface)")
	f.IntVarP(&port, "port", "p", 8000, "Port to serve the dashboard on")
	f.BoolVar(&offline, "offline", false, "Use simulated candles instead of live Binance data")
	f.StringVar(&webDist, "web-dist", "", "Serve the built page from this folder (default: the page compiled into the binary, else web/dist)")
	return cmd
}

func runDashboard(host string, port int, live bool, webDist string) error {
	cfg, err := config.Load("config")
	if err != nil {
		return err
	}
	page, from := pageToServe(webDist)
	srv := server.New(server.Options{Config: cfg, Live: live, Page: page})
	addr := net.JoinHostPort(host, strconv.Itoa(port))
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	srv.Start()
	httpServer := &http.Server{Handler: srv.Handler(), ReadHeaderTimeout: 10 * time.Second}
	shown := host
	if host == "0.0.0.0" || host == "::" { // a wildcard bind is not a browsable address
		shown = "127.0.0.1"
	}
	fmt.Printf("\nΔ DeltaRisk running at: http://%s (listening on %s; page: %s)\n\n", net.JoinHostPort(shown, strconv.Itoa(port)), host, from)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	errs := make(chan error, 1)
	go func() { errs <- httpServer.Serve(listener) }()
	select {
	case err := <-errs:
		srv.Stop()
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	case <-ctx.Done():
	}
	shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	srv.Stop() // closes the WebSocket streams, so Shutdown doesn't wait on them
	return httpServer.Shutdown(shutdown)
}
