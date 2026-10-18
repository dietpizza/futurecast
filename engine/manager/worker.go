// Package manager runs the dashboard's coins: closed candles in, risk readings, momentum and
// market state out.
//
// Every number comes from fully closed 5m, 15m and 1h candles of the Binance perpetual behind
// the CoinDCX pair, aligned with the coin's benchmark (history.Aligned; the benchmark's
// volatility is one of the risk inputs). A coin's history (history.Bars per chart) is downloaded
// in the background when it is added, one coin at a time; downloads are kept on disk
// (<home>/cache/candles), so a restart only fetches the candles that closed since. Then the risk
// models are trained per window (risk.Train, with its walk-forward check; status "training"), and
// again once a day. Downloads and training run on separate queues, one coin each: the next coin
// downloads while the previous one trains. Afterwards the stream loop appends each candle as it
// closes, rescores the windows on that chart and rereads the funding rate; between candles only
// the displayed price refreshes. Risk readings are served for the coin's leverage (its tier's
// unless overridden) and DefaultStopPct; the page re-reads the quantiles for another stop.
package manager

import (
	"fmt"
	"log/slog"
	"math"
	"math/rand/v2"
	"strings"
	"sync"
	"time"

	"deltarisk/engine/binance"
	"deltarisk/engine/candles"
	"deltarisk/engine/catalog"
	"deltarisk/engine/config"
	"deltarisk/engine/funding"
	"deltarisk/engine/history"
	"deltarisk/engine/jsonx"
	"deltarisk/engine/regime"
	"deltarisk/engine/risk"
	"deltarisk/engine/series"
)

const (
	OfflineBars    = 3_000
	DefaultStopPct = 5.0 // stop loss in % of margin the readings are served for
	RetrainAfter   = 24 * time.Hour
	MaxLeverage    = 125.0
)

// Progress is what a download or training is doing right now.
type Progress struct {
	Label    string  `json:"label"`
	Step     int     `json:"step"`
	Steps    int     `json:"steps"`
	Done     int     `json:"done"`
	Total    int     `json:"total"`
	Fraction float64 `json:"fraction"`
}

// Worker is one coin's candle history and latest readings. Safe for concurrent use.
//
// Status: "pending" (nothing loaded), "queued", "downloading", "training", "ready" or "failed".
type Worker struct {
	Symbol, Benchmark, Sector string
	cfg                       config.App

	mu               sync.Mutex
	status           string
	err              string
	progress         *Progress
	charts           map[string]*candles.Chart // replaced, never changed in place
	dataSource       string                    // "binance:<market>"; a 1000x market's candles aren't in pair units
	funding          *funding.Reading
	livePrice        *float64
	risk             *risk.Coin
	view             map[string]risk.WindowView
	tierOverride     *string
	leverageOverride *float64
}

// NewWorker makes a coin with nothing loaded.
func NewWorker(symbol string, cfg config.App) *Worker {
	return &Worker{Symbol: symbol, Benchmark: catalog.Benchmark(symbol), Sector: catalog.Sector(symbol), cfg: cfg,
		status: "pending", charts: map[string]*candles.Chart{}, view: map[string]risk.WindowView{}}
}

// Status is the worker's status.
func (w *Worker) Status() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.status
}

func (w *Worker) setStatus(s string) {
	w.mu.Lock()
	w.status = s
	w.mu.Unlock()
}

// ───────── tier and leverage (callers hold w.mu) ─────────

func (w *Worker) volume24h() *float64 {
	h := w.charts["1h"]
	if h.Empty() || !h.Has(w.Symbol, "quote_volume") {
		return nil
	}
	col := h.Col(w.Symbol, "quote_volume")
	v := series.Sum(col[max(0, len(col)-24):])
	return &v
}

func (w *Worker) tierAuto() string { return risk.CoinTier(w.Symbol, w.volume24h()) }

func (w *Worker) tier() string {
	if w.tierOverride != nil {
		return *w.tierOverride
	}
	return w.tierAuto()
}

func (w *Worker) leverage() float64 {
	if w.leverageOverride != nil {
		return *w.leverageOverride
	}
	return risk.Tiers[w.tier()].Leverage
}

// SetOverrides overrides the tier and/or leverage (nil goes back to the automatic one).
func (w *Worker) SetOverrides(tier *string, leverage *float64) error {
	if tier != nil {
		if _, ok := risk.Tiers[*tier]; !ok {
			return fmt.Errorf("tier must be one of %s", strings.Join(risk.TierNames, ", "))
		}
	}
	if leverage != nil && !(*leverage >= 1 && *leverage <= MaxLeverage) {
		return fmt.Errorf("leverage must be between 1 and %g", MaxLeverage)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.tierOverride, w.leverageOverride = tier, leverage
	w.updateView()
	return nil
}

// Overrides are the tier and leverage overrides (nil: automatic).
func (w *Worker) Overrides() (*string, *float64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.tierOverride, w.leverageOverride
}

// ───────── candles ─────────

func base(symbol string) string { b, _ := catalog.BaseQuote(symbol); return b }

func (w *Worker) setProgress(label string, step, steps, done, total int) {
	within := 0.0
	if total > 0 {
		within = float64(done) / float64(total)
	}
	p := &Progress{Label: label, Step: step, Steps: steps, Done: done, Total: total,
		Fraction: roundTo((float64(step-1)+within)/float64(steps), 4)}
	w.mu.Lock()
	w.progress = p
	w.mu.Unlock()
}

func roundTo(x float64, digits int) float64 {
	p := math.Pow(10, float64(digits))
	return math.RoundToEven(x*p) / p
}

// DownloadHistory downloads history.Bars closed candles per chart (status "downloading"; what the
// disk caches hold is not fetched again) and computes the funding reading. It leaves status
// "training" (waiting for PrepareModels); a failure leaves status "failed" and the error.
func (w *Worker) DownloadHistory(client *binance.Client) error {
	w.mu.Lock()
	w.status, w.err = "downloading", ""
	w.mu.Unlock()
	fail := func(err error) error {
		w.mu.Lock()
		w.status, w.err, w.progress = "failed", err.Error(), nil
		w.mu.Unlock()
		slog.Error("loading history failed", "symbol", w.Symbol, "err", err)
		return err
	}
	if _, ok, err := client.Resolve(w.Symbol); err != nil {
		return fail(err)
	} else if !ok {
		return fail(fmt.Errorf("no trading Binance perpetual found for %s", w.Symbol))
	}
	type dl struct{ iv, sym string }
	var downloads []dl
	for _, iv := range history.Charts {
		downloads = append(downloads, dl{iv, w.Symbol}, dl{iv, w.Benchmark})
	}
	loaded := map[string]*candles.Chart{}
	source := ""
	for _, iv := range history.Charts {
		progress := func(sym string, done, total int) {
			for i, d := range downloads {
				if d.iv == iv && d.sym == sym {
					w.setProgress(fmt.Sprintf("%s candles · %s", iv, base(sym)), i+1, len(downloads), done, total)
				}
			}
		}
		chart, src, err := history.Aligned(client, w.Symbol, w.Benchmark, iv, history.Bars[iv], 0, progress)
		if err != nil {
			return fail(err)
		}
		if chart.Empty() {
			return fail(fmt.Errorf("no closed %s candles available", iv))
		}
		loaded[iv], source = chart, src
	}
	w.mu.Lock()
	for iv, c := range loaded {
		w.charts[iv] = c
	}
	w.dataSource = source
	w.updateStats()
	w.status = "training"
	w.progress = &Progress{Label: "waiting to train (one coin at a time)"}
	w.mu.Unlock()
	slog.Info("loaded candles", "symbol", w.Symbol, "5m", loaded["5m"].Len(), "15m", loaded["15m"].Len(), "1h", loaded["1h"].Len())
	return nil
}

// PrepareModels trains the risk models (status "training"), then status "ready". A training
// failure is logged and leaves the coin ready without risk readings.
func (w *Worker) PrepareModels() {
	w.setStatus("training")
	if err := w.Train(true); err != nil {
		slog.Error("training risk models failed", "symbol", w.Symbol, "err", err)
	}
	w.mu.Lock()
	w.status, w.progress = "ready", nil
	w.mu.Unlock()
}

// RefreshCandles appends newly closed candles per chart and rescores; it returns the charts
// updated. It does nothing until the history has loaded.
func (w *Worker) RefreshCandles(client *binance.Client, now time.Time) []string {
	w.mu.Lock()
	ready := w.status == "ready"
	have := map[string]*candles.Chart{}
	for iv, c := range w.charts {
		have[iv] = c
	}
	w.mu.Unlock()
	if !ready {
		return nil
	}
	var updated []string
	nowMs := now.UnixMilli()
	for _, iv := range history.Charts {
		step := candles.MustIntervalMs(iv)
		keep := history.Bars[iv]
		h := have[iv]
		lastClosed := nowMs/step*step - step
		if !h.Empty() && h.OpenTime[h.Len()-1] >= lastClosed {
			continue
		}
		missing := keep
		if !h.Empty() {
			missing = int((lastClosed - h.OpenTime[h.Len()-1]) / step)
		}
		fresh, source, err := history.Aligned(client, w.Symbol, w.Benchmark, iv, min(keep, missing+3), nowMs, nil)
		if err != nil {
			slog.Warn("candle refresh failed", "symbol", w.Symbol, "chart", iv, "err", err)
			continue
		}
		if fresh.Empty() {
			continue
		}
		var next candles.Chart
		if h.Empty() || missing >= keep-3 { // a gap too long to bridge: start over
			next = fresh.Tail(keep)
		} else {
			added := fresh.After(h.OpenTime[h.Len()-1])
			if added.Empty() {
				continue
			}
			joined := h.Concat(&added)
			next = joined.Tail(keep)
		}
		w.mu.Lock()
		w.dataSource = source
		w.charts[iv] = &next
		w.mu.Unlock()
		updated = append(updated, iv)
	}
	if len(updated) > 0 {
		w.mu.Lock()
		w.updateStats()
		if w.risk != nil {
			w.risk.Score(w.charts, updated)
		}
		w.updateView()
		w.mu.Unlock()
	}
	return updated
}

// syntheticBars are random-walk closed candles ending at the last closed one (offline mode only).
func syntheticBars(minutes int64, n int, rng *rand.Rand) candles.Bars {
	step := minutes * 60_000
	end := time.Now().UnixMilli()/step*step - step
	var b candles.Bars
	p := 100.0
	sigma := 0.002 * math.Sqrt(float64(minutes))
	for i := 0; i < n; i++ {
		open := p
		p *= math.Exp(rng.NormFloat64() * sigma)
		wick := math.Abs(rng.NormFloat64() * 0.001 * math.Sqrt(float64(minutes)))
		volume := 1e5 + 4e5*rng.Float64()
		taker := 0.4 + 0.2*rng.Float64()
		b.Append(candles.Bar{Time: end - int64(n-1-i)*step, Open: open, High: math.Max(open, p) * (1 + wick),
			Low: math.Min(open, p) * (1 - wick), Close: p, Volume: volume, QuoteVolume: volume * p, Count: 100,
			TakerBuyVolume: volume * taker, TakerBuyQuoteVolume: volume * p * taker})
	}
	return b
}

// LoadSynthetic fills random-walk candles and trains on them (offline mode only), so the page has
// something to render.
func (w *Worker) LoadSynthetic(seed uint64) {
	rng := rand.New(rand.NewPCG(seed, 0))
	charts := map[string]*candles.Chart{}
	for _, iv := range history.Charts {
		minutes := candles.MustIntervalMs(iv) / 60_000
		target, bench := syntheticBars(minutes, OfflineBars, rng), syntheticBars(minutes, OfflineBars, rng)
		c, err := candles.Align(minutes*60_000, 3, candles.Asset{Symbol: w.Symbol, Bars: target}, candles.Asset{Symbol: w.Benchmark, Bars: bench})
		if err == nil {
			charts[iv] = &c
		}
	}
	w.mu.Lock()
	w.charts = charts
	w.dataSource = "synthetic (offline mode)"
	w.updateStats()
	w.mu.Unlock()
	if err := w.Train(false); err != nil {
		slog.Error("training risk models failed", "symbol", w.Symbol, "err", err)
	}
	w.setStatus("ready")
}

// ───────── statistics ─────────

// updateStats rereads the funding rate (callers hold w.mu).
func (w *Worker) updateStats() { w.funding = funding.View(w.charts, w.Symbol) }

// Train fits and checks the risk models on the candles on hand; the models in use keep serving
// until the new ones replace them.
func (w *Worker) Train(progress bool) error {
	w.mu.Lock()
	tier := risk.Tiers[w.tier()]
	charts := map[string]*candles.Chart{}
	for iv, c := range w.charts {
		charts[iv] = c
	}
	w.mu.Unlock()
	var report func(string, int, int)
	if progress {
		report = func(window string, step, steps int) {
			w.setProgress("training risk models · "+window, step, steps, 0, 0)
		}
	}
	coin, err := risk.Train(w.Symbol, w.Benchmark, charts, risk.StopMove(DefaultStopPct, tier.Leverage),
		risk.LiquidationMove(tier.Leverage, tier.MMR), risk.DefaultSpec, regime.DefaultSpec, report)
	if err != nil {
		return err
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	var closed []string
	for iv, c := range w.charts {
		if charts[iv] != c { // candles closed while it trained
			closed = append(closed, iv)
		}
	}
	if len(closed) > 0 {
		coin.Score(w.charts, closed)
	}
	w.risk = coin
	w.updateView()
	return nil
}

// NeedsTraining says whether a ready coin's models are missing or RetrainAfter old.
func (w *Worker) NeedsTraining(now time.Time) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.status != "ready" {
		return false
	}
	return w.risk == nil || now.Sub(w.risk.TrainedAt) >= RetrainAfter
}

// TrainedAt is when the models in use were trained (zero without models).
func (w *Worker) TrainedAt() time.Time {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.risk == nil {
		return time.Time{}
	}
	return w.risk.TrainedAt
}

// updateView recomputes the readings (callers hold w.mu).
func (w *Worker) updateView() {
	if w.risk == nil {
		w.view = map[string]risk.WindowView{}
		return
	}
	w.view = w.risk.View(w.leverage(), risk.Tiers[w.tier()].MMR, DefaultStopPct)
}

// SetLivePrice sets the displayed price.
func (w *Worker) SetLivePrice(p float64) {
	w.mu.Lock()
	w.livePrice = &p
	w.mu.Unlock()
}

// ───────── output ─────────

// Chart is one interval's candles (nil before they load).
func (w *Worker) Chart(interval string) *candles.Chart {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.charts[interval]
}

// Funding is the funding reading (nil without one).
func (w *Worker) Funding() *funding.Reading {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.funding
}

// Snapshot is the coin as the page reads it (web/src/lib/types.ts Coin). Type is set on WebSocket messages.
type Snapshot struct {
	Type         string                 `json:"type,omitempty"`
	Symbol       string                 `json:"symbol"`
	Sector       string                 `json:"sector"`
	HasData      bool                   `json:"has_data"`
	Status       string                 `json:"status"`
	Progress     *Progress              `json:"progress"`
	Error        *string                `json:"error"`
	History      map[string]HistorySpan `json:"history"`
	CurrentPrice *jsonx.F               `json:"current_price"`
	Tier         string                 `json:"tier"`
	TierAuto     string                 `json:"tier_auto"`
	Leverage     float64                `json:"leverage"`
	Overrides    Overrides              `json:"overrides"`
	Risk         RiskSnapshot           `json:"risk"`
	Funding      *funding.Reading       `json:"funding"`
}

// HistorySpan is one chart's candles: how many and the first one's open time.
type HistorySpan struct {
	Candles int     `json:"candles"`
	From    *string `json:"from"`
}

// Overrides are the tier and leverage overrides (null: automatic).
type Overrides struct {
	Tier     *string  `json:"tier"`
	Leverage *float64 `json:"leverage"`
}

// RiskSnapshot is the readings at DefaultStopPct and the coin's leverage.
type RiskSnapshot struct {
	StopPct   float64                    `json:"stop_pct"`
	MMR       float64                    `json:"mmr"`
	TrainedAt *string                    `json:"trained_at"`
	Windows   map[string]risk.WindowView `json:"windows"`
}

// Snapshot is the coin's current state.
func (w *Worker) Snapshot() Snapshot {
	w.mu.Lock()
	defer w.mu.Unlock()
	c15 := w.charts["15m"]
	var price *jsonx.F
	if w.livePrice != nil {
		p := jsonx.F(*w.livePrice)
		price = &p
	} else if !c15.Empty() && !strings.Contains(w.dataSource, ":1000") { // skipped for 1000x-quoted contracts
		col := c15.Col(w.Symbol, "close")
		p := jsonx.F(col[len(col)-1])
		price = &p
	}
	hist := map[string]HistorySpan{}
	for _, iv := range history.Charts {
		span := HistorySpan{}
		if c := w.charts[iv]; !c.Empty() {
			from := jsonx.Stamp(c.OpenTime[0]).String()
			span = HistorySpan{Candles: c.Len(), From: &from}
		}
		hist[iv] = span
	}
	var errText *string
	if w.err != "" {
		e := w.err
		errText = &e
	}
	var trained *string
	if w.risk != nil {
		s := w.risk.TrainedAt.UTC().Format("2006-01-02 15:04:05.000000+00:00")
		trained = &s
	}
	tier := w.tier()
	return Snapshot{Symbol: w.Symbol, Sector: w.Sector, HasData: !c15.Empty(), Status: w.status, Progress: w.progress,
		Error: errText, History: hist, CurrentPrice: price, Tier: tier, TierAuto: w.tierAuto(), Leverage: w.leverage(),
		Overrides: Overrides{w.tierOverride, w.leverageOverride},
		Risk:      RiskSnapshot{StopPct: DefaultStopPct, MMR: risk.Tiers[tier].MMR, TrainedAt: trained, Windows: w.view},
		Funding:   w.funding}
}
