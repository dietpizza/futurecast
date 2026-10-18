package manager

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"deltarisk/engine/binance"
	"deltarisk/engine/config"
	"deltarisk/engine/history"
	"deltarisk/engine/paths"
)

// Manager holds one Worker per tracked coin, loads their history on two one-coin queues
// (downloads, training) and runs the stream loop. Every snapshot goes to Broadcast as a
// FORECAST_UPDATE message (the name is historical).
type Manager struct {
	Config      config.App
	PersistPath string // "" keeps nothing on disk
	Live        bool
	// Broadcast sends a message to every page (set by the server; nil drops them).
	Broadcast func(msg any)
	// NewClient makes a Binance client (history.Client by default; tests swap it).
	NewClient func() *binance.Client

	mu       sync.Mutex
	workers  map[string]*Worker
	order    []string // insertion order, as the page lists them
	saved    map[string]savedCoin
	loading  map[string]bool
	training map[string]bool
	wg       sync.WaitGroup

	downloadQ, trainQ chan struct{} // one coin each: the next coin downloads while the previous one trains
	clientOnce        sync.Once
	client            *binance.Client
}

type savedCoin struct {
	Symbol   string   `json:"symbol"`
	Tier     *string  `json:"tier"`
	Leverage *float64 `json:"leverage"`
}

// New makes a manager; persistPath is relative to the home folder unless absolute ("" keeps nothing).
func New(cfg config.App, persistPath string, live bool) *Manager {
	if persistPath != "" {
		persistPath = paths.Resolve(persistPath)
	}
	m := &Manager{Config: cfg, PersistPath: persistPath, Live: live, workers: map[string]*Worker{},
		saved: map[string]savedCoin{}, loading: map[string]bool{}, training: map[string]bool{},
		downloadQ: make(chan struct{}, 1), trainQ: make(chan struct{}, 1)}
	m.NewClient = func() *binance.Client { return history.Client(cfg) }
	return m
}

func (m *Manager) streamClient() *binance.Client {
	m.clientOnce.Do(func() { m.client = m.NewClient() })
	return m.client
}

func (m *Manager) broadcast(msg any) {
	if m.Broadcast != nil {
		m.Broadcast(msg)
	}
}

func (m *Manager) broadcastWorker(w *Worker) {
	s := w.Snapshot()
	s.Type = "FORECAST_UPDATE"
	m.broadcast(s)
}

// Worker is a tracked coin, or nil.
func (m *Manager) Worker(symbol string) *Worker {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.workers[strings.ToUpper(strings.TrimSpace(symbol))]
}

// Workers are the tracked coins in the order they were added.
func (m *Manager) Workers() []*Worker {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]*Worker, 0, len(m.order))
	for _, s := range m.order {
		out = append(out, m.workers[s])
	}
	return out
}

func (m *Manager) tracked(w *Worker) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.workers[w.Symbol] == w
}

// ───────── watchlist ─────────

// LoadPersisted reads the saved watchlist's symbols; their tier and leverage overrides are
// applied as they are added. Entries are {"symbol", "tier", "leverage"} objects, or plain
// symbols (older files).
func (m *Manager) LoadPersisted() []string {
	if m.PersistPath == "" {
		return nil
	}
	raw, err := os.ReadFile(m.PersistPath)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.Warn("could not load tracked symbols", "path", m.PersistPath, "err", err)
		}
		return nil
	}
	var data struct {
		Symbols []json.RawMessage `json:"symbols"`
	}
	if err := json.Unmarshal(raw, &data); err != nil {
		slog.Warn("could not load tracked symbols", "path", m.PersistPath, "err", err)
		return nil
	}
	var out []string
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, entry := range data.Symbols {
		var item savedCoin
		var plain string
		if json.Unmarshal(entry, &plain) == nil {
			item.Symbol = plain
		} else if json.Unmarshal(entry, &item) != nil {
			continue
		}
		sym := strings.ToUpper(strings.TrimSpace(item.Symbol))
		if sym == "" {
			continue
		}
		item.Symbol = sym
		out = append(out, sym)
		m.saved[sym] = item
	}
	slog.Info("loaded tracked symbols", "path", m.PersistPath, "symbols", out)
	return out
}

// Persist saves the watchlist (symbols with their tier and leverage overrides).
func (m *Manager) Persist() {
	if m.PersistPath == "" {
		return
	}
	coins := []savedCoin{}
	for _, w := range m.Workers() {
		tier, lev := w.Overrides()
		coins = append(coins, savedCoin{w.Symbol, tier, lev})
	}
	payload := map[string]any{"symbols": coins, "updated_at": time.Now().UTC().Format("2006-01-02T15:04:05.000000+00:00")}
	raw, _ := json.MarshalIndent(payload, "", "  ")
	if err := os.MkdirAll(filepath.Dir(m.PersistPath), 0o755); err == nil {
		tmp := m.PersistPath + ".tmp"
		if err = os.WriteFile(tmp, raw, 0o644); err == nil {
			err = os.Rename(tmp, m.PersistPath)
		}
		if err != nil {
			slog.Warn("could not persist tracked symbols", "path", m.PersistPath, "err", err)
		}
	}
}

// Add starts tracking a coin (offline: with synthetic candles, ready at once). In live mode its
// history loads once ScheduleLoad is called.
func (m *Manager) Add(symbol string, persist bool) *Worker {
	sym := strings.ToUpper(strings.TrimSpace(symbol))
	if w := m.Worker(sym); w != nil {
		return w
	}
	w := NewWorker(sym, m.Config)
	slog.Info("adding coin", "symbol", sym, "sector", w.Sector, "benchmark", w.Benchmark)
	m.mu.Lock()
	saved, ok := m.saved[sym]
	delete(m.saved, sym)
	m.mu.Unlock()
	if ok {
		if err := w.SetOverrides(saved.Tier, saved.Leverage); err != nil {
			slog.Warn("ignoring saved overrides", "symbol", sym, "err", err)
		}
	}
	if !m.Live {
		var seed uint64
		for _, r := range sym {
			seed += uint64(r)
		}
		w.LoadSynthetic(seed)
	}
	m.mu.Lock()
	if existing := m.workers[sym]; existing != nil { // added meanwhile
		m.mu.Unlock()
		return existing
	}
	m.workers[sym] = w
	m.order = append(m.order, sym)
	m.mu.Unlock()
	if persist {
		m.Persist()
	}
	return w
}

// SetOverrides sets a coin's tier and leverage overrides (nil: automatic) and saves the watchlist.
func (m *Manager) SetOverrides(w *Worker, tier *string, leverage *float64) error {
	if err := w.SetOverrides(tier, leverage); err != nil {
		return err
	}
	m.Persist()
	return nil
}

// Remove stops tracking a coin; false when it wasn't tracked.
func (m *Manager) Remove(symbol string, persist bool) bool {
	sym := strings.ToUpper(strings.TrimSpace(symbol))
	m.mu.Lock()
	if m.workers[sym] == nil {
		m.mu.Unlock()
		return false
	}
	delete(m.workers, sym)
	for i, s := range m.order {
		if s == sym {
			m.order = append(m.order[:i], m.order[i+1:]...)
			break
		}
	}
	m.mu.Unlock()
	if persist {
		m.Persist()
	}
	slog.Info("removed coin", "symbol", sym)
	return true
}

// Snapshots are every tracked coin's state.
func (m *Manager) Snapshots() []Snapshot {
	out := []Snapshot{}
	for _, w := range m.Workers() {
		out = append(out, w.Snapshot())
	}
	return out
}

// ───────── history ─────────

// ScheduleLoad queues a background history download and model training unless one is queued or running.
func (m *Manager) ScheduleLoad(symbol string) {
	sym := strings.ToUpper(strings.TrimSpace(symbol))
	m.mu.Lock()
	if m.loading[sym] || m.workers[sym] == nil {
		m.mu.Unlock()
		return
	}
	m.loading[sym] = true
	m.wg.Add(1)
	m.mu.Unlock()
	go func() {
		defer m.wg.Done()
		defer func() {
			m.mu.Lock()
			delete(m.loading, sym)
			m.mu.Unlock()
		}()
		m.LoadHistory(sym)
	}()
}

// EnsureLoaded queues every coin that has nothing loaded or failed.
func (m *Manager) EnsureLoaded() {
	for _, w := range m.Workers() {
		if s := w.Status(); s == "pending" || s == "failed" {
			m.ScheduleLoad(w.Symbol)
		}
	}
}

// Download and Prepare are the steps LoadHistory runs (swapped in tests).
var (
	Download = func(w *Worker, c *binance.Client) error { return w.DownloadHistory(c) }
	Prepare  = func(w *Worker) { w.PrepareModels() }
)

// LoadHistory downloads a coin's history on the download queue, then trains on the training
// queue; each queue takes one coin at a time.
func (m *Manager) LoadHistory(symbol string) {
	w := m.Worker(symbol)
	if w == nil {
		return
	}
	w.setStatus("queued")
	m.broadcastWorker(w)

	m.downloadQ <- struct{}{}
	w.setStatus("downloading")
	m.broadcastWorker(w)
	client := m.NewClient()
	err := Download(w, client)
	<-m.downloadQ
	if err != nil { // the worker logged it and holds status "failed" with the error
		if m.tracked(w) {
			m.broadcastWorker(w)
		}
		return
	}
	m.broadcastWorker(w)

	m.trainQ <- struct{}{}
	if m.tracked(w) {
		Prepare(w)
	}
	<-m.trainQ
	if m.tracked(w) {
		m.broadcastWorker(w)
	}
}

// ScheduleTraining queues a retrain of a ready coin's models on the training queue.
func (m *Manager) ScheduleTraining(symbol string) {
	sym := strings.ToUpper(strings.TrimSpace(symbol))
	m.mu.Lock()
	if m.training[sym] || m.loading[sym] {
		m.mu.Unlock()
		return
	}
	m.training[sym] = true
	m.wg.Add(1)
	m.mu.Unlock()
	go func() {
		defer m.wg.Done()
		defer func() {
			m.mu.Lock()
			delete(m.training, sym)
			m.mu.Unlock()
		}()
		m.trainQ <- struct{}{}
		defer func() { <-m.trainQ }()
		w := m.Worker(sym)
		if w == nil || !w.NeedsTraining(time.Now()) {
			return
		}
		if err := w.Train(false); err != nil {
			slog.Error("retraining failed", "symbol", sym, "err", err)
			return
		}
		slog.Info("retrained risk models", "symbol", sym)
	}()
}

// Wait waits for the queued loads and retrains (tests).
func (m *Manager) Wait() { m.wg.Wait() }

// ───────── streaming ─────────

// Run refreshes prices every poll; candles and risk readings when a candle closes, models once a
// day. Offline it only re-sends the synthetic snapshots. It returns when ctx ends.
func (m *Manager) Run(ctx context.Context, poll time.Duration) {
	slog.Info("starting stream loop", "interval", poll, "live", m.Live)
	if m.Live {
		m.EnsureLoaded()
	}
	for {
		workers := m.Workers()
		wait := poll
		if len(workers) == 0 {
			wait = time.Second
		} else {
			if m.Live {
				client := m.streamClient()
				symbols := make([]string, len(workers))
				for i, w := range workers {
					symbols[i] = w.Symbol
				}
				prices := client.LivePrices(symbols)
				for _, w := range workers {
					if p, ok := prices[w.Symbol]; ok {
						w.SetLivePrice(p)
					}
				}
				var wg sync.WaitGroup
				now := time.Now()
				for _, w := range workers {
					wg.Add(1)
					go func(w *Worker) {
						defer wg.Done()
						w.RefreshCandles(client, now)
					}(w)
				}
				wg.Wait()
				for _, w := range workers {
					if w.NeedsTraining(time.Now()) {
						m.ScheduleTraining(w.Symbol)
					}
				}
			}
			for _, w := range workers {
				if m.tracked(w) {
					m.broadcastWorker(w)
				}
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(wait):
		}
	}
}
