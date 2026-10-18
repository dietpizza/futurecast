// Package server is the dashboard's web server: REST for the watchlist and coin settings, /ws for
// live updates, and the Svelte page (compiled into the binary, or web/dist on disk).
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"

	"deltarisk/engine/binance"
	"deltarisk/engine/board"
	"deltarisk/engine/catalog"
	"deltarisk/engine/config"
	"deltarisk/engine/manager"
)

// NotBuilt is the page served when there is no built page.
const NotBuilt = `<!DOCTYPE html><meta charset="utf-8"><title>DeltaRisk</title>
<body style="font-family: sans-serif; max-width: 40em; margin: 4em auto; line-height: 1.5">
<h1>DeltaRisk</h1>
<p>The dashboard hasn't been built. Run <code>mise run build</code> and reload, or use <code>mise dev</code>
for the development server (it serves the page on port 5000).</p>
<p>The API is up: <a href="/api/coins">/api/coins</a>.</p>`

// Options configure a Server.
type Options struct {
	Config         config.App
	InitialSymbols []string // nil: the saved watchlist
	Live           bool
	PersistPath    string // "tracked_symbols.json" under the home folder by default; "-" keeps nothing
	Page           fs.FS  // the built page (index.html, favicon.svg, assets/); nil: not built
	// PageURL, when set, is where the page is served instead (the Vite dev server in `mise dev`):
	// the page's routes redirect there and Page is ignored.
	PageURL string
	Poll    time.Duration
	// Catalog lists Binance perpetuals for the add-coin search (a 4-second, one-try client by default).
	Catalog catalog.PerpetualLister
	// BoardClient makes the market board's client (a client from the config by default).
	BoardClient func() board.Client
	// NewClient makes the manager's Binance clients (history.Client by default).
	NewClient func() *binance.Client
}

// Server is the dashboard.
type Server struct {
	Manager *manager.Manager
	Board   *board.Board
	Catalog *catalog.Catalog
	opts    Options
	hub     *hub
	cancel  context.CancelFunc
	done    chan struct{}
}

// New makes the server; Start restores the watchlist and starts the stream loop.
func New(o Options) *Server {
	if o.PersistPath == "" {
		o.PersistPath = "tracked_symbols.json"
	}
	if o.PersistPath == "-" {
		o.PersistPath = ""
	}
	if o.Poll == 0 {
		o.Poll = 3 * time.Second
	}
	m := manager.New(o.Config, o.PersistPath, o.Live)
	if o.NewClient != nil {
		m.NewClient = o.NewClient
	}
	if o.Catalog == nil {
		src := o.Config.Assets.DataSource
		o.Catalog = binance.New(binance.Options{BaseURL: src.BaseURL, Timeout: 4 * time.Second, MaxRetries: 1})
	}
	if o.BoardClient == nil {
		o.BoardClient = func() board.Client { return m.NewClient() }
	}
	s := &Server{Manager: m, Board: board.New(o.BoardClient), Catalog: catalog.New(o.Catalog), opts: o, hub: newHub()}
	m.Broadcast = s.hub.broadcast
	return s
}

// Start restores the coins the user added (InitialSymbols win, e.g. in tests; there are no
// default coins) and starts the stream loop.
func (s *Server) Start() {
	symbols := s.opts.InitialSymbols
	if symbols == nil {
		symbols = s.Manager.LoadPersisted()
	}
	for _, sym := range symbols {
		s.Manager.Add(sym, false)
	}
	s.Manager.Persist()
	ctx, cancel := context.WithCancel(context.Background())
	s.cancel, s.done = cancel, make(chan struct{})
	go func() {
		defer close(s.done)
		s.Manager.Run(ctx, s.opts.Poll)
	}()
}

// Stop ends the stream loop and the market board.
func (s *Server) Stop() {
	if s.cancel != nil {
		s.cancel()
		<-s.done
	}
	s.Board.Stop()
	s.hub.closeAll()
}

// ───────── helpers ─────────

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Warn("writing a response failed", "err", err)
	}
}

func fail(w http.ResponseWriter, status int, detail any) {
	writeJSON(w, status, map[string]any{"detail": detail})
}

// Handler routes every request.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	if u := strings.TrimRight(s.opts.PageURL, "/"); u != "" {
		toPage := func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, u+r.URL.RequestURI(), http.StatusTemporaryRedirect)
		}
		mux.HandleFunc("GET /{$}", toPage)
		mux.HandleFunc("GET /favicon.svg", toPage)
		mux.HandleFunc("GET /assets/{path...}", toPage)
		s.routes(mux)
		return mux
	}
	page := s.opts.Page
	isFile := func(name string) bool {
		if page == nil {
			return false
		}
		info, err := fs.Stat(page, name)
		return err == nil && !info.IsDir()
	}

	// index.html and favicon.svg are revalidated on every load, so a new build shows on reload
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		if !isFile("index.html") {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprint(w, NotBuilt)
			return
		}
		w.Header().Set("Cache-Control", "no-cache")
		http.ServeFileFS(w, r, page, "index.html")
	})
	mux.HandleFunc("GET /favicon.svg", func(w http.ResponseWriter, r *http.Request) {
		if !isFile("favicon.svg") {
			fail(w, http.StatusNotFound, "Not Found")
			return
		}
		w.Header().Set("Cache-Control", "no-cache")
		http.ServeFileFS(w, r, page, "favicon.svg")
	})
	// Vite's hashed assets: a changed file gets a new name, so the browser may keep each one for good
	mux.HandleFunc("GET /assets/{path...}", func(w http.ResponseWriter, r *http.Request) {
		name := "assets/" + r.PathValue("path")
		if !fs.ValidPath(name) || !isFile(name) {
			fail(w, http.StatusNotFound, "Not Found")
			return
		}
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		http.ServeFileFS(w, r, page, name)
	})
	s.routes(mux)
	return mux
}

// routes adds the API and the WebSocket.
func (s *Server) routes(mux *http.ServeMux) {

	mux.HandleFunc("GET /api/coins", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, s.Manager.Snapshots())
	})
	mux.HandleFunc("GET /api/coindcx/futures", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, s.Catalog.Instruments())
	})
	mux.HandleFunc("GET /api/market", s.market)
	mux.HandleFunc("POST /api/coins", s.addCoin)
	mux.HandleFunc("POST /api/coins/{symbol}/reload", s.reloadCoin)
	mux.HandleFunc("PATCH /api/coins/{symbol}", s.updateCoin)
	mux.HandleFunc("DELETE /api/coins/{symbol}", s.deleteCoin)
	mux.HandleFunc("GET /ws", s.websocket)
}

// market is price, 24h volume and the change over 5m … 24h for every catalog pair (the add-pair
// drawer). The board starts on the first request and stops when the drawer has been closed for a
// while; offline there is no market data.
func (s *Server) market(w http.ResponseWriter, r *http.Request) {
	if !s.Manager.Live {
		writeJSON(w, http.StatusOK, board.Offline())
		return
	}
	s.Board.Touch()
	s.Board.WaitReady(5 * time.Second)
	items := s.Catalog.Instruments()
	symbols := make([]string, len(items))
	for i, it := range items {
		symbols[i] = it.Symbol
	}
	writeJSON(w, http.StatusOK, s.Board.View(symbols))
}

func (s *Server) addCoin(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Symbol *string `json:"symbol"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Symbol == nil {
		fail(w, http.StatusUnprocessableEntity, "body must be {\"symbol\": \"B-SOL_USDT\"}")
		return
	}
	sym := strings.ToUpper(strings.TrimSpace(*req.Symbol))
	if sym == "" {
		fail(w, http.StatusBadRequest, "Invalid symbol")
		return
	}
	if catalog.DetectExchange(sym) != "coindcx" { // CoinDCX symbols only (B- or I- or containing _)
		fail(w, http.StatusBadRequest, fmt.Sprintf("Only CoinDCX symbols are supported (e.g. B-DOGE_USDT, B-SOL_USDT, "+
			"B-PEPE_USDT, I-BTC_INR). Received '%s'.", sym))
		return
	}
	worker := s.Manager.Add(sym, true)
	// Download its history in the background (one coin at a time)
	if st := worker.Status(); s.Manager.Live && (st == "pending" || st == "failed") {
		s.Manager.ScheduleLoad(worker.Symbol)
	}
	item := worker.Snapshot()
	msg := item
	msg.Type = "FORECAST_UPDATE"
	s.hub.broadcast(msg)
	writeJSON(w, http.StatusOK, item)
}

func (s *Server) reloadCoin(w http.ResponseWriter, r *http.Request) {
	sym := strings.ToUpper(strings.TrimSpace(r.PathValue("symbol")))
	if s.Manager.Worker(sym) == nil {
		fail(w, http.StatusNotFound, "Coin not tracked")
		return
	}
	status := "offline"
	if s.Manager.Live {
		s.Manager.ScheduleLoad(sym)
		status = "load_queued"
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": status, "symbol": sym})
}

// updateCoin sets the tier and/or leverage; fields left out stay as they are, null goes back to
// the automatic tier or leverage.
func (s *Server) updateCoin(w http.ResponseWriter, r *http.Request) {
	worker := s.Manager.Worker(r.PathValue("symbol"))
	if worker == nil {
		fail(w, http.StatusNotFound, "Coin not tracked")
		return
	}
	var req map[string]json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fail(w, http.StatusUnprocessableEntity, "body must be a JSON object")
		return
	}
	tier, leverage := worker.Overrides()
	if raw, ok := req["tier"]; ok {
		var t *string
		if err := json.Unmarshal(raw, &t); err != nil {
			fail(w, http.StatusUnprocessableEntity, "tier must be a string or null")
			return
		}
		tier = nil
		if t != nil && *t != "" {
			lower := strings.ToLower(*t)
			tier = &lower
		}
	}
	if raw, ok := req["leverage"]; ok {
		var l *float64
		if err := json.Unmarshal(raw, &l); err != nil {
			fail(w, http.StatusUnprocessableEntity, "leverage must be a number or null")
			return
		}
		leverage = l
	}
	if err := s.Manager.SetOverrides(worker, tier, leverage); err != nil {
		fail(w, http.StatusBadRequest, err.Error())
		return
	}
	item := worker.Snapshot()
	msg := item
	msg.Type = "FORECAST_UPDATE"
	s.hub.broadcast(msg)
	writeJSON(w, http.StatusOK, item)
}

func (s *Server) deleteCoin(w http.ResponseWriter, r *http.Request) {
	symbol := r.PathValue("symbol")
	if !s.Manager.Remove(symbol, true) {
		fail(w, http.StatusNotFound, "Coin not tracked")
		return
	}
	s.hub.broadcast(map[string]string{"type": "COIN_REMOVED", "symbol": strings.ToUpper(symbol)})
	writeJSON(w, http.StatusOK, map[string]string{"status": "removed", "symbol": strings.ToUpper(symbol)})
}

// ───────── /ws ─────────

// websocket sends the coins as INIT_STATE, then every update; the page only listens (coins change
// through the REST API), so whatever it sends is read and dropped.
func (s *Server) websocket(w http.ResponseWriter, r *http.Request) {
	// Any origin, as the Python server allowed: the dev page on :5000 reaches the API on :5001 through
	// Vite's proxy, and the dashboard is read-only market data
	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
	if err != nil {
		return
	}
	// INIT_STATE is queued before the client joins the hub, so it is always the first message
	c := &client{conn: conn, out: make(chan []byte, 256)}
	init, _ := json.Marshal(map[string]any{"type": "INIT_STATE", "coins": s.Manager.Snapshots()})
	c.send(init)
	s.hub.add(c)
	slog.Info("websocket client connected", "clients", s.hub.count())
	defer func() {
		s.hub.remove(c)
		slog.Info("websocket client disconnected", "clients", s.hub.count())
	}()
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	go func() { // reads and drops what the page sends; ends the stream when the page goes
		defer cancel()
		for {
			if _, _, err := conn.Read(ctx); err != nil {
				return
			}
		}
	}()
	for {
		select {
		case <-ctx.Done():
			return
		case msg, ok := <-c.out:
			if !ok {
				conn.Close(websocket.StatusGoingAway, "")
				return
			}
			wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
			err := conn.Write(wctx, websocket.MessageText, msg)
			cancel()
			if err != nil {
				return
			}
		}
	}
}

type client struct {
	conn   *websocket.Conn
	out    chan []byte
	mu     sync.Mutex
	closed bool
}

// send queues a message; a client too slow to keep up loses it.
func (c *client) send(msg []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	select {
	case c.out <- msg:
	default:
	}
}

func (c *client) close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.closed {
		c.closed = true
		close(c.out)
	}
}

type hub struct {
	mu      sync.Mutex
	clients map[*client]bool
}

func newHub() *hub { return &hub{clients: map[*client]bool{}} }

func (h *hub) add(c *client) {
	h.mu.Lock()
	h.clients[c] = true
	h.mu.Unlock()
}

func (h *hub) remove(c *client) {
	h.mu.Lock()
	delete(h.clients, c)
	h.mu.Unlock()
}

func (h *hub) count() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.clients)
}

func (h *hub) broadcast(msg any) {
	raw, err := json.Marshal(msg)
	if err != nil {
		slog.Warn("broadcast failed", "err", err)
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	for c := range h.clients {
		c.send(raw)
	}
}

func (h *hub) closeAll() {
	h.mu.Lock()
	defer h.mu.Unlock()
	for c := range h.clients {
		c.close()
	}
}
