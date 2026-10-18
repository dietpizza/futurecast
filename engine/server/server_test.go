package server

import (
	"bytes"
	"context"
	"encoding/json"
	"io/fs"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"testing/fstest"
	"time"

	"github.com/coder/websocket"

	"deltarisk/engine/catalog"
	"deltarisk/engine/config"
	"deltarisk/engine/paths"
	"deltarisk/engine/regime"
)

// offlineCatalog: the search sees only the curated registry.
type offlineCatalog struct{}

func (offlineCatalog) Perpetuals() (map[string]catalog.Perpetual, error) {
	return map[string]catalog.Perpetual{}, nil
}

func tempHome(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	home := filepath.Join(tmp, "deltarisk_home")
	t.Setenv("DELTARISK_HOME", home)
	t.Setenv("FUTURECAST_HOME", "")
	oldDefault, oldLegacy := paths.DefaultHome, paths.LegacyHome
	paths.DefaultHome, paths.LegacyHome = filepath.Join(tmp, "default_home"), filepath.Join(tmp, "legacy_home")
	t.Cleanup(func() { paths.DefaultHome, paths.LegacyHome = oldDefault, oldLegacy })
	return home
}

// app starts an offline dashboard; symbols nil restores the saved watchlist.
func app(t *testing.T, symbols []string, persist, webDist string) (*Server, *httptest.Server) {
	t.Helper()
	var page fs.FS
	if webDist != "" {
		page = os.DirFS(webDist)
	}
	s := New(Options{Config: config.Default(), InitialSymbols: symbols, Live: false, PersistPath: persist, Page: page,
		Catalog: offlineCatalog{}, Poll: time.Hour})
	s.Start()
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() {
		ts.Close()
		s.Stop()
	})
	return s, ts
}

func call(t *testing.T, method, url string, body any) (*http.Response, []byte) {
	t.Helper()
	var r *bytes.Reader
	if body != nil {
		raw, _ := json.Marshal(body)
		r = bytes.NewReader(raw)
	} else {
		r = bytes.NewReader(nil)
	}
	req, _ := http.NewRequest(method, url, r)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var buf bytes.Buffer
	buf.ReadFrom(resp.Body)
	return resp, buf.Bytes()
}

func decode[T any](t *testing.T, raw []byte) T {
	t.Helper()
	var v T
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("%v: %s", err, raw)
	}
	return v
}

type coin = map[string]any

func coins(t *testing.T, base string) []coin {
	_, raw := call(t, "GET", base+"/api/coins", nil)
	return decode[[]coin](t, raw)
}

func symbolsOf(cs []coin) []string {
	var out []string
	for _, c := range cs {
		out = append(out, c["symbol"].(string))
	}
	return out
}

func TestServesTheBuiltPage(t *testing.T) {
	tempHome(t)
	dist := filepath.Join(t.TempDir(), "dist")
	os.MkdirAll(filepath.Join(dist, "assets"), 0o755)
	os.WriteFile(filepath.Join(dist, "index.html"), []byte(`<title>DeltaRisk</title><script src="/assets/index-abc.js"></script>`), 0o644)
	os.WriteFile(filepath.Join(dist, "assets", "index-abc.js"), []byte("console.log(1)"), 0o644)
	os.WriteFile(filepath.Join(dist, "favicon.svg"), []byte("<svg/>"), 0o644)
	_, ts := app(t, []string{}, "-", dist)

	resp, body := call(t, "GET", ts.URL+"/", nil)
	if resp.StatusCode != 200 || !strings.Contains(string(body), "DeltaRisk") || resp.Header.Get("Cache-Control") != "no-cache" {
		t.Fatal(resp.StatusCode, resp.Header)
	}
	resp, _ = call(t, "GET", ts.URL+"/assets/index-abc.js", nil)
	if resp.StatusCode != 200 || !strings.Contains(resp.Header.Get("Cache-Control"), "immutable") {
		t.Fatal(resp.StatusCode, resp.Header)
	}
	for _, missing := range []string{"/assets/missing.js", "/assets/../index.html", "/assets/"} {
		if resp, _ := call(t, "GET", ts.URL+missing, nil); resp.StatusCode != 404 {
			t.Fatal(missing, resp.StatusCode)
		}
	}
	resp, _ = call(t, "GET", ts.URL+"/favicon.svg", nil)
	if resp.StatusCode != 200 || resp.Header.Get("Cache-Control") != "no-cache" {
		t.Fatal(resp.StatusCode, resp.Header)
	}
}

// The page compiled into the binary is an fs.FS like this one.
func TestServesAnEmbeddedPage(t *testing.T) {
	tempHome(t)
	page := fstest.MapFS{
		"index.html":           {Data: []byte(`<title>DeltaRisk</title>`)},
		"favicon.svg":          {Data: []byte("<svg/>")},
		"assets/index-abc.js":  {Data: []byte("console.log(1)")},
		"assets/index-abc.css": {Data: []byte("body{}")},
	}
	s := New(Options{Config: config.Default(), InitialSymbols: []string{}, PersistPath: "-", Page: page,
		Catalog: offlineCatalog{}, Poll: time.Hour})
	s.Start()
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() { ts.Close(); s.Stop() })

	resp, body := call(t, "GET", ts.URL+"/", nil)
	if resp.StatusCode != 200 || string(body) != "<title>DeltaRisk</title>" || resp.Header.Get("Cache-Control") != "no-cache" ||
		!strings.HasPrefix(resp.Header.Get("Content-Type"), "text/html") {
		t.Fatal(resp.StatusCode, resp.Header, string(body))
	}
	resp, _ = call(t, "GET", ts.URL+"/assets/index-abc.css", nil)
	if resp.StatusCode != 200 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/css") ||
		!strings.Contains(resp.Header.Get("Cache-Control"), "immutable") {
		t.Fatal(resp.StatusCode, resp.Header)
	}
	if resp, _ := call(t, "GET", ts.URL+"/favicon.svg", nil); resp.StatusCode != 200 {
		t.Fatal(resp.StatusCode)
	}
	for _, missing := range []string{"/assets/nope.js", "/assets/", "/assets/..%2findex.html", "/index.html"} {
		if resp, _ := call(t, "GET", ts.URL+missing, nil); resp.StatusCode == 200 {
			t.Fatal(missing, resp.StatusCode)
		}
	}
	if resp, _ := call(t, "GET", ts.URL+"/api/coins", nil); resp.StatusCode != 200 {
		t.Fatal("the API still answers", resp.StatusCode)
	}
}

// In mise dev the page comes from Vite on :5000: the backend's page routes send the browser there.
func TestPageURLRedirectsThePageToTheDevServer(t *testing.T) {
	tempHome(t)
	s := New(Options{Config: config.Default(), InitialSymbols: []string{}, PersistPath: "-", PageURL: "http://127.0.0.1:5000/",
		Page: fstest.MapFS{"index.html": {Data: []byte("stale build")}}, Catalog: offlineCatalog{}, Poll: time.Hour})
	s.Start()
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() { ts.Close(); s.Stop() })
	noFollow := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	for path, want := range map[string]string{"/": "http://127.0.0.1:5000/", "/assets/a.js?v=1": "http://127.0.0.1:5000/assets/a.js?v=1",
		"/favicon.svg": "http://127.0.0.1:5000/favicon.svg"} {
		resp, err := noFollow.Get(ts.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusTemporaryRedirect || resp.Header.Get("Location") != want {
			t.Fatal(path, resp.StatusCode, resp.Header.Get("Location"))
		}
	}
	if resp, _ := call(t, "GET", ts.URL+"/api/coins", nil); resp.StatusCode != 200 {
		t.Fatal("the API still answers", resp.StatusCode)
	}
}

func TestWithoutABuildSaysHowToMakeOne(t *testing.T) {
	tempHome(t)
	_, ts := app(t, []string{}, "-", "")
	resp, body := call(t, "GET", ts.URL+"/", nil)
	if resp.StatusCode != 503 || !strings.Contains(string(body), "mise run build") {
		t.Fatal(resp.StatusCode, string(body))
	}
	if resp, _ := call(t, "GET", ts.URL+"/api/coins", nil); resp.StatusCode != 200 {
		t.Fatal(resp.StatusCode)
	}
	if resp, _ := call(t, "GET", ts.URL+"/favicon.svg", nil); resp.StatusCode != 404 {
		t.Fatal(resp.StatusCode)
	}
}

func TestCoinDCXFuturesCatalog(t *testing.T) {
	tempHome(t)
	_, ts := app(t, []string{"B-DOGE_USDT", "B-SOL_USDT"}, "-", "")
	_, raw := call(t, "GET", ts.URL+"/api/coindcx/futures", nil)
	items := decode[[]map[string]string](t, raw)
	by := map[string]map[string]string{}
	memes := 0
	for _, it := range items {
		by[it["symbol"]] = it
		if it["sector"] == "MEME" {
			memes++
		}
	}
	if len(items) < 30 || memes < 5 {
		t.Fatal(len(items), memes)
	}
	for _, s := range []string{"B-DOGE_USDT", "B-SOL_USDT", "B-PEPE_USDT", "B-BTC_USDT", "B-ZK_USDT"} {
		if by[s] == nil {
			t.Fatal("missing", s)
		}
	}
	zk := by["B-ZK_USDT"]
	if zk["base"] != "ZK" || zk["name"] != "zkSync" || zk["coindcx_url"] != "https://coindcx.com/futures/B-ZK_USDT" || len(zk) != 6 {
		t.Fatal(zk)
	}
	if by["I-BTC_INR"]["coindcx_url"] != "https://coindcx.com/trade/I-BTC_INR" {
		t.Fatal(by["I-BTC_INR"])
	}
}

func TestCoinsServeOnlyWhatThePageReads(t *testing.T) {
	tempHome(t)
	_, ts := app(t, []string{"B-DOGE_USDT", "B-SOL_USDT"}, "-", "")
	cs := coins(t, ts.URL)
	if got := symbolsOf(cs); !reflect.DeepEqual(got, []string{"B-DOGE_USDT", "B-SOL_USDT"}) {
		t.Fatal(got)
	}
	want := []string{"symbol", "sector", "has_data", "status", "progress", "error", "history", "current_price", "tier",
		"tier_auto", "leverage", "overrides", "risk", "funding"}
	sort.Strings(want)
	for _, c := range cs {
		var keys []string
		for k := range c {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		if !reflect.DeepEqual(keys, want) {
			t.Fatal(keys)
		}
		// Offline mode trains the baseline on synthetic candles straight away
		if c["has_data"] != true || c["status"] != "ready" || len(c["history"].(map[string]any)) != 3 {
			t.Fatal(c["status"], c["history"])
		}
	}
	if cs[0]["sector"] != "MEME" {
		t.Fatal(cs[0]["sector"])
	}
}

func TestReloadEndpoint(t *testing.T) {
	tempHome(t)
	_, ts := app(t, []string{"B-DOGE_USDT"}, "-", "")
	_, raw := call(t, "POST", ts.URL+"/api/coins/B-DOGE_USDT/reload", nil)
	if got := decode[map[string]string](t, raw); !reflect.DeepEqual(got, map[string]string{"status": "offline", "symbol": "B-DOGE_USDT"}) {
		t.Fatal(got)
	}
	if resp, _ := call(t, "POST", ts.URL+"/api/coins/B-NOPE_USDT/reload", nil); resp.StatusCode != 404 {
		t.Fatal(resp.StatusCode)
	}
}

func TestAddAndDeleteCoin(t *testing.T) {
	tempHome(t)
	_, ts := app(t, []string{"B-DOGE_USDT", "B-SOL_USDT"}, "-", "")
	resp, raw := call(t, "POST", ts.URL+"/api/coins", map[string]string{"symbol": "BTCUSDT"})
	if resp.StatusCode != 400 || !strings.Contains(decode[map[string]string](t, raw)["detail"], "Only CoinDCX symbols are supported") {
		t.Fatal(resp.StatusCode, string(raw))
	}
	if resp, _ := call(t, "POST", ts.URL+"/api/coins", map[string]string{"symbol": "  "}); resp.StatusCode != 400 {
		t.Fatal(resp.StatusCode)
	}
	if resp, _ := call(t, "POST", ts.URL+"/api/coins", map[string]int{"coin": 1}); resp.StatusCode != 422 {
		t.Fatal(resp.StatusCode)
	}
	_, raw = call(t, "POST", ts.URL+"/api/coins", map[string]string{"symbol": "b-pepe_usdt"})
	pepe := decode[coin](t, raw)
	if pepe["symbol"] != "B-PEPE_USDT" || pepe["sector"] != "MEME" || pepe["tier"] != "shitcoin" {
		t.Fatal(pepe)
	}
	_, raw = call(t, "POST", ts.URL+"/api/coins", map[string]string{"symbol": "B-BLUR_USDT"})
	blur := decode[coin](t, raw)
	if blur["status"] != "ready" || blur["risk"].(map[string]any)["windows"].(map[string]any)["4h"] == nil {
		t.Fatal(blur["status"])
	}
	if got := symbolsOf(coins(t, ts.URL)); !reflect.DeepEqual(got, []string{"B-DOGE_USDT", "B-SOL_USDT", "B-PEPE_USDT", "B-BLUR_USDT"}) {
		t.Fatal(got)
	}
	for _, s := range []string{"B-PEPE_USDT", "b-blur_usdt"} {
		resp, raw := call(t, "DELETE", ts.URL+"/api/coins/"+s, nil)
		if resp.StatusCode != 200 || decode[map[string]string](t, raw)["status"] != "removed" {
			t.Fatal(resp.StatusCode)
		}
	}
	if resp, _ := call(t, "DELETE", ts.URL+"/api/coins/B-PEPE_USDT", nil); resp.StatusCode != 404 {
		t.Fatal(resp.StatusCode)
	}
	if got := symbolsOf(coins(t, ts.URL)); !reflect.DeepEqual(got, []string{"B-DOGE_USDT", "B-SOL_USDT"}) {
		t.Fatal(got)
	}
}

func TestWebSocketStream(t *testing.T) {
	tempHome(t)
	_, ts := app(t, []string{"B-DOGE_USDT", "B-SOL_USDT"}, "-", "")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, strings.Replace(ts.URL, "http", "ws", 1)+"/ws", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.CloseNow()
	conn.SetReadLimit(1 << 24)
	_, raw, err := conn.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	init := decode[map[string]any](t, raw)
	if init["type"] != "INIT_STATE" || len(init["coins"].([]any)) != 2 {
		t.Fatal(init["type"])
	}
	// Whatever the page sends is dropped; the stream stays up and carries updates
	conn.Write(ctx, websocket.MessageText, []byte(`{"action": "add_coin", "symbol": "B-BLUR_USDT"}`))
	call(t, "POST", ts.URL+"/api/coins", map[string]string{"symbol": "B-WAL_USDT"})
	_, raw, err = conn.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if msg := decode[map[string]any](t, raw); msg["type"] != "FORECAST_UPDATE" || msg["symbol"] != "B-WAL_USDT" || msg["status"] != "ready" {
		t.Fatal(msg["type"], msg["symbol"])
	}
	call(t, "DELETE", ts.URL+"/api/coins/b-wal_usdt", nil)
	_, raw, _ = conn.Read(ctx)
	if msg := decode[map[string]any](t, raw); msg["type"] != "COIN_REMOVED" || msg["symbol"] != "B-WAL_USDT" {
		t.Fatal(msg)
	}
}

func TestTrackedCoinsPersistAcrossRestarts(t *testing.T) {
	tempHome(t)
	file := filepath.Join(t.TempDir(), "saved_coins.json")
	_, ts := app(t, []string{"B-DOGE_USDT", "B-SOL_USDT"}, file, "")
	call(t, "POST", ts.URL+"/api/coins", map[string]string{"symbol": "B-TRUTH_USDT"})
	raw, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	var saved struct {
		Symbols []struct{ Symbol string } `json:"symbols"`
	}
	json.Unmarshal(raw, &saved)
	if len(saved.Symbols) != 3 || saved.Symbols[2].Symbol != "B-TRUTH_USDT" {
		t.Fatal(string(raw))
	}

	_, ts2 := app(t, nil, file, "")
	if got := symbolsOf(coins(t, ts2.URL)); !reflect.DeepEqual(got, []string{"B-DOGE_USDT", "B-SOL_USDT", "B-TRUTH_USDT"}) {
		t.Fatal(got)
	}
	call(t, "DELETE", ts2.URL+"/api/coins/B-DOGE_USDT", nil)

	_, ts3 := app(t, nil, file, "")
	if got := symbolsOf(coins(t, ts3.URL)); !reflect.DeepEqual(got, []string{"B-SOL_USDT", "B-TRUTH_USDT"}) {
		t.Fatal(got)
	}
}

func TestWatchlistStartsEmptyWithoutDefaultCoins(t *testing.T) {
	home := tempHome(t)
	_, ts := app(t, nil, "", "") // the default file in the home folder
	if cs := coins(t, ts.URL); len(cs) != 0 {
		t.Fatal(cs)
	}
	call(t, "POST", ts.URL+"/api/coins", map[string]string{"symbol": "B-WAL_USDT"})
	if _, err := os.Stat(filepath.Join(home, "tracked_symbols.json")); err != nil {
		t.Fatal("the watchlist lives in the home folder:", err)
	}
	_, ts2 := app(t, nil, "", "")
	if got := symbolsOf(coins(t, ts2.URL)); !reflect.DeepEqual(got, []string{"B-WAL_USDT"}) {
		t.Fatal(got)
	}
	call(t, "DELETE", ts2.URL+"/api/coins/B-WAL_USDT", nil)
	_, ts3 := app(t, nil, "", "")
	if cs := coins(t, ts3.URL); len(cs) != 0 {
		t.Fatal(cs)
	}
}

func TestCoinsServeRiskReadings(t *testing.T) {
	tempHome(t)
	_, ts := app(t, []string{"B-DOGE_USDT", "B-SOL_USDT"}, "-", "")
	cs := coins(t, ts.URL)
	doge, sol := cs[0], cs[1]
	if doge["tier"] != "shitcoin" || doge["leverage"] != 3.0 || sol["tier"] != "megacap" || sol["leverage"] != 20.0 {
		t.Fatal(doge["tier"], sol["tier"])
	}
	if !reflect.DeepEqual(sol["overrides"], map[string]any{"tier": nil, "leverage": nil}) || sol["funding"] != nil {
		t.Fatal(sol["overrides"], sol["funding"]) // synthetic candles carry no funding
	}
	r := sol["risk"].(map[string]any)
	if r["stop_pct"] != 5.0 || r["mmr"] != 0.004 || r["trained_at"] == nil {
		t.Fatal(r["stop_pct"], r["mmr"], r["trained_at"])
	}
	windows := r["windows"].(map[string]any)
	if len(windows) != len(regime.Windows) {
		t.Fatal(len(windows))
	}
	in := func(v any, options ...string) bool {
		for _, o := range options {
			if v == o {
				return true
			}
		}
		return false
	}
	for name, raw := range windows {
		w := raw.(map[string]any)
		if !in(w["state"], "trending", "choppy", "quiet", "wild") || !in(w["momentum_label"], "bullish", "bearish", "neutral") ||
			!in(w["volatility"], "calm", "normal", "high", "wild") || math.Abs(w["momentum"].(float64)) > 100 {
			t.Fatal(name, w["state"], w["momentum_label"], w["volatility"])
		}
		rec := w["trend_record"].(map[string]any)
		if rec["base"].(map[string]any)["n"].(float64) <= 0 || math.Abs(rec["round_trip_pct"].(float64)-0.177) > 1e-12 || len(rec) != 4 {
			t.Fatal(name, rec)
		}
		for _, side := range []string{"long", "short"} {
			s := w[side].(map[string]any)
			hit := s["stop_hit_chance"].(float64)
			if _, has := s["model"]; has || s["score"].(float64) != math.RoundToEven(100*hit) ||
				math.Abs(s["bad_case_margin_pct"].(float64)-20*s["bad_case_pct"].(float64)) > 1e-9 ||
				!(0 <= s["liquidation_chance"].(float64) && s["liquidation_chance"].(float64) <= hit && hit <= 1) ||
				s["reason"] == "" {
				t.Fatal(name, side, s)
			}
			if _, ok := s["checked"].(bool); !ok {
				t.Fatal(name, side, "checked")
			}
			q := s["quantiles"].(map[string]any)
			moves := q["moves_pct"].([]any)
			if len(q["alphas"].([]any)) != len(moves) || !sort.SliceIsSorted(moves, func(i, j int) bool { return moves[i].(float64) < moves[j].(float64) }) {
				t.Fatal(name, side, "quantiles")
			}
		}
	}
}

func TestTierAndLeverageOverridesRoundTrip(t *testing.T) {
	tempHome(t)
	file := filepath.Join(t.TempDir(), "saved_coins.json")
	_, ts := app(t, []string{"B-SOL_USDT"}, file, "")
	hit := func(c coin) float64 {
		return c["risk"].(map[string]any)["windows"].(map[string]any)["1h"].(map[string]any)["long"].(map[string]any)["stop_hit_chance"].(float64)
	}
	before := hit(coins(t, ts.URL)[0])
	_, raw := call(t, "PATCH", ts.URL+"/api/coins/B-SOL_USDT", map[string]any{"tier": "Altcoin"})
	out := decode[coin](t, raw)
	if out["tier"] != "altcoin" || out["tier_auto"] != "megacap" || out["leverage"] != 10.0 || !(hit(out) < before) {
		t.Fatal(out["tier"], out["leverage"]) // 10x: the -5% stop is twice as far
	}
	_, raw = call(t, "PATCH", ts.URL+"/api/coins/B-SOL_USDT", map[string]any{"leverage": 5})
	out = decode[coin](t, raw)
	if out["tier"] != "altcoin" || out["leverage"] != 5.0 || !reflect.DeepEqual(out["overrides"], map[string]any{"tier": "altcoin", "leverage": 5.0}) {
		t.Fatal(out["overrides"])
	}
	for _, c := range []struct {
		body   map[string]any
		symbol string
		status int
	}{{map[string]any{"tier": "moon"}, "B-SOL_USDT", 400}, {map[string]any{"leverage": 500}, "B-SOL_USDT", 400},
		{map[string]any{"leverage": 5}, "B-NOPE_USDT", 404}, {map[string]any{"leverage": "lots"}, "B-SOL_USDT", 422}} {
		if resp, _ := call(t, "PATCH", ts.URL+"/api/coins/"+c.symbol, c.body); resp.StatusCode != c.status {
			t.Fatal(c.body, resp.StatusCode)
		}
	}
	raw, _ = os.ReadFile(file)
	var saved struct {
		Symbols []map[string]any `json:"symbols"`
	}
	json.Unmarshal(raw, &saved)
	if !reflect.DeepEqual(saved.Symbols, []map[string]any{{"symbol": "B-SOL_USDT", "tier": "altcoin", "leverage": 5.0}}) {
		t.Fatal(saved.Symbols)
	}
	_, ts2 := app(t, nil, file, "")
	c := coins(t, ts2.URL)[0]
	if c["tier"] != "altcoin" || c["leverage"] != 5.0 {
		t.Fatal(c["tier"], c["leverage"])
	}
	_, raw = call(t, "PATCH", ts2.URL+"/api/coins/B-SOL_USDT", map[string]any{"tier": nil, "leverage": nil})
	if out := decode[coin](t, raw); out["tier"] != "megacap" || out["leverage"] != 20.0 {
		t.Fatal(out["tier"], out["leverage"])
	}
}

func TestOldWatchlistFormatStillLoads(t *testing.T) {
	tempHome(t)
	file := filepath.Join(t.TempDir(), "saved_coins.json")
	os.WriteFile(file, []byte(`{"symbols": ["B-ETH_USDT", "b-sol_usdt", "", 7], "updated_at": "2026-10-01"}`), 0o644)
	_, ts := app(t, nil, file, "")
	if got := symbolsOf(coins(t, ts.URL)); !reflect.DeepEqual(got, []string{"B-ETH_USDT", "B-SOL_USDT"}) {
		t.Fatal(got)
	}
	raw, _ := os.ReadFile(file)
	var saved struct {
		Symbols []map[string]any `json:"symbols"`
	}
	json.Unmarshal(raw, &saved)
	if !reflect.DeepEqual(saved.Symbols[0], map[string]any{"symbol": "B-ETH_USDT", "tier": nil, "leverage": nil}) {
		t.Fatal(saved.Symbols)
	}
}

func TestOfflineHasNoMarketData(t *testing.T) {
	tempHome(t)
	_, ts := app(t, []string{}, "-", "")
	_, raw := call(t, "GET", ts.URL+"/api/market", nil)
	out := decode[map[string]any](t, raw)
	if out["offline"] != true || len(out["coins"].(map[string]any)) != 0 ||
		!reflect.DeepEqual(out["windows"], []any{"5m", "15m", "1h", "4h", "24h"}) || out["updated_at"] != nil {
		t.Fatal(out)
	}
}
