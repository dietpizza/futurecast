package risk

import (
	"fmt"
	"math"
	"strings"
	"time"

	"deltarisk/internal/candles"
	"deltarisk/internal/jsonx"
	"deltarisk/internal/regime"
	"deltarisk/internal/series"
)

// PageAlphas are the quantiles sent to the page, dense enough that interpolating them
// (exceedChance in web/src/lib/risk.ts) matches the empirical stop-hit chance.
var PageAlphas = func() []float64 {
	var out []float64
	for i := 0; ; i++ {
		a := 0.02 + float64(i)*0.02 // np.arange(0.02, 0.985, 0.02)
		if a >= 0.985 {
			break
		}
		out = append(out, math.Round(a*1000)/1000)
	}
	return append(out, 0.99, 0.995, 0.999)
}()

// MinTraining is the readings with known targets a window needs before it gets a model.
const MinTraining = 1000

// Reason gives the main drivers of one side's risk in a few words. fundingPct and otherBadRatio
// are NaN when unknown.
func Reason(side int, volRatio, fundingPct float64, reg regime.Row, otherBadRatio float64) string {
	var parts []string
	if series.Finite(volRatio) && (volRatio >= 1.3 || volRatio <= 0.75) {
		parts = append(parts, fmt.Sprintf("volatility %.1f× normal", volRatio))
	}
	if reg.State == "wild" {
		parts = append(parts, "market wild")
	}
	if series.Finite(fundingPct) {
		if fundingPct >= 90 && side == Long {
			parts = append(parts, "funding crowded long")
		} else if fundingPct <= 10 && side == Short {
			parts = append(parts, "funding crowded short")
		}
	}
	against := [2]string{"bearish", "bullish"}[side]
	if reg.MomentumLabel == against {
		parts = append(parts, "momentum "+against)
	}
	if series.Finite(otherBadRatio) && otherBadRatio >= 1.15 {
		parts = append(parts, fmt.Sprintf("%.1f× the %s side's bad case", otherBadRatio, SideNames[1-side]))
	}
	if len(parts) == 0 {
		return "normal conditions"
	}
	return strings.Join(parts, ", ")
}

// WindowState is a window's fitted model, its walk-forward check and the forecast at the latest close.
type WindowState struct {
	Model      *Model
	Check      Check
	Record     regime.TrendRecord // set at training
	At         int64              // open time of the latest scored candle
	Swing      float64            // HAR forecast there (price fraction over the window)
	Regime     regime.Row
	FundingPct float64 // NaN without funding
	scored     bool
}

// Update forecasts at the data's latest usable candle.
func (s *WindowState) Update(data *WindowData) {
	if len(data.Usable) == 0 {
		return
	}
	pos := data.Usable[len(data.Usable)-1]
	s.Swing = s.Model.Har.Swing(&data.Inputs, []int{pos}, data.Window.Candles)[0]
	s.At = data.OpenTime[pos]
	s.Regime = data.Regime.At(pos)
	s.FundingPct = math.NaN()
	if col := data.Inputs.Col("funding_pct"); col != nil {
		s.FundingPct = col[pos]
	}
	s.scored = true
}

// Quantiles are a side's page quantiles as price moves.
type Quantiles struct {
	Alphas   []float64 `json:"alphas"`
	MovesPct []jsonx.F `json:"moves_pct"`
}

// SideView is one side's reading as the page reads it.
type SideView struct {
	Checked           bool      `json:"checked"`
	Coverage          *float64  `json:"coverage"`
	BadCasePct        jsonx.F   `json:"bad_case_pct"`
	BadCaseMarginPct  jsonx.F   `json:"bad_case_margin_pct"`
	StopHitChance     jsonx.F   `json:"stop_hit_chance"`
	Score             *int      `json:"score"`
	LiquidationChance jsonx.F   `json:"liquidation_chance"`
	Reason            string    `json:"reason"`
	Quantiles         Quantiles `json:"quantiles"`
}

// WindowView is one window's reading as the page reads it.
type WindowView struct {
	At            jsonx.Stamp         `json:"at"`
	SwingPct      jsonx.F             `json:"swing_pct"`
	Momentum      jsonx.F             `json:"momentum"`
	MomentumLabel jsonx.Label         `json:"momentum_label"`
	Efficiency    jsonx.F             `json:"efficiency"`
	VolPct        jsonx.F             `json:"vol_pct"`
	Volatility    jsonx.Label         `json:"volatility"`
	State         jsonx.Label         `json:"state"`
	TrendRecord   *regime.TrendRecord `json:"trend_record"`
	Long          SideView            `json:"long"`
	Short         SideView            `json:"short"`
}

// Side is the view of one side.
func (v *WindowView) Side(side int) *SideView {
	if side == Long {
		return &v.Long
	}
	return &v.Short
}

// View is the window's reading at a leverage, stop and liquidation distance (price fractions).
func (s *WindowState) View(leverage, stop, liquidation float64, spec Spec) WindowView {
	reg := s.Regime
	record := s.Record
	out := WindowView{
		At: jsonx.Stamp(s.At), SwingPct: jsonx.F(100.0 * s.Swing),
		Momentum: jsonx.F(reg.Momentum), MomentumLabel: jsonx.Label(reg.MomentumLabel), Efficiency: jsonx.F(reg.Efficiency),
		VolPct: jsonx.F(reg.VolPct), Volatility: jsonx.Label(reg.VolatilityLabel), State: jsonx.Label(reg.State),
		TrendRecord: &record,
	}
	var bad [2]float64
	for side := range 2 {
		bad[side] = s.Model.Quantiles(side, []float64{spec.BadCase})[0] * s.Swing
	}
	lo, hi := spec.CoverageBand[0], spec.CoverageBand[1]
	for side := range 2 {
		v := out.Side(side)
		if c := s.Check.Sides[side]; c != nil {
			coverage := c.Coverage
			v.Coverage = &coverage
			v.Checked = lo <= coverage && coverage <= hi
		}
		hit := s.Model.Chance(s.Swing, side, stop)
		otherRatio := math.NaN()
		if bad[1-side] > 0 {
			otherRatio = bad[side] / bad[1-side]
		}
		moves := s.Model.Quantiles(side, PageAlphas)
		for i := range moves {
			moves[i] = 100.0 * (moves[i] * s.Swing)
		}
		v.BadCasePct = jsonx.F(100.0 * bad[side])
		v.BadCaseMarginPct = jsonx.F(100.0 * bad[side] * leverage)
		v.StopHitChance = jsonx.F(hit)
		v.Score = Score(hit)
		v.LiquidationChance = jsonx.F(s.Model.Chance(s.Swing, side, liquidation))
		v.Reason = Reason(side, s.Swing/s.Model.TypicalSwing, s.FundingPct, reg, otherRatio)
		v.Quantiles = Quantiles{Alphas: PageAlphas, MovesPct: jsonx.Floats(moves)}
	}
	return out
}

// Coin is the risk models of every window of one coin: trained (with the walk-forward check)
// when the history loads and once a day, scored at every close of a window's chart. Not safe for
// concurrent use: the caller locks.
type Coin struct {
	Symbol, Benchmark string
	Spec              Spec
	RegimeSpec        regime.Spec
	Windows           map[string]*WindowState
	TrainedAt         time.Time
}

func (c *Coin) data(charts map[string]*candles.Chart, w regime.Window) *WindowData {
	if charts[w.Interval].Empty() {
		return nil
	}
	return NewWindowData(charts, c.Symbol, w, c.Benchmark, c.RegimeSpec)
}

// Train checks and fits every window the charts allow. stop and liquidation (price fractions)
// only shape the check's stop-hit diagnostics; progress(window, step, steps) may be nil.
func Train(symbol, benchmark string, charts map[string]*candles.Chart, stop, liquidation float64, spec Spec,
	regimeSpec regime.Spec, progress func(window string, step, steps int)) (*Coin, error) {
	c := &Coin{Symbol: symbol, Benchmark: benchmark, Spec: spec, RegimeSpec: regimeSpec, Windows: map[string]*WindowState{}}
	for i, w := range regime.Windows {
		if progress != nil {
			progress(w.Name, i+1, len(regime.Windows))
		}
		data := c.data(charts, w)
		if data == nil || len(data.Trainable) < MinTraining {
			continue
		}
		check, err := WalkForward(data, stop, liquidation, spec)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", w.Name, err)
		}
		model, err := Fit(data, data.Trainable, spec)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", w.Name, err)
		}
		state := &WindowState{Model: model, Check: check,
			Record: regime.ComputeTrendRecord(charts[w.Interval], symbol, w, &data.Regime, RoundTripPct)}
		state.Update(data)
		c.Windows[w.Name] = state
	}
	c.TrainedAt = time.Now().UTC()
	return c, nil
}

// Score forecasts at the latest close of the windows on the given charts (nil: every chart).
func (c *Coin) Score(charts map[string]*candles.Chart, intervals []string) {
	for _, w := range regime.Windows {
		state := c.Windows[w.Name]
		if state == nil || (intervals != nil && !contains(intervals, w.Interval)) {
			continue
		}
		if data := c.data(charts, w); data != nil {
			state.Update(data)
		}
	}
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// View is per window: momentum and state, and for each side the bad case, stop-hit and
// liquidation chances, score, reason, whether it passed the check, and the quantiles the page
// re-reads for another stop.
func (c *Coin) View(leverage, mmr, stopPct float64) map[string]WindowView {
	stop, liquidation := StopMove(stopPct, leverage), LiquidationMove(leverage, mmr)
	out := map[string]WindowView{}
	for name, state := range c.Windows {
		if state.scored {
			out[name] = state.View(leverage, stop, liquidation, c.Spec)
		}
	}
	return out
}

// Reading is today's numbers for one side (the risk command).
type Reading struct {
	QuantilesPct      []float64
	BadCasePct        float64
	BadCaseMarginPct  float64
	StopHitChance     float64
	LiquidationChance float64
	Score             *int
}

// Latest is today's numbers at the chart's last usable candle, per side: at, swing and readings.
func Latest(model *Model, data *WindowData, stop, liquidation, leverage float64) (at int64, swingPct float64, sides [2]Reading) {
	pos := data.Usable[len(data.Usable)-1]
	swing := model.Har.Swing(&data.Inputs, []int{pos}, data.Window.Candles)[0]
	bad := model.Spec.BadIndex()
	for side := range 2 {
		q := model.Quantiles(side, model.Spec.Alphas)
		pct := make([]float64, len(q))
		for i := range q {
			pct[i] = 100.0 * (q[i] * swing)
		}
		hit := model.Chance(swing, side, stop)
		sides[side] = Reading{QuantilesPct: pct, BadCasePct: pct[bad], BadCaseMarginPct: pct[bad] * leverage,
			StopHitChance: hit, LiquidationChance: model.Chance(swing, side, liquidation), Score: Score(hit)}
	}
	return data.OpenTime[pos], 100.0 * swing, sides
}
