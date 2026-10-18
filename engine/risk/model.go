package risk

import (
	"fmt"
	"math"
	"sort"
	"sync"

	"gonum.org/v1/gonum/mat"

	"deltarisk/engine/catalog"
	"deltarisk/engine/regime"
	"deltarisk/engine/series"
)

// Sides.
const (
	Long  = 0
	Short = 1
)

// SideNames are the sides as the page names them.
var SideNames = [2]string{"long", "short"}

// ───────── tiers, stops and the score ─────────

// Tier is a leverage tier.
type Tier struct {
	Name     string
	Leverage float64
	MMR      float64 // maintenance margin rate (fixed: Binance's brackets need an authenticated call)
}

// Tiers by name.
var Tiers = map[string]Tier{
	"megacap":  {"megacap", 20, 0.004},
	"altcoin":  {"altcoin", 10, 0.005},
	"shitcoin": {"shitcoin", 3, 0.01},
}

// TierNames in the page's order.
var TierNames = []string{"megacap", "altcoin", "shitcoin"}

var megacaps = map[string]bool{"BTC": true, "ETH": true, "SOL": true, "BNB": true, "XRP": true}

// ShitcoinVolumeUSD is the 24h Binance quote volume below which a coin is a shitcoin.
const ShitcoinVolumeUSD = 50e6

// CoinTier is megacap (a fixed list), shitcoin (meme sector, or under $50M traded in 24h; volume
// nil when unknown), else altcoin.
func CoinTier(symbol string, volume24h *float64) string {
	base, _ := catalog.BaseQuote(symbol)
	if megacaps[base] {
		return "megacap"
	}
	if catalog.Sector(symbol) == "MEME" {
		return "shitcoin"
	}
	if volume24h != nil && *volume24h < ShitcoinVolumeUSD {
		return "shitcoin"
	}
	return "altcoin"
}

// StopMove is the price move (fraction) that loses stopMarginPct % of the margin at leverage.
func StopMove(stopMarginPct, leverage float64) float64 {
	return math.Abs(stopMarginPct) / 100.0 / leverage
}

// LiquidationMove is the isolated-margin liquidation distance as a price fraction: about 1/L
// minus the maintenance rate.
func LiquidationMove(leverage, mmr float64) float64 { return math.Max(1.0/leverage-mmr, 0) }

// Score is the stop-hit chance in percent, rounded (half to even, like Python); nil for NaN.
func Score(stopHitChance float64) *int {
	if !series.Finite(stopHitChance) {
		return nil
	}
	s := int(math.RoundToEven(100.0 * math.Min(math.Max(stopHitChance, 0), 1)))
	return &s
}

// ───────── the model ─────────

// Spec holds the model's settings.
type Spec struct {
	Alphas       []float64 // quantiles of the walk-forward check (pinball loss)
	BadCase      float64
	TrainShare   float64
	Folds        int
	GapHours     float64
	CoverageBand [2]float64
}

// DefaultSpec is the dashboard's.
var DefaultSpec = Spec{Alphas: []float64{0.25, 0.5, 0.75, 0.9, 0.97}, BadCase: 0.9, TrainShare: 0.6, Folds: 4,
	GapHours: 8, CoverageBand: [2]float64{0.07, 0.13}}

// BadIndex is the bad case's position in Alphas.
func (s Spec) BadIndex() int {
	for i, a := range s.Alphas {
		if a == s.BadCase {
			return i
		}
	}
	panic(fmt.Sprintf("bad case %g not among the alphas", s.BadCase))
}

// Gap is the candles between the last training reading and the first test reading.
func (s Spec) Gap(w regime.Window) int {
	return w.Candles + int(math.Ceil(s.GapHours*60*float64(w.Candles)/float64(w.Minutes())))
}

// ExceedChance is P(move ≥ x) from one row of predicted quantiles (any order): linear between
// (0, 0) and the quantiles, an exponential tail fitted to the last two beyond the last.
func ExceedChance(quantiles, alphas []float64, x float64) float64 {
	q := append([]float64(nil), quantiles...)
	sort.Float64s(q)
	n := len(q)
	if x >= q[n-1] {
		step := q[n-1] - q[n-2]
		scale := 1e-12
		if step > 0 {
			scale = step / math.Log((1-alphas[n-2])/(1-alphas[n-1]))
		}
		return (1 - alphas[n-1]) * math.Exp(-(x-q[n-1])/scale)
	}
	knots := append([]float64{0}, q...)
	levels := append([]float64{0}, alphas...)
	j := 0
	for _, k := range knots {
		if k < x {
			j++
		}
	}
	if j == 0 {
		return 1
	}
	lo, hi := knots[j-1], knots[j]
	cdf := levels[j-1] + float64((levels[j]-levels[j-1])*(x-lo))/(hi-lo)
	return 1.0 - cdf
}

// Har is a linear regression of the log future volatility on the HAR inputs.
type Har struct{ Coef []float64 }

func design(inputs *Frame, rows []int) *mat.Dense {
	X := mat.NewDense(len(rows), len(HARColumns)+1, nil)
	for k, name := range HARColumns {
		col := inputs.Col(name)
		for i, r := range rows {
			X.Set(i, 0, 1)
			X.Set(i, k+1, col[r])
		}
	}
	return X
}

// FitHar fits the regression on the rows (least squares through QR).
func FitHar(inputs *Frame, target []float64, rows []int) (Har, error) {
	X := design(inputs, rows)
	y := mat.NewVecDense(len(rows), nil)
	for i, r := range rows {
		y.SetVec(i, target[r])
	}
	var qr mat.QR
	qr.Factorize(X)
	var coef mat.VecDense
	if err := qr.SolveVecTo(&coef, false, y); err != nil {
		return Har{}, err
	}
	return Har{Coef: append([]float64(nil), coef.RawVector().Data...)}, nil
}

// Swing is the forecast volatility over the next h candles as a price fraction, per row.
func (m Har) Swing(inputs *Frame, rows []int, h int) []float64 {
	out := make([]float64, len(rows))
	cols := make([][]float64, len(HARColumns))
	for k, name := range HARColumns {
		cols[k] = inputs.Col(name)
	}
	for i, r := range rows {
		s := m.Coef[0]
		for k := range cols {
			s += float64(m.Coef[k+1] * cols[k][r])
		}
		out[i] = math.Exp(s) * math.Sqrt(float64(h))
	}
	return out
}

// Model is the HAR forecast and the training windows' adverse moves in swing units, per side.
type Model struct {
	Spec         Spec
	Har          Har
	Past         [2][]float64 // sorted
	TypicalSwing float64      // median HAR forecast over the training rows
}

// Fit trains a model on the data's rows.
func Fit(data *WindowData, rows []int, spec Spec) (*Model, error) {
	har, err := FitHar(&data.Inputs, data.FutureVol, rows)
	if err != nil {
		return nil, err
	}
	swing := har.Swing(&data.Inputs, rows, data.Window.Candles)
	m := &Model{Spec: spec, Har: har, TypicalSwing: series.Median(swing)}
	for side := range 2 {
		units := make([]float64, len(rows))
		for i, r := range rows {
			units[i] = data.Adverse[side][r] / swing[i]
		}
		sort.Float64s(units)
		m.Past[side] = units
	}
	return m, nil
}

// Quantiles are the side's adverse-move quantiles in swing units.
func (m *Model) Quantiles(side int, alphas []float64) []float64 {
	out := make([]float64, len(alphas))
	for i, a := range alphas {
		out[i] = series.Quantile(m.Past[side], a)
	}
	return out
}

// Chance is P(adverse move ≥ move, a price fraction) for a row with this swing.
func (m *Model) Chance(swing float64, side int, move float64) float64 {
	past := m.Past[side]
	return 1.0 - float64(series.SearchLeft(past, move/swing))/float64(len(past))
}

// ───────── the honesty check ─────────

// Pinball is the mean pinball loss over rows and quantiles (q is rows × alphas, row-major).
func Pinball(actual []float64, q [][]float64, alphas []float64) float64 {
	losses := make([]float64, 0, len(actual)*len(alphas))
	for i, y := range actual {
		for k, a := range alphas {
			d := y - q[i][k]
			losses = append(losses, math.Max(a*d, (a-1.0)*d))
		}
	}
	return series.Mean(losses)
}

// Bucket is one calibration bucket: predicted chance against how often it happened.
type Bucket struct {
	From      float64 `json:"from"`
	To        float64 `json:"to"`
	N         int     `json:"n"`
	Predicted float64 `json:"predicted"`
	Actual    float64 `json:"actual"`
}

// Calibration buckets the predictions by tenths (np.linspace(0, 1, 11) edges).
func Calibration(predicted []float64, happened []bool) []Bucket {
	var out []Bucket
	for b := 0; b < 10; b++ {
		lo, hi := float64(b)*0.1, float64(b+1)*0.1
		if b == 9 {
			hi = 1.0
		}
		var p, a []float64
		for i, v := range predicted {
			if v >= lo && (v < hi || hi >= 1.0) {
				p = append(p, v)
				if happened[i] {
					a = append(a, 1)
				} else {
					a = append(a, 0)
				}
			}
		}
		if len(p) > 0 {
			out = append(out, Bucket{lo, hi, len(p), series.Mean(p), series.Mean(a)})
		}
	}
	return out
}

// SplitPositions are (train, test) positions per fold: test blocks after the first TrainShare,
// each trained on every earlier position at least gap candles before the block starts.
func SplitPositions(trainable []int, spec Spec, gap int) [][2][]int {
	n := len(trainable)
	start := int(float64(n) * spec.TrainShare)
	block := (n - start) / spec.Folds
	var folds [][2][]int
	for i := 0; i < spec.Folds; i++ {
		test := trainable[start+i*block : start+(i+1)*block]
		if len(test) == 0 {
			continue
		}
		var train []int
		for _, p := range trainable {
			if p < test[0]-gap {
				train = append(train, p)
			}
		}
		if len(train) > 0 {
			folds = append(folds, [2][]int{train, test})
		}
	}
	return folds
}

// SideCheck is the walk-forward result for one side.
type SideCheck struct {
	Coverage             float64  `json:"coverage"`
	PinballPct           float64  `json:"pinball_pct"`
	StopHitPredicted     float64  `json:"stop_hit_predicted"`
	Brier                float64  `json:"brier"`
	LiquidationPredicted float64  `json:"liquidation_predicted"`
	Checked              bool     `json:"checked"`
	StopHitRate          float64  `json:"stop_hit_rate"`
	LiquidationRate      float64  `json:"liquidation_rate"`
	Calibration          []Bucket `json:"calibration"`
}

// Check is the walk-forward result for a window; Sides are nil without folds.
type Check struct {
	Tests, Folds, TrainFrom int
	Sides                   [2]*SideCheck
}

type foldResult struct {
	actual, hit, liq [2][]float64
	quant            [2][][]float64
	err              error
}

// WalkForward is the out-of-sample check per side: coverage of the bad case, pinball loss, the
// stop-hit chance against what happened. Folds run in parallel.
func WalkForward(data *WindowData, stop, liquidation float64, spec Spec) (Check, error) {
	folds := SplitPositions(data.Trainable, spec, spec.Gap(data.Window))
	if len(folds) == 0 {
		return Check{}, nil
	}
	results := make([]foldResult, len(folds))
	var wg sync.WaitGroup
	for f, fold := range folds {
		wg.Add(1)
		go func(f int, train, test []int) {
			defer wg.Done()
			res := &results[f]
			model, err := Fit(data, train, spec)
			if err != nil {
				res.err = err
				return
			}
			swing := model.Har.Swing(&data.Inputs, test, data.Window.Candles)
			for side := range 2 {
				q := model.Quantiles(side, spec.Alphas)
				for i, r := range test {
					res.actual[side] = append(res.actual[side], data.Adverse[side][r])
					row := make([]float64, len(q))
					for k := range q {
						row[k] = q[k] * swing[i]
					}
					res.quant[side] = append(res.quant[side], row)
					res.hit[side] = append(res.hit[side], model.Chance(swing[i], side, stop))
					res.liq[side] = append(res.liq[side], model.Chance(swing[i], side, liquidation))
				}
			}
		}(f, fold[0], fold[1])
	}
	wg.Wait()

	out := Check{Folds: len(folds), TrainFrom: len(folds[0][0])}
	for f, fold := range folds {
		if results[f].err != nil {
			return Check{}, results[f].err
		}
		out.Tests += len(fold[1])
	}
	lo, hi := spec.CoverageBand[0], spec.CoverageBand[1]
	bad := spec.BadIndex()
	for side := range 2 {
		var y, p, l []float64
		var q [][]float64
		for f := range folds {
			y = append(y, results[f].actual[side]...)
			q = append(q, results[f].quant[side]...)
			p = append(p, results[f].hit[side]...)
			l = append(l, results[f].liq[side]...)
		}
		exceeded, hitStop, hitLiq, brier := make([]float64, len(y)), make([]bool, len(y)), make([]float64, len(y)), make([]float64, len(y))
		stopRate := make([]float64, len(y))
		for i := range y {
			if y[i] > q[i][bad] {
				exceeded[i] = 1
			}
			hitStop[i] = y[i] >= stop
			hs := 0.0
			if hitStop[i] {
				hs = 1
			}
			stopRate[i] = hs
			if y[i] >= liquidation {
				hitLiq[i] = 1
			}
			d := p[i] - hs
			brier[i] = d * d
		}
		coverage := series.Mean(exceeded)
		out.Sides[side] = &SideCheck{
			Coverage: coverage, PinballPct: 100.0 * Pinball(y, q, spec.Alphas), StopHitPredicted: series.Mean(p),
			Brier: series.Mean(brier), LiquidationPredicted: series.Mean(l), Checked: lo <= coverage && coverage <= hi,
			StopHitRate: series.Mean(stopRate), LiquidationRate: series.Mean(hitLiq), Calibration: Calibration(p, hitStop),
		}
	}
	return out, nil
}
