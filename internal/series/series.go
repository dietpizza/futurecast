// Package series holds the pandas and numpy operations the statistics need, on []float64 with
// NaN as missing, written to give pandas' answers: rolling windows by pandas' own online
// algorithms (Kahan-compensated sums, Welford variance, the repeated-value shortcut), ranks with
// ties averaged, EWM without adjustment, numpy's linear quantile. Each is tested against values
// pandas produced.
package series

import (
	"math"
	"sort"
)

var nan = math.NaN()

// NaNs is n NaNs.
func NaNs(n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = nan
	}
	return out
}

// Map applies f to every value.
func Map(x []float64, f func(float64) float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = f(v)
	}
	return out
}

// Zip applies f to x[i], y[i].
func Zip(x, y []float64, f func(a, b float64) float64) []float64 {
	out := make([]float64, len(x))
	for i := range x {
		out[i] = f(x[i], y[i])
	}
	return out
}

// Shift moves values k places later (k < 0: earlier), filling with NaN.
func Shift(x []float64, k int) []float64 {
	out := NaNs(len(x))
	for i := range x {
		if j := i - k; j >= 0 && j < len(x) {
			out[i] = x[j]
		}
	}
	return out
}

// Diff is x[i] - x[i-1] (NaN first).
func Diff(x []float64) []float64 {
	out := NaNs(len(x))
	for i := 1; i < len(x); i++ {
		out[i] = x[i] - x[i-1]
	}
	return out
}

// Where keeps x where cond holds and puts other elsewhere (pandas .where: a comparison with NaN is false).
func Where(x []float64, cond []bool, other float64) []float64 {
	out := make([]float64, len(x))
	for i := range x {
		if cond[i] {
			out[i] = x[i]
		} else {
			out[i] = other
		}
	}
	return out
}

// Clip bounds values to [lo, hi], leaving NaN alone.
func Clip(x []float64, lo, hi float64) []float64 {
	return Map(x, func(v float64) float64 {
		if v < lo {
			return lo
		}
		if v > hi {
			return hi
		}
		return v
	})
}

// Finite says whether v is neither NaN nor ±Inf.
func Finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// ───────── rolling windows ─────────

// window runs pandas' fixed-window loop: start = max(0, i+1-n), end = i+1. Over the first
// window (and whenever a window doesn't overlap the previous one, n = 1) the state is reset and
// filled; afterwards values leave and enter one at a time.
func window(n int, length int, reset func(first int), add func(j int), remove func(j int), out func(i int)) {
	for i := 0; i < length; i++ {
		s, e := max(0, i+1-n), i+1
		if i == 0 || s >= i { // s >= the previous window's end
			reset(s)
			for j := s; j < e; j++ {
				add(j)
			}
		} else {
			for j := max(0, i-n); j < s; j++ {
				remove(j)
			}
			add(i)
		}
		out(i)
	}
}

// sumState is pandas' roll_sum / roll_mean state.
type sumState struct {
	nobs, negCt, same     int
	sum, compAdd, compRem float64
	prev                  float64
}

func (s *sumState) add(v float64) {
	if math.IsNaN(v) {
		return
	}
	s.nobs++
	y := v - s.compAdd
	t := s.sum + y
	s.compAdd = t - s.sum - y
	s.sum = t
	if math.Signbit(v) {
		s.negCt++
	}
	if v == s.prev {
		s.same++
	} else {
		s.same = 1
	}
	s.prev = v
}

func (s *sumState) remove(v float64) {
	if math.IsNaN(v) {
		return
	}
	s.nobs--
	y := -v - s.compRem
	t := s.sum + y
	s.compRem = t - s.sum - y
	s.sum = t
	if math.Signbit(v) {
		s.negCt--
	}
}

// rolling runs the sum/mean state over x; each reset starts the repeated-value tracker at the
// window's first value, as pandas does.
func rolling(x []float64, n int, out func(i int, s *sumState)) {
	var s sumState
	window(n, len(x),
		func(first int) { s = sumState{prev: x[first]} },
		func(j int) { s.add(x[j]) },
		func(j int) { s.remove(x[j]) },
		func(i int) { out(i, &s) })
}

// RollingSum is x.rolling(n, min_periods=minPeriods).sum() (minPeriods 0 means n).
func RollingSum(x []float64, n, minPeriods int) []float64 {
	if minPeriods == 0 {
		minPeriods = n
	}
	out := make([]float64, len(x))
	rolling(x, n, func(i int, s *sumState) {
		switch {
		case s.nobs == 0 && minPeriods == 0:
			out[i] = 0
		case s.nobs >= minPeriods:
			if s.same >= s.nobs {
				out[i] = s.prev * float64(s.nobs)
			} else {
				out[i] = s.sum
			}
		default:
			out[i] = nan
		}
	})
	return out
}

// RollingMean is x.rolling(n, min_periods=minPeriods).mean() (minPeriods 0 means n).
func RollingMean(x []float64, n, minPeriods int) []float64 {
	if minPeriods == 0 {
		minPeriods = n
	}
	out := make([]float64, len(x))
	rolling(x, n, func(i int, s *sumState) {
		if s.nobs >= minPeriods && s.nobs > 0 {
			r := s.sum / float64(s.nobs)
			switch {
			case s.same >= s.nobs:
				r = s.prev
			case s.negCt == 0 && r < 0:
				r = 0
			case s.negCt == s.nobs && r > 0:
				r = 0
			}
			out[i] = r
		} else {
			out[i] = nan
		}
	})
	return out
}

// invCondTol flags a variance update as ill-conditioned (pandas: ε × 1e3, about 3 significant
// digits left), after which the window is recomputed from scratch.
var invCondTol = (math.Nextafter(1, 2) - 1) * 1e3

// varState is pandas 3's roll_var state (Welford with Kahan compensation).
type varState struct {
	nobs, mean, ssqdm float64
	compAdd, compRem  float64 // pandas keeps separate add and remove compensations
	unstable          bool
}

func (s *varState) add(v float64) {
	if math.IsNaN(v) {
		return
	}
	prevM2 := s.ssqdm
	s.nobs++
	prevMean := s.mean - s.compAdd
	y := v - s.compAdd
	t := y - s.mean
	s.compAdd = t + s.mean - y
	if s.nobs != 0 {
		s.mean = s.mean + t/s.nobs
	} else {
		s.mean = 0
	}
	s.ssqdm = s.ssqdm + float64((v-prevMean)*(v-s.mean)) // no FMA: see Quantile
	if float64(prevM2*invCondTol) > s.ssqdm {
		s.unstable = true // possible catastrophic cancellation
	}
}

func (s *varState) remove(v float64) {
	if math.IsNaN(v) {
		return
	}
	prevM2 := s.ssqdm
	s.nobs--
	if s.nobs != 0 {
		prevMean := s.mean - s.compRem
		y := v - s.compRem
		t := y - s.mean
		s.compRem = t + s.mean - y
		s.mean = s.mean - t/s.nobs
		s.ssqdm = s.ssqdm - float64((v-prevMean)*(v-s.mean))
		if float64(prevM2*invCondTol) > s.ssqdm {
			s.unstable = true
		}
	} else {
		s.mean, s.ssqdm, s.unstable = 0, 0, false
	}
}

// RollingStd is x.rolling(n, min_periods=minPeriods).std() with ddof = 1 (minPeriods 0 means
// n), by pandas 3's algorithm: Welford updates, and a fresh pass over the window whenever an
// update loses too many digits.
func RollingStd(x []float64, n, minPeriods int) []float64 {
	if minPeriods == 0 {
		minPeriods = n
	}
	minPeriods = max(minPeriods, 1)
	out := make([]float64, len(x))
	var st varState
	for i := range x {
		s, e := max(0, i+1-n), i+1
		recompute := i == 0 || s >= i
		if !recompute {
			for j := max(0, i-n); j < s; j++ {
				st.remove(x[j])
			}
			st.add(x[i])
		}
		if recompute || st.unstable {
			st = varState{}
			for j := s; j < e; j++ {
				st.add(x[j])
			}
			st.unstable = false
		}
		if st.nobs >= float64(minPeriods) && st.nobs > 1 {
			v := st.ssqdm / (st.nobs - 1)
			if v < 0 {
				v = 0 // zsqrt
			}
			out[i] = math.Sqrt(v)
		} else {
			out[i] = nan
		}
	}
	return out
}

// RollingMin is x.rolling(n).min(): NaN until n values, and wherever the window holds a NaN.
func RollingMin(x []float64, n int) []float64 {
	return rollingExtreme(x, n, func(a, b float64) bool { return a <= b })
}

// RollingMax is x.rolling(n).max().
func RollingMax(x []float64, n int) []float64 {
	return rollingExtreme(x, n, func(a, b float64) bool { return a >= b })
}

// rollingExtreme keeps a monotonic deque of positions (better first).
func rollingExtreme(x []float64, n int, better func(a, b float64) bool) []float64 {
	out := NaNs(len(x))
	var dq []int
	lastNaN := -1
	for i, v := range x {
		if math.IsNaN(v) {
			lastNaN = i
		} else {
			for len(dq) > 0 && better(v, x[dq[len(dq)-1]]) {
				dq = dq[:len(dq)-1]
			}
			dq = append(dq, i)
		}
		for len(dq) > 0 && dq[0] <= i-n {
			dq = dq[1:]
		}
		if i >= n-1 && lastNaN <= i-n && len(dq) > 0 {
			out[i] = x[dq[0]]
		}
	}
	return out
}

// RollingRankPct is x.rolling(n, min_periods=minPeriods).rank(pct=True): the latest value's
// rank in its window, ties averaged, over the window's non-NaN count; NaN where the value is NaN.
func RollingRankPct(x []float64, n, minPeriods int) []float64 {
	out := NaNs(len(x))
	// Coordinate-compress the values and count them in a Fenwick tree
	var vals []float64
	for _, v := range x {
		if !math.IsNaN(v) {
			vals = append(vals, v)
		}
	}
	sort.Float64s(vals)
	uniq := vals[:0:0]
	for i, v := range vals {
		if i == 0 || v != vals[i-1] {
			uniq = append(uniq, v)
		}
	}
	tree := make([]int, len(uniq)+1)
	addAt := func(k, d int) {
		for k++; k < len(tree); k += k & -k {
			tree[k] += d
		}
	}
	prefix := func(k int) int { // count of indexes < k
		s := 0
		for ; k > 0; k -= k & -k {
			s += tree[k]
		}
		return s
	}
	idx := func(v float64) int { return sort.SearchFloat64s(uniq, v) }
	nobs := 0
	for i, v := range x {
		if !math.IsNaN(v) {
			addAt(idx(v), 1)
			nobs++
		}
		if j := i - n; j >= 0 && !math.IsNaN(x[j]) {
			addAt(idx(x[j]), -1)
			nobs--
		}
		if math.IsNaN(v) || nobs < minPeriods || nobs == 0 {
			continue
		}
		k := idx(v)
		less := prefix(k)
		equal := prefix(k+1) - less
		rank := float64(less) + float64(equal+1)/2
		out[i] = rank / float64(nobs)
	}
	return out
}

// EWMMean is x.ewm(span=span, adjust=False).mean(): α = 2 / (span + 1).
func EWMMean(x []float64, span float64) []float64 { return EWMMeanCom(x, (span-1)/2) }

// EWMMeanAlpha is x.ewm(alpha=alpha, adjust=False).mean() (pandas turns α into a centre of mass).
func EWMMeanAlpha(x []float64, alpha float64) []float64 { return EWMMeanCom(x, (1-alpha)/alpha) }

// EWMMeanCom is x.ewm(com=com, adjust=False).mean(), seeded with the first value, computed as
// pandas does (α = 1 / (1 + com), normalising by the weights each step).
func EWMMeanCom(x []float64, com float64) []float64 {
	alpha := 1 / (1 + com)
	oldFactor, newWt := 1-alpha, alpha
	out := NaNs(len(x))
	weighted := nan
	oldWt := 1.0
	nobs := 0
	for i, cur := range x {
		obs := !math.IsNaN(cur)
		if obs {
			nobs++
		}
		if math.IsNaN(weighted) {
			if obs {
				weighted = cur
			}
		} else {
			oldWt *= oldFactor
			if obs {
				if weighted != cur {
					if com == 1 {
						newWt = 1 - oldWt // pandas: irregular-interval update, taken for span 3
					}
					weighted = math.FMA(oldWt, weighted, float64(newWt*cur)) / (oldWt + newWt) // pandas (clang, arm64) fuses this multiply-add
				}
				oldWt = 1
			}
		}
		if nobs > 0 {
			out[i] = weighted
		}
	}
	return out
}

// ───────── whole-array statistics ─────────

// Quantile is np.quantile(sorted, q) (linear) on ascending values without NaN.
func Quantile(sorted []float64, q float64) float64 {
	n := len(sorted)
	if n == 0 {
		return nan
	}
	v := float64(float64(n-1) * q) // rounded: Go may fuse it into v - lo below (FMA), across statements
	if v >= float64(n-1) {
		return sorted[n-1]
	}
	if v < 0 {
		return sorted[0]
	}
	lo := math.Floor(v)
	a, b := sorted[int(lo)], sorted[int(lo)+1]
	t := v - lo
	d := b - a
	// float64(...) rounds the product first: Go may otherwise fuse a multiply-add (FMA on arm64),
	// which rounds differently from numpy
	if t >= 0.5 {
		return b - float64(d*(1-t))
	}
	return a + float64(d*t)
}

// Median is np.median.
func Median(x []float64) float64 {
	s := append([]float64(nil), x...)
	sort.Float64s(s)
	n := len(s)
	if n == 0 {
		return nan
	}
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// Sum adds with numpy's pairwise summation (blocks of 8 below 128 values), so long sums round
// like numpy's.
func Sum(x []float64) float64 {
	n := len(x)
	if n < 8 {
		s := 0.0
		for _, v := range x {
			s += v
		}
		return s
	}
	if n <= 128 {
		var r [8]float64
		copy(r[:], x[:8])
		i := 8
		for ; i < n-n%8; i += 8 {
			for k := 0; k < 8; k++ {
				r[k] += x[i+k]
			}
		}
		s := ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
		for ; i < n; i++ {
			s += x[i]
		}
		return s
	}
	m := n / 2
	m -= m % 8
	return Sum(x[:m]) + Sum(x[m:])
}

// Mean is np.mean (NaN for no values).
func Mean(x []float64) float64 {
	if len(x) == 0 {
		return nan
	}
	return Sum(x) / float64(len(x))
}

// SearchLeft is np.searchsorted(sorted, v, side="left"): how many values are below v.
func SearchLeft(sorted []float64, v float64) int {
	return sort.Search(len(sorted), func(i int) bool { return sorted[i] >= v })
}
