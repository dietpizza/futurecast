// Package jsonx has the JSON value types the page's snapshot needs, written the way the Python
// app wrote them: NaN and ±Inf as null, a missing label as null, timestamps as pandas prints them.
package jsonx

import (
	"encoding/json"
	"math"
	"strconv"
	"time"
)

// F is a float64 that writes NaN and ±Inf as null (and reads null as NaN).
type F float64

func (f F) MarshalJSON() ([]byte, error) {
	v := float64(f)
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return []byte("null"), nil
	}
	return json.Marshal(v)
}

func (f *F) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		*f = F(math.NaN())
		return nil
	}
	var v float64
	err := json.Unmarshal(b, &v)
	*f = F(v)
	return err
}

// Floats converts a slice.
func Floats(v []float64) []F {
	out := make([]F, len(v))
	for i, x := range v {
		out[i] = F(x)
	}
	return out
}

// Label is a string that writes "" as null.
type Label string

func (l Label) MarshalJSON() ([]byte, error) {
	if l == "" {
		return []byte("null"), nil
	}
	return json.Marshal(string(l))
}

func (l *Label) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		*l = ""
		return nil
	}
	var s string
	err := json.Unmarshal(b, &s)
	*l = Label(s)
	return err
}

// Stamp is a time in ms written as pandas prints a UTC timestamp ("2026-10-17 19:00:00+00:00").
type Stamp int64

func (s Stamp) String() string {
	return time.UnixMilli(int64(s)).UTC().Format("2006-01-02 15:04:05+00:00")
}

func (s Stamp) MarshalJSON() ([]byte, error) { return []byte(strconv.Quote(s.String())), nil }
