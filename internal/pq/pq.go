// Package pq reads and writes the flat Parquet files deltarisk keeps: one table of int64,
// float64, bool or string columns, with string key-value metadata (the caches' covered_from,
// covered_to and fetched_at). The files pyarrow wrote before the Go rewrite load as they are.
package pq

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet"
	"github.com/apache/arrow-go/v18/parquet/compress"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
)

// Column is one column; exactly one of the slices is set. Valid is nil when no value is null
// (a null float also reads as NaN, as pandas sees it).
type Column struct {
	Name    string
	Int64   []int64
	Float64 []float64
	Bool    []bool
	String  []string
	Valid   []bool
}

// Len is the column's row count.
func (c *Column) Len() int {
	switch {
	case c.Int64 != nil:
		return len(c.Int64)
	case c.Float64 != nil:
		return len(c.Float64)
	case c.Bool != nil:
		return len(c.Bool)
	default:
		return len(c.String)
	}
}

// IsNull says whether row i is null.
func (c *Column) IsNull(i int) bool { return c.Valid != nil && !c.Valid[i] }

// Floats is the column as float64 (ints converted, nulls NaN); nil for bool and string columns.
func (c *Column) Floats() []float64 {
	switch {
	case c.Float64 != nil:
		return c.Float64
	case c.Int64 != nil:
		out := make([]float64, len(c.Int64))
		for i, v := range c.Int64 {
			out[i] = float64(v)
			if c.IsNull(i) {
				out[i] = math.NaN()
			}
		}
		return out
	}
	return nil
}

// Table is a whole file.
type Table struct {
	Columns []Column
	Meta    map[string]string
}

// Col is the named column, or nil.
func (t *Table) Col(name string) *Column {
	for i := range t.Columns {
		if t.Columns[i].Name == name {
			return &t.Columns[i]
		}
	}
	return nil
}

// Rows is the table's row count.
func (t *Table) Rows() int {
	if len(t.Columns) == 0 {
		return 0
	}
	return t.Columns[0].Len()
}

// Read loads a whole file.
func Read(path string) (*Table, error) {
	rdr, err := file.OpenParquetFile(path, false)
	if err != nil {
		return nil, err
	}
	defer rdr.Close()
	meta := map[string]string{}
	kv := rdr.MetaData().KeyValueMetadata()
	for i, k := range kv.Keys() {
		meta[k] = kv.Values()[i]
	}
	fr, err := pqarrow.NewFileReader(rdr, pqarrow.ArrowReadProperties{}, memory.DefaultAllocator)
	if err != nil {
		return nil, err
	}
	tbl, err := fr.ReadTable(context.Background())
	if err != nil {
		return nil, err
	}
	defer tbl.Release()

	out := &Table{Meta: meta}
	for i := 0; i < int(tbl.NumCols()); i++ {
		col := tbl.Column(i)
		c := Column{Name: col.Name()}
		n := int(col.Len())
		nulls := col.NullN() > 0
		if nulls {
			c.Valid = make([]bool, 0, n)
		}
		switch col.DataType().ID() {
		case arrow.INT64, arrow.INT32:
			c.Int64 = make([]int64, 0, n)
		case arrow.FLOAT64, arrow.FLOAT32:
			c.Float64 = make([]float64, 0, n)
		case arrow.BOOL:
			c.Bool = make([]bool, 0, n)
		case arrow.STRING, arrow.LARGE_STRING:
			c.String = make([]string, 0, n)
		default:
			return nil, fmt.Errorf("%s: column %s has unsupported type %s", path, c.Name, col.DataType())
		}
		for _, chunk := range col.Data().Chunks() {
			for j := 0; j < chunk.Len(); j++ {
				null := chunk.IsNull(j)
				if nulls {
					c.Valid = append(c.Valid, !null)
				}
				switch a := chunk.(type) {
				case *array.Int64:
					c.Int64 = append(c.Int64, a.Value(j))
				case *array.Int32:
					c.Int64 = append(c.Int64, int64(a.Value(j)))
				case *array.Float64:
					v := a.Value(j)
					if null {
						v = math.NaN()
					}
					c.Float64 = append(c.Float64, v)
				case *array.Float32:
					v := float64(a.Value(j))
					if null {
						v = math.NaN()
					}
					c.Float64 = append(c.Float64, v)
				case *array.Boolean:
					c.Bool = append(c.Bool, a.Value(j))
				case *array.String:
					c.String = append(c.String, a.Value(j))
				case *array.LargeString:
					c.String = append(c.String, a.Value(j))
				}
			}
		}
		out.Columns = append(out.Columns, c)
	}
	return out, nil
}

// Write replaces the file at path with the columns (int64 or float64; NaN floats are stored as
// null, as pyarrow stores pandas' NaN) and the metadata, zstd-compressed. It writes a temp file
// first, so readers never see a half-written one.
func Write(path string, cols []Column, meta map[string]string) error {
	keys, values := make([]string, 0, len(meta)), make([]string, 0, len(meta))
	for k, v := range meta {
		keys, values = append(keys, k), append(values, v)
	}
	fields := make([]arrow.Field, len(cols))
	arrays := make([]arrow.Array, len(cols))
	rows := 0
	if len(cols) > 0 {
		rows = cols[0].Len()
	}
	defer func() {
		for _, a := range arrays {
			if a != nil {
				a.Release()
			}
		}
	}()
	for i, c := range cols {
		if c.Len() != rows {
			return fmt.Errorf("column %s has %d rows, want %d", c.Name, c.Len(), rows)
		}
		switch {
		case c.Int64 != nil || (c.Float64 == nil && c.Bool == nil && c.String == nil):
			b := array.NewInt64Builder(memory.DefaultAllocator)
			b.AppendValues(c.Int64, nil)
			fields[i], arrays[i] = arrow.Field{Name: c.Name, Type: arrow.PrimitiveTypes.Int64}, b.NewArray()
			b.Release()
		case c.Float64 != nil:
			b := array.NewFloat64Builder(memory.DefaultAllocator)
			for _, v := range c.Float64 {
				if math.IsNaN(v) {
					b.AppendNull()
				} else {
					b.Append(v)
				}
			}
			fields[i], arrays[i] = arrow.Field{Name: c.Name, Type: arrow.PrimitiveTypes.Float64, Nullable: true}, b.NewArray()
			b.Release()
		default:
			return fmt.Errorf("column %s: only int64 and float64 columns can be written", c.Name)
		}
	}
	md := arrow.NewMetadata(keys, values)
	schema := arrow.NewSchema(fields, &md)
	rec := array.NewRecordBatch(schema, arrays, int64(rows))
	defer rec.Release()

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name()) // no-op once renamed
	if err := tmp.Chmod(0o644); err != nil { // CreateTemp makes 0600; the caches were 0644 under Python
		tmp.Close()
		return err
	}
	props := parquet.NewWriterProperties(parquet.WithCompression(compress.Codecs.Zstd))
	w, err := pqarrow.NewFileWriter(schema, tmp, props, pqarrow.NewArrowWriterProperties(pqarrow.WithStoreSchema()))
	if err != nil {
		tmp.Close()
		return err
	}
	if err := w.Write(rec); err != nil {
		w.Close()
		return err
	}
	if err := w.Close(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
		return err
	}
	return os.Rename(tmp.Name(), path)
}
