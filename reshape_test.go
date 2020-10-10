package cnns

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

var (
	reshapeExampleVec = mat.NewDense(12, 1, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	reshapeCorrect2D  = [][]float64{[]float64{1, 2, 3}, []float64{4, 5, 6}, []float64{7, 8, 9}, []float64{10, 11, 12}}
)

func TestReshape(t *testing.T) {
	reshaped, err := Reshape(reshapeExampleVec, 4, 3)
	if err != nil {
		t.Error(err)
		return
	}
	rows, _ := reshaped.Dims()
	if rows != len(reshapeCorrect2D) {
		t.Errorf("'rows' must be %d, but got %d after Reshape()", len(reshapeCorrect2D), rows)
		return
	}
	for r := 0; r < rows; r++ {
		rowView := reshaped.RawRowView(r)
		if len(rowView) != len(reshapeCorrect2D[r]) {
			t.Errorf("'cols' in single row should be %d, but got %d after Reshape()", len(reshapeCorrect2D[r]), len(rowView))
			return
		}
		for c := range rowView {
			if rowView[c] != reshapeCorrect2D[r][c] {
				t.Errorf("Element in position r = %d, c = %d should be %f, but got %f", r, c, reshapeCorrect2D[r][c], rowView[c])
				return
			}
		}
	}
}

func TestReshapeUnsafe(t *testing.T) {
	copyUnsafe := &mat.Dense{}
	copyUnsafe.CloneFrom(reshapeExampleVec)
	err := ReshapeUnsafe(copyUnsafe, 4, 3)
	if err != nil {
		t.Error(err)
		return
	}
	rows, _ := copyUnsafe.Dims()
	if rows != len(reshapeCorrect2D) {
		t.Errorf("'rows' must be %d, but got %d after Reshape()", len(reshapeCorrect2D), rows)
		return
	}
	for r := 0; r < rows; r++ {
		rowView := copyUnsafe.RawRowView(r)
		if len(rowView) != len(reshapeCorrect2D[r]) {
			t.Errorf("'cols' in single row should be %d, but got %d after Reshape()", len(reshapeCorrect2D[r]), len(rowView))
			return
		}
		for c := range rowView {
			if rowView[c] != reshapeCorrect2D[r][c] {
				t.Errorf("Element in position r = %d, c = %d should be %f, but got %f", r, c, reshapeCorrect2D[r][c], rowView[c])
				return
			}
		}
	}
}

func BenchmarkReshape(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := Reshape(reshapeExampleVec, 4, 3)
		if err != nil {
			b.Error(err)
			return
		}
	}
}

func BenchmarkReshapeUnsafe(b *testing.B) {
	copyUnsafe := &mat.Dense{}
	copyUnsafe.CloneFrom(reshapeExampleVec)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := ReshapeUnsafe(copyUnsafe, 4, 3)
		if err != nil {
			b.Error(err)
			return
		}
	}
}
