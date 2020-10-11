package cnns

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestIm2Col(t *testing.T) {
	imgRows := 5
	imgCols := 5
	kernelSize := 3
	redChannel := mat.NewDense(imgRows, imgCols, []float64{
		1, 0, 1, 0, 2,
		1, 1, 3, 2, 1,
		1, 1, 0, 1, 1,
		2, 3, 2, 1, 3,
		0, 2, 0, 1, 0,
	})

	correct := []float64{
		1, 0, 1, 1, 1, 3, 1, 1, 0,
		0, 1, 0, 1, 3, 2, 1, 0, 1,
		1, 0, 2, 3, 2, 1, 0, 1, 1,
		1, 1, 3, 1, 1, 0, 2, 3, 2,
		1, 3, 2, 1, 0, 1, 3, 2, 1,
		3, 2, 1, 0, 1, 1, 2, 1, 3,
		1, 1, 0, 2, 3, 2, 0, 2, 0,
		1, 0, 1, 3, 2, 1, 2, 0, 1,
		0, 1, 1, 2, 1, 3, 0, 1, 0,
	}
	stride := 1
	im2Col := Im2Col(redChannel, kernelSize, kernelSize, stride)
	raw := im2Col.RawMatrix().Data
	for i := range correct {
		if correct[i] != raw[i] {
			t.Errorf("Element in position i = %d should be %f, but got %f", i, correct[i], raw[i])
			return
		}
	}
}
