package cnns

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestConvolve2D(t *testing.T) {

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
	greenChannel := mat.NewDense(imgRows, imgCols, []float64{
		1, 0, 0, 1, 0,
		2, 0, 1, 2, 0,
		3, 1, 1, 3, 0,
		0, 3, 0, 3, 2,
		1, 0, 3, 2, 1,
	})
	blueChannel := mat.NewDense(imgRows, imgCols, []float64{
		2, 0, 1, 2, 1,
		3, 3, 1, 3, 2,
		2, 1, 1, 1, 0,
		3, 1, 3, 2, 0,
		1, 1, 2, 1, 1,
	})

	kernel1R := mat.NewDense(kernelSize, kernelSize, []float64{
		0, 1, 0,
		0, 0, 2,
		0, 1, 0,
	})
	kernel1G := mat.NewDense(kernelSize, kernelSize, []float64{
		2, 1, 0,
		0, 0, 0,
		0, 3, 0,
	})
	kernel1B := mat.NewDense(kernelSize, kernelSize, []float64{
		1, 0, 0,
		1, 0, 0,
		0, 0, 2,
	})

	stride := 1
	channels := 3

	correct := [][]float64{
		[]float64{19, 13, 15},
		[]float64{28, 16, 20},
		[]float64{23, 18, 25},
	}

	img2 := &mat.Dense{}
	img2.Stack(redChannel, greenChannel)
	imageRGB := &mat.Dense{}
	imageRGB.Stack(img2, blueChannel)

	kernel1 := &mat.Dense{}
	kernel1.Stack(kernel1R, kernel1G)
	kernelRGB := &mat.Dense{}
	kernelRGB.Stack(kernel1, kernel1B)

	outMatrix, err := Convolve2D(imageRGB, kernelRGB, channels, stride)
	if err != nil {
		t.Error(err)
		return
	}

	rows, _ := outMatrix.Dims()
	if rows != len(correct) {
		t.Errorf("'rows' must be %d, but got %d after Convolve2D()", len(correct), rows)
		return
	}

	for r := 0; r < rows; r++ {
		rowView := outMatrix.RawRowView(r)
		if len(rowView) != len(correct[r]) {
			t.Errorf("'cols' in single row should be %d, but got %d after Convolve2D()", len(correct[r]), len(rowView))
			return
		}
		for c := range rowView {
			if rowView[c] != correct[r][c] {
				t.Errorf("Element in position r = %d, c = %d should be %f, but got %f", r, c, correct[r][c], rowView[c])
				return
			}
		}
	}

}
