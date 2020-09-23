package cnns

import (
	"gonum.org/v1/gonum/mat"
)

// ZeroPadding Apply zero padding to source matrix
/*
	matrix - source matrix
	num - number of columns to add and fill with zeroes
*/
func ZeroPadding(matrix *mat.Dense, num int) *mat.Dense {
	r, c := matrix.Dims()
	outRows := r + num*2
	outCols := c + num*2
	flattenMatrix := make([]float64, outRows*outCols)
	for y := 0; y < outRows; y++ {
		zeroPadding(matrix, flattenMatrix, r, c, outCols, num, y)
	}
	return mat.NewDense(outRows, outCols, flattenMatrix)
}

// zeroPadding See ZeroPadding()
func zeroPadding(matrix *mat.Dense, newFlattenMatrix []float64, rows, cols, outCols int, w, y int) {
	for x := 0; x < outCols; x++ {
		if y > w-1 && y < rows+w && x > w-1 && x < cols+w {
			newFlattenMatrix[y*outCols+x] = matrix.At(y-w, x-w)
		} else {
			newFlattenMatrix[y*outCols+x] = 0.0
		}
	}
}
