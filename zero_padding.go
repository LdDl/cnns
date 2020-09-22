package cnns

import (
	"sync"

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
	wg := sync.WaitGroup{}
	for y := 0; y < outRows; y++ {
		wg.Add(1)
		go zeroPadding(matrix, flattenMatrix, r, c, outCols, num, y, &wg)
	}
	wg.Wait()
	return mat.NewDense(outRows, outCols, flattenMatrix)
}

// zeroPadding See ZeroPadding()
func zeroPadding(matrix *mat.Dense, newFlattenMatrix []float64, rows, cols, outCols int, w, y int, wg *sync.WaitGroup) {
	for x := 0; x < outCols; x++ {
		if y > w-1 && y < rows+w && x > w-1 && x < cols+w {
			newFlattenMatrix[y*outCols+x] = matrix.At(y-w, x-w)
		} else {
			newFlattenMatrix[y*outCols+x] = 0.0
		}
	}
	wg.Done()
}
