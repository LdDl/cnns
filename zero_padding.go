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
	newRows := r + num*2
	newCols := c + num*2
	flattenMatrix := make([]float64, newRows*newCols)
	wg := sync.WaitGroup{}
	for y := 0; y < newRows; y++ {
		wg.Add(1)
		go zeroPadding(matrix, r, c, newRows, newCols, flattenMatrix, num, y, &wg)
	}
	wg.Wait()
	return mat.NewDense(newRows, newCols, flattenMatrix)
}

// zeroPadding See ZeroPadding()
func zeroPadding(matrix *mat.Dense, rows, cols, newRows, newCols int, newFlattenMatrix []float64, w, y int, wg *sync.WaitGroup) {
	for x := 0; x < newCols; x++ {
		if y > w-1 && y < rows+w && x > w-1 && x < cols+w {
			newFlattenMatrix[y*newCols+x] = matrix.At(y-w, x-w)
		} else {
			newFlattenMatrix[y*newCols+x] = 0.0
		}
	}
	wg.Done()
}
