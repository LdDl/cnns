package cnns

import (
	"sync"

	"gonum.org/v1/gonum/mat"
)

// Flatten Convert matrix to vector. Result is defined as NewDense(1, num of rows * num of cols, matrix data)
/*
	matrix - source matrix
*/
func Flatten(matrix *mat.Dense) *mat.Dense {
	height, width := matrix.Dims()
	numElements := height * width
	flattenMatrix := make([]float64, numElements)

	wg := sync.WaitGroup{}
	for row := 0; row < height; row++ {
		wg.Add(1)
		go flatten(matrix, flattenMatrix, row, width, &wg)
	}
	return mat.NewDense(1, numElements, flattenMatrix)
}

// flatten Indexing vector as matrix. See ref. Flatten()
func flatten(matrix *mat.Dense, flattenMatrix []float64, row, width int, wg *sync.WaitGroup) {
	for column := 0; column < width; column++ {
		flattenMatrix[row*width+column] = matrix.At(row, column)
	}
	wg.Done()
}
