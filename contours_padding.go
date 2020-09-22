package cnns

import (
	"gonum.org/v1/gonum/mat"
)

// ContoursPadding Apply edge padding to source matrix
/*
	matrix - source matrix
	num - number of columns to add and fill with edge values
*/
func ContoursPadding(matrix *mat.Dense, num int) *mat.Dense {
	r, c := matrix.Dims()
	outRows := r + num*2
	outCols := c + num*2
	flattenMatrix := make([]float64, outRows*outCols)
	for y := 0; y < outRows; y++ {
		contoursPadding(matrix, flattenMatrix, r, c, outCols, num, num-1, y)
	}
	return mat.NewDense(outRows, outCols, flattenMatrix)
}

// contoursPadding See ContoursPadding()
func contoursPadding(matrix *mat.Dense, newFlattenMatrix []float64, rows, cols, outCols int, w, wprev, y int) {
	for x := 0; x < outCols; x++ {
		if y < w && x < w {
			newFlattenMatrix[y*outCols+x] = matrix.At(0, 0)
		} else if y < w && x > wprev && x < cols+w {
			newFlattenMatrix[y*outCols+x] = matrix.At(0, x-w)
		} else if y < w && x > cols+wprev {
			newFlattenMatrix[y*outCols+x] = matrix.At(0, cols-1)
		} else if y > wprev && y < rows+w && x < w {
			newFlattenMatrix[y*outCols+x] = matrix.At(y-w, 0)
		} else if y > wprev && y < rows+w && x > cols+wprev {
			newFlattenMatrix[y*outCols+x] = matrix.At(y-w, cols-1)
		} else if y > rows+wprev && x < w {
			newFlattenMatrix[y*outCols+x] = matrix.At(rows-1, 0)
		} else if y > rows+wprev && x > wprev && x < cols+w {
			newFlattenMatrix[y*outCols+x] = matrix.At(rows-1, x-w)
		} else if y > rows+wprev && x > cols+wprev {
			newFlattenMatrix[y*outCols+x] = matrix.At(rows-1, cols-1)
		} else {
			newFlattenMatrix[y*outCols+x] = matrix.At(y-w, x-w)
		}
	}
}
