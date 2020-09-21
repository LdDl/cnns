package cnns

import "gonum.org/v1/gonum/mat"

// Reshape Reshape matrix to given r - rows(height) and c - cols(width)
/*
	Product of target dimensions should be equal to product of source dimensions
*/
func Reshape(matrix *mat.Dense, r, c int) (*mat.Dense, error) {
	matrixR, matrixC := matrix.Dims()
	if matrixR*matrixC != r*c {
		return nil, ErrDimensionsAreNotEqual
	}
	row := 0
	col := 0
	newMat := make([]float64, r*c)
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			if col > matrixC-1 {
				row++
				col = 0
			}
			newMat[y*c+x] = matrix.At(row, col)
			col++
		}
	}
	return mat.NewDense(r, c, newMat), nil
}
