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

// ExtractChannel Returns selected channel from N-channeled matrix (usefully in terms of image processing)
/*
	matrix - Source matrix
	rows - Number of rows (height)
	cols - Number of columns (width)
	channels - Number of channels
	channel - Index of channel
*/
func ExtractChannel(matrix *mat.Dense, rows, cols, channels, channel int) *mat.Dense {
	return matrix.Slice(channel*rows/channels, (channel+1)*rows/channels, 0, cols).(*mat.Dense)
}
