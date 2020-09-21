package cnns

import "gonum.org/v1/gonum/mat"

// Rot2D90 Rotate tensor (2d component) by 90 degrees.
/*
	matrix - source matrix
	times - [optional] number of times to rotate matrix by 90 degrees
*/
func Rot2D90(matrix *mat.Dense, times ...int) *mat.Dense {
	rows, cols := matrix.Dims()
	ret := mat.NewDense(rows, cols, nil)
	ret.CloneFrom(matrix)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			ret.Set(x, y, matrix.At(cols-y-1, x))
		}
	}
	if len(times) != 0 && times[0] > 1 {
		for t := 0; t < times[0]-1; t++ {
			ret = Rot2D90(ret)
		}
	}
	return ret
}

// Rot2D180 Rotate matrix (2d component) by 180 degrees.
/*
	matrix - source matrix
*/
func Rot2D180(matrix *mat.Dense) *mat.Dense {
	return Rot2D90(matrix, 2)
}

// Rot2D270 Rotate matrix (2d component) by 270 degrees.
/*
	matrix - source matrix
*/
func Rot2D270(matrix *mat.Dense) *mat.Dense {
	return Rot2D90(matrix, 3)
}
