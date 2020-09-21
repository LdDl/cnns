package cnns

import "errors"

var (
	// ErrDimensionsAreNotEqual When matrix1.Dims() != matrix2.Dims()
	ErrDimensionsAreNotEqual = errors.New("Dimensions are not equal")
)
