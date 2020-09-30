package cnns

import "fmt"

var (
	// ErrDimensionsAreNotEqual When matrix1.Dims() != matrix2.Dims()
	ErrDimensionsAreNotEqual = fmt.Errorf("Dimensions are not equal")
)
