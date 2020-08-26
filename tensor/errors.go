package tensor

import "errors"

var (
	// ErrDimensionsAreNotEqual For Hadamard product
	ErrDimensionsAreNotEqual = errors.New("Tensors' dimensions are not equal")
	// ErrDimensionsNotFit For matrix multiplication
	ErrDimensionsNotFit = errors.New("Tensors' dimensions doesn't fit each other")
)
