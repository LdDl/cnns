package tensor

import "errors"

var (
	// ErrDimensionsAreNotEqual For Hadamard product
	ErrDimensionsAreNotEqual = errors.New("Tensors' dimensions are not equal")
	// ErrDimensionsNotFit For matrix multiplication
	ErrDimensionsNotFit = errors.New("Tensors have incompatible dimensions")
	// ErrKernelZAxis Kernel may have Z = 1 only (in convolution between matrix and kernel).
	ErrKernelZAxis = errors.New("Kernel size should be defined as (Any X, Any Y, 1)")
)
