package tensor

import (
	"fmt"
)

var (
	// ErrDimensionsAreNotEqual For Hadamard product
	ErrDimensionsAreNotEqual = fmt.Errorf("Tensors' dimensions are not equal")
	// ErrDimensionsNotFit For matrix multiplication
	ErrDimensionsNotFit = fmt.Errorf("Tensors have incompatible dimensions")
	// ErrKernelZAxis Kernel may have Z = 1 only (in convolution between matrix and kernel).
	ErrKernelZAxis = fmt.Errorf("Kernel size should be defined as (Any X, Any Y, 1)")
)
