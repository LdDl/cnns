package cnns

import (
	"github.com/LdDl/cnns/tensor"
)

// Layer - interface for all layer types
type Layer interface {
	// OutSize - returns output size (dimensions)
	GetOutputSize() *tensor.TDsize

	// GetInputSize - returns input size (dimensions)
	GetInputSize() *tensor.TDsize

	// GetOutput - returns layer's output
	GetOutput() *tensor.Tensor

	// GetWeights - returns layer's weights
	GetWeights() []*tensor.Tensor

	// GetGradients - returns layer's gradients
	GetGradients() *tensor.Tensor

	// FeedForward - feed data to layer
	FeedForward(t *tensor.Tensor)

	// CalculateGradients - calculate layers' gradients
	CalculateGradients(nextLayerGradients *tensor.Tensor)

	// UpdateWeights - update layer's weights
	UpdateWeights()

	// PrintOutput - print layer's output
	PrintOutput()

	// PrintWeights - print layer's weights
	PrintWeights()

	// PrintGradients - print layer's gradients
	PrintGradients()

	// GetStride - get stride of layer
	GetStride() int

	// GetKernelSize - get kernel size of layer
	GetKernelSize() int

	// GetType - get type of layer
	GetType() string

	SetActivationFunc(f func(v float64) float64)
	SetActivationDerivativeFunc(f func(v float64) float64)
	SetCustomWeights(t []*tensor.Tensor)
}
