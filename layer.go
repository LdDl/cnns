package cnns

import (
	t "github.com/LdDl/cnns/tensor"
)

// Layer - interface for all layer types
type Layer interface {
	// OutSize - returns output size (dimensions)
	GetOutputSize() *t.TDsize

	// GetInputSize - returns input size (dimensions)
	GetInputSize() *t.TDsize

	// GetOutput - returns layer's output
	GetOutput() *t.Tensor

	// GetWeights - returns layer's weights
	GetWeights() []*t.Tensor

	// GetGradients - returns layer's gradients
	GetGradients() *t.Tensor

	// FeedForward - feed data to layer
	FeedForward(t *t.Tensor)

	// CalculateGradients - calculate layers' gradients
	CalculateGradients(nextLayerGradients *t.Tensor)

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
	SetCustomWeights(t []*t.Tensor)
}
