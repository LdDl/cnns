package cnns

import (
	"github.com/LdDl/cnns/tensor"
	"gonum.org/v1/gonum/mat"
)

// Layer Interface for all layer types
type Layer interface {
	// GetInputSize Returns dimensions of incoming data for layer
	GetInputSize() *tensor.TDsize

	// GetOutputSize Returns output size (dimensions) of layer
	GetOutputSize() *tensor.TDsize

	// GetActivatedOutput Returns activated layer's output
	GetActivatedOutput() *mat.Dense

	// GetWeights Returns layer's weights
	GetWeights() []*mat.Dense

	// GetGradients Returns layer's gradients dense
	GetGradients() *mat.Dense

	// FeedForward Feed data to layer
	FeedForward(input *mat.Dense) error

	// CalculateGradients Evaluate layers' gradients
	CalculateGradients(errorsDense *mat.Dense) error

	// UpdateWeights Call updating process for layer's weights
	UpdateWeights(lp *LearningParams)

	// PrintOutput Pretty print layer's output
	PrintOutput()

	// PrintWeights Pretty print layer's weights
	PrintWeights()

	// GetStride Returns stride of layer
	GetStride() int

	// GetType Returns type of layer in string representation
	GetType() string

	// SetActivationFunc Set activation function
	SetActivationFunc(f func(v float64) float64)

	// SetActivationDerivativeFunc Set derivative of activation function (for backpropagation)
	SetActivationDerivativeFunc(f func(v float64) float64)

	// SetCustomWeights Set provided data as layer's weights
	SetCustomWeights(weights []*mat.Dense)
}
