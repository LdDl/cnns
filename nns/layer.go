package nns

// WholeNet - net itself (array of layers)
type WholeNet struct {
	Layers []Layer
}

// Layer - interface for all layer types
type Layer interface {
	// OutSize - returns output size (dimensions)
	OutSize() Point

	// GetOutput - returns layer's output
	GetOutput() Tensor

	// GetWeights - returns layer's weights
	GetWeights() []Tensor

	// GetGradients - returns layer's gradients
	GetGradients() Tensor

	// FeedForward - feed data to layer
	FeedForward(t *Tensor)

	// CalculateGradients - calculate layers' gradients
	CalculateGradients(nextLayerGradients *Tensor)

	// UpdateWeights - update layer's weights
	UpdateWeights()

	// PrintOutput - print layer's output
	PrintOutput()

	// PrintWeights - print layer's weights
	PrintWeights()

	// PrintGradients - print layer's gradients
	PrintGradients()

	SetActivationFunc(f func(v float64) float64)
	SetActivationDerivativeFunc(f func(v float64) float64)
	SetCustomWeights(t *[]Tensor)
}

// LayerStruct - struct wraps layer interface
type LayerStruct struct {
	Layer
}
