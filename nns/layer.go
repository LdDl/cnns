package nns

// WholeNet - net itself (array of layers)
type WholeNet struct {
	Layers []Layer
}

// Layer - interface for all layer types
type Layer interface {
	// PrintWeights - print layer's weights
	PrintWeights()
	// PrintOutput - print layer's output
	PrintOutput()
	// GetOutput - get layer's output
	GetOutput() *Tensor
	// FeedForward - feed data to layer
	FeedForward(t *Tensor)
	// GetGradients - get layer's gradients
	GetGradients() *Tensor
	// CalculateGradients - calculate layers' gradients
	CalculateGradients(nextLayerGrad *Tensor)
	// UpdateWeights - update layer's weights
	UpdateWeights()
}

// LayerStruct - struct wraps layer interface
type LayerStruct struct {
	Layer
}
