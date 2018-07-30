package nns

// WholeNet - net itself (array of layers)
type WholeNet struct {
	Layers []Layer
}

// FeedForward - forward pass throught the net
func (wh *WholeNet) FeedForward(t *Tensor) {
	(*wh).Layers[0].FeedForward(t)
	for l := 1; l < len((*wh).Layers); l++ {
		out := (*wh).Layers[l-1].GetOutput()
		(*wh).Layers[l].FeedForward(&out)
	}
}

// Backpropagate - backward pass throught the net (training)
func (wh *WholeNet) Backpropagate(target *Tensor) {
	difference := (*wh).Layers[len((*wh).Layers)-1].GetOutput()
	difference.Sub(target)
	(*wh).Layers[len((*wh).Layers)-1].CalculateGradients(&difference)
	for i := len((*wh).Layers) - 2; i >= 0; i-- {
		grad := (*wh).Layers[i+1].GetGradients()
		(*wh).Layers[i].CalculateGradients(&grad)
	}
	for i := range (*wh).Layers {
		(*wh).Layers[i].UpdateWeights()
	}
}

// PrintOutput - prints net's output (last layer output)
func (wh *WholeNet) PrintOutput() {
	(*wh).Layers[len((*wh).Layers)-1].PrintOutput()
}

// GetOutput - returns net's output (last layer output)
func (wh *WholeNet) GetOutput() Tensor {
	return (*wh).Layers[len((*wh).Layers)-1].GetOutput()
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
