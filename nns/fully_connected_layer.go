package nns

import (
	"fmt"
	"math"
	"math/rand"
)

// FullConnectedLayer is simple layer structure (so this layer can be used for simple neural networks like XOR problem), where
// In - input data
// Out - output data
// Weights - array of neurons
// NewGradients - new values of gradients
// Gradients - old values of gradients
// InputGradients - gradients
// ActivationFunc       - activation function. You can set custom func(v float64) float64, see SetActivationFunc
// ActivationDerivative - derivative of activation function. You can set custom func(v float64) float64, see SetActivationDerivativeFunc
type FullConnectedLayer struct {
	In                   *Tensor
	Out                  *Tensor
	Weights              *Tensor
	NewGradients         *Tensor
	Gradients            *Tensor
	InputGradients       *Tensor
	ActivationFunc       func(v float64) float64
	ActivationDerivative func(v float64) float64
}

// ActivationSygmoid is default ActivationFunc
func ActivationSygmoid(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1.0*v))
}

// ActivationSygmoidDerivative is default derivative of ActivationFunc (which is ActivationSygmoid)
func ActivationSygmoidDerivative(v float64) float64 {
	return ActivationSygmoid(v) * (1 - ActivationSygmoid(v))
}

// SetActivationFunc sets activation function for fully connected layer. You need to specify function: func(v float64) float64
func (fc *FullConnectedLayer) SetActivationFunc(f func(v float64) float64) {
	(*fc).ActivationFunc = f
}

// SetActivationDerivativeFunc sets derivative of activation function for fully connected layer. You need to specify function: func(v float64) float64
func (fc *FullConnectedLayer) SetActivationDerivativeFunc(f func(v float64) float64) {
	(*fc).ActivationDerivative = f
}

// NewFullConnectedLayer - constructor for new fully connected layer. You need to specify input size and output size
func NewFullConnectedLayer(width, height, depth int, outputSize int) *FullConnectedLayer {
	newLayer := &FullConnectedLayer{
		In:                   NewTensorEmpty(width, height, depth),
		Out:                  NewTensorEmpty(outputSize, 1, 1),
		Weights:              NewTensorEmpty(width*height*depth, outputSize, 1),
		Gradients:            NewTensorEmpty(outputSize, 1, 1),
		NewGradients:         NewTensorEmpty(outputSize, 1, 1),
		InputGradients:       NewTensorEmpty(outputSize, 1, 1),
		ActivationFunc:       ActivationSygmoid,           // Default Activation function is Sygmoid
		ActivationDerivative: ActivationSygmoidDerivative, // Default derivative of activation function is Sygmoid*(1-Sygmoid)
	}
	for i := 0; i < width*height*depth; i++ {
		for j := 0; j < outputSize; j++ {
			newLayer.Weights.SetValue(i, j, 0, rand.Float64())
		}
	}
	return newLayer
}

// PrintWeights - print fully connected layer's weights
func (fc *FullConnectedLayer) PrintWeights() {
	fmt.Println("Printing Fully Connected Layer kernels...")
	(*fc).Weights.Print()
}

// PrintOutput - print fully connected layer's output
func (fc *FullConnectedLayer) PrintOutput() {

}

// GetOutput - get fully connected layer's output
func (fc *FullConnectedLayer) GetOutput() *Tensor {
	return (*fc).Out
}

// FeedForward - feed data to fully connected layer
func (fc *FullConnectedLayer) FeedForward(t *Tensor) {
	(*fc).In = t
	(*fc).DoActivation()
}

// GetGradients - get fully connected layer's gradients
func (fc *FullConnectedLayer) GetGradients() *Tensor {
	return (*fc).InputGradients
}

// CalculateGradients - calculate fully connected layer's gradients
func (fc *FullConnectedLayer) CalculateGradients(nextLayerGrad *Tensor) {

}

// UpdateWeights - update fully connected layer's weights
func (fc *FullConnectedLayer) UpdateWeights() {

}

// DoActivation - fully connected layer's output activation
func (fc *FullConnectedLayer) DoActivation() {

}
