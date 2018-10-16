package nns

import (
	"fmt"
)

// ReLULayer is Rectified Linear Unit layer (activation: max(0, x))
type ReLULayer struct {
	In                    Tensor
	Out                   Tensor
	InputGradientsWeights Tensor
}

// NewReLULayer - constructor for new ReLU layer. You need to specify input size
func NewReLULayer(inSize TDsize) *LayerStruct {
	newLayer := &ReLULayer{
		InputGradientsWeights: NewTensor(inSize.X, inSize.Y, inSize.Z),
		In:  NewTensor(inSize.X, inSize.Y, inSize.Z),
		Out: NewTensor(inSize.X, inSize.Y, inSize.Z),
	}
	return &LayerStruct{
		Layer: newLayer,
	}
}

// SetCustomWeights - set user's weights (make it carefully)
func (relu *ReLULayer) SetCustomWeights(t *[]Tensor) {
	fmt.Println("There are no weights for ReLU layer")
}

// OutSize - returns output size (dimensions)
func (relu *ReLULayer) OutSize() Point {
	return (*relu).Out.Size
}

// GetInputSize - returns input size (dimensions)
func (relu *ReLULayer) GetInputSize() Point {
	return (*relu).In.Size
}

// GetOutput - returns ReLU layer's output
func (relu *ReLULayer) GetOutput() Tensor {
	return (*relu).Out
}

// GetWeights - returns ReLU layer's weights
func (relu *ReLULayer) GetWeights() []Tensor {
	fmt.Println("There are no weights for ReLU layer")
	return []Tensor{}
}

// GetGradients - returns ReLU layer's gradients
func (relu *ReLULayer) GetGradients() Tensor {
	return (*relu).InputGradientsWeights
}

// FeedForward - feed data to ReLU layer
func (relu *ReLULayer) FeedForward(t *Tensor) {
	(*relu).In = (*t)
	(*relu).DoActivation()
}

// DoActivation - ReLU layer's output activation
func (relu *ReLULayer) DoActivation() {
	// Rectify(relu.In, relu.Out)
	for i := 0; i < (*relu).In.Size.X; i++ {
		for j := 0; j < (*relu).In.Size.Y; j++ {
			for z := 0; z < (*relu).In.Size.Z; z++ {
				v := (*relu).In.Get(i, j, z)
				if v < 0 {
					v = 0
				}
				(*relu).Out.Set(i, j, z, v)
			}
		}
	}
}

// CalculateGradients - calculate ReLU layer's gradients
func (relu *ReLULayer) CalculateGradients(nextLayerGrad *Tensor) {
	for i := 0; i < (*relu).In.Size.X; i++ {
		for j := 0; j < (*relu).In.Size.Y; j++ {
			for z := 0; z < (*relu).In.Size.Z; z++ {
				if (*relu).In.Get(i, j, z) < 0 {
					(*relu).InputGradientsWeights.Set(i, j, z, 0)
				} else {
					(*relu).InputGradientsWeights.Set(i, j, z, 1.0*nextLayerGrad.Get(i, j, z))
				}
			}
		}
	}
}

// UpdateWeights - just to point, that ReLU layer does NOT updating weights
func (relu *ReLULayer) UpdateWeights() {
	/*
		Empty
		Need for layer interface.
	*/
}

// PrintOutput - print ReLU layer's output
func (relu *ReLULayer) PrintOutput() {
	fmt.Println("Printing ReLU Layer output...")
	(*relu).Out.Print()
}

// PrintWeights - just to point, that ReLU layer has not weights
func (relu *ReLULayer) PrintWeights() {
	fmt.Println("There are no weights for ReLU layer")
}

// PrintGradients - print relu layer's local gradients
func (relu *ReLULayer) PrintGradients() {
	fmt.Println("Printing ReLU Layer gradients...")
	(*relu).InputGradientsWeights.Print()
}

// SetActivationFunc - sets activation function for layer
func (relu *ReLULayer) SetActivationFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set activation function for ReLU layer")
}

// SetActivationDerivativeFunc sets derivative of activation function
func (relu *ReLULayer) SetActivationDerivativeFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set derivative of activation function for ReLU layer")
}

// GetStride - get stride of layer
func (relu *ReLULayer) GetStride() int {
	return 0
}

// GetKernelSize - return "conv" as layer's type
func (relu *ReLULayer) GetKernelSize() int {
	return 0
}

// GetType - return "relu" as layer's type
func (relu *ReLULayer) GetType() string {
	return "relu"
}
