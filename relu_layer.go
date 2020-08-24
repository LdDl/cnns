package cnns

import (
	"fmt"

	t "github.com/LdDl/cnns/tensor"
)

// ReLULayer - Rectified Linear Unit layer (activation: max(0, x))
/*
	In - Input data
	Out - Output data
	LocalDelta - incoming gradients*weights (backpropagation)
*/
type ReLULayer struct {
	In         t.Tensor
	Out        t.Tensor
	LocalDelta t.Tensor
}

// NewReLULayer - Constructor for new ReLU layer. You need to specify input size
/*
	inSize - input layer's size
*/
func NewReLULayer(inSize *t.TDsize) *LayerStruct {
	newLayer := &ReLULayer{
		LocalDelta: t.NewTensor(inSize.X, inSize.Y, inSize.Z),
		In:         t.NewTensor(inSize.X, inSize.Y, inSize.Z),
		Out:        t.NewTensor(inSize.X, inSize.Y, inSize.Z),
	}
	return &LayerStruct{
		Layer: newLayer,
	}
}

// SetCustomWeights - Set user's weights (make it carefully)
func (relu *ReLULayer) SetCustomWeights(t *[]t.Tensor) {
	fmt.Println("There are no weights for ReLU layer")
}

// GetOutputSize - Return output size (dimensions)
func (relu *ReLULayer) GetOutputSize() *t.TDsize {
	return relu.Out.Size
}

// GetInputSize - Return input size (dimensions)
func (relu *ReLULayer) GetInputSize() *t.TDsize {
	return relu.In.Size
}

// GetOutput - Return ReLU layer's output
func (relu *ReLULayer) GetOutput() t.Tensor {
	return relu.Out
}

// GetWeights - Return ReLU layer's weights
func (relu *ReLULayer) GetWeights() []t.Tensor {
	fmt.Println("There are no weights for ReLU layer")
	return []t.Tensor{}
}

// GetGradients - Return ReLU layer's gradients
func (relu *ReLULayer) GetGradients() t.Tensor {
	return relu.LocalDelta
}

// FeedForward - Feed data to ReLU layer
func (relu *ReLULayer) FeedForward(t *t.Tensor) {
	relu.In = (*t)
	relu.DoActivation()
}

// DoActivation - ReLU layer's output activation
func (relu *ReLULayer) DoActivation() {
	for i := 0; i < relu.In.Size.X; i++ {
		for j := 0; j < relu.In.Size.Y; j++ {
			for z := 0; z < relu.In.Size.Z; z++ {
				v := relu.In.Get(i, j, z)
				if v < 0 {
					v = 0
				}
				relu.Out.Set(i, j, z, v)
			}
		}
	}
}

// CalculateGradients - Calculate ReLU layer's gradients
func (relu *ReLULayer) CalculateGradients(nextLayerGrad *t.Tensor) {
	for i := 0; i < relu.In.Size.X; i++ {
		for j := 0; j < relu.In.Size.Y; j++ {
			for z := 0; z < relu.In.Size.Z; z++ {
				if relu.In.Get(i, j, z) < 0 {
					relu.LocalDelta.Set(i, j, z, 0)
				} else {
					relu.LocalDelta.Set(i, j, z, 1.0*nextLayerGrad.Get(i, j, z))
				}
			}
		}
	}
}

// UpdateWeights - Just to point, that ReLU layer does NOT updating weights
func (relu *ReLULayer) UpdateWeights() {
	/*
		Empty
		Need for layer interface.
	*/
}

// PrintOutput - Pretty print ReLU layer's output
func (relu *ReLULayer) PrintOutput() {
	fmt.Println("Printing ReLU Layer output...")
	relu.Out.Print()
}

// PrintWeights - Just to point, that ReLU layer has not weights
func (relu *ReLULayer) PrintWeights() {
	fmt.Println("There are no weights for ReLU layer")
}

// PrintGradients - Print relu layer's local gradients
func (relu *ReLULayer) PrintGradients() {
	fmt.Println("Printing ReLU Layer gradients...")
	relu.LocalDelta.Print()
}

// SetActivationFunc - Set activation function for layer
func (relu *ReLULayer) SetActivationFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set activation function for ReLU layer")
}

// SetActivationDerivativeFunc - Set derivative of activation function
func (relu *ReLULayer) SetActivationDerivativeFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set derivative of activation function for ReLU layer")
}

// GetStride - Return stride of layer
func (relu *ReLULayer) GetStride() int {
	return 0
}

// GetKernelSize - Return "conv" as layer's type
func (relu *ReLULayer) GetKernelSize() int {
	return 0
}

// GetType - Return "relu" as layer's type
func (relu *ReLULayer) GetType() string {
	return "relu"
}
