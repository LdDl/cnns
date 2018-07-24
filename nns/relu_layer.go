package nns

import (
	"fmt"
)

// ReLULayer is Rectified Linear Unit layer (activation: max(0, x))
// In - input data
// Out - output data
// GradsIn - gradients
type ReLULayer struct {
	In             *Tensor
	Out            *Tensor
	InputGradients *Tensor
}

// NewReLULayer - constructor for new ReLU layer. You need to specify input size
func NewReLULayer(width, height, depth int) *ReLULayer {
	newLayer := &ReLULayer{
		InputGradients: NewTensorEmpty(width, height, depth),
		In:             NewTensorEmpty(width, height, depth),
		Out:            NewTensorEmpty(width, height, depth),
	}
	return newLayer
}

// PrintWeights - just to point, that ReLU layer has not gradients
func (relu *ReLULayer) PrintWeights() {
	fmt.Println("No weights for ReLU")
}

// PrintOutput - print ReLU layer's output
func (relu *ReLULayer) PrintOutput() {
	fmt.Println("Printing ReLU Layer output...")
	(*relu).Out.Print()
}

// GetOutput - get ReLU layer's output
func (relu *ReLULayer) GetOutput() *Tensor {
	return (*relu).Out
}

// FeedForward - feed data to ReLU layer
func (relu *ReLULayer) FeedForward(t *Tensor) {
	(*relu).In = t
	(*relu).DoActivation()
}

// PrintGradients - print relu layer's gradients
func (relu *ReLULayer) PrintGradients() {
	fmt.Println("Printing ReLU Layer gradients-weights...")
	(*relu).InputGradients.Print()
}

// PrintSumGradWeights - print relu layer's summ of grad*weight
func (relu *ReLULayer) PrintSumGradWeights() {

}

// GetGradients - get ReLU layer's gradients
func (relu *ReLULayer) GetGradients() *Tensor {
	return (*relu).InputGradients
}

// CalculateGradients - calculate ReLU layer's gradients
func (relu *ReLULayer) CalculateGradients(nextLayerGrad *Tensor) {
	for i := 0; i < (*relu).In.X; i++ {
		for j := 0; j < (*relu).In.Y; j++ {
			for z := 0; z < (*relu).In.Z; z++ {
				if (*relu).In.GetValue(i, j, z) < 0 {
					// (*relu).InputGradients.SetValue(i, j, z, 0)
				} else {
					// (*relu).InputGradients.SetValue(i, j, z, nextLayerGrad.GetValue(i, j, z))
				}
				(*relu).InputGradients.SetValue(i, j, z, nextLayerGrad.GetValue(i, j, z))
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

// DoActivation - ReLU layer's output activation
func (relu *ReLULayer) DoActivation() {
	Rectify(relu.In, relu.Out)
}
