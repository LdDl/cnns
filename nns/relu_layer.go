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
func NewReLULayer(inSize TDsize) *ReLULayer {
	newLayer := &ReLULayer{
		InputGradientsWeights: NewTensor(inSize.X, inSize.Y, inSize.Z),
		In:  NewTensor(inSize.X, inSize.Y, inSize.Z),
		Out: NewTensor(inSize.X, inSize.Y, inSize.Z),
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
func (relu *ReLULayer) GetOutput() Tensor {
	return (*relu).Out
}

// FeedForward - feed data to ReLU layer
func (relu *ReLULayer) FeedForward(t *Tensor) {
	(*relu).In = (*t)
	(*relu).DoActivation()
}

// PrintGradients - print relu layer's gradients
func (relu *ReLULayer) PrintGradients() {
	fmt.Println("Printing ReLU Layer gradients-weights...")
	(*relu).InputGradientsWeights.Print()
}

// PrintSumGradWeights - print relu layer's summ of grad*weight
func (relu *ReLULayer) PrintSumGradWeights() {

}

// GetGradients - get ReLU layer's gradients
func (relu *ReLULayer) GetGradients() Tensor {
	return (*relu).InputGradientsWeights
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
