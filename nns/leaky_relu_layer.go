package nns

import (
	"fmt"

	t "github.com/LdDl/cnns/nns/tensor"
)

// LeakyReLULayer - Rectified Linear Unit layer with leak.
/*
	In - Input data;
	Out - Output data;
	InputGradientsWeights - Incoming gradients*weights (backpropagation);
	alpha - In simple ReLU you have f(x) = max(x,0) as activation function,
	but in Leaky ReLU it is: f(x) = alpha*x (for x < 0) and f(x) = x (for x >= 0).
*/
type LeakyReLULayer struct {
	In                    t.Tensor
	Out                   t.Tensor
	InputGradientsWeights t.Tensor
	alpha                 float64
}

// NewLeakyReLULayer - Constructor for new Leaky ReLU layer. You need to specify input size
/*
	inSize - Input layer's size;
	alpha - Coefficient in activation function. Should small (for example 0.01).
*/
func NewLeakyReLULayer(inSize t.TDsize, alpha float64) *LayerStruct {
	newLayer := &LeakyReLULayer{
		InputGradientsWeights: t.NewTensor(inSize.X, inSize.Y, inSize.Z),
		In:    t.NewTensor(inSize.X, inSize.Y, inSize.Z),
		Out:   t.NewTensor(inSize.X, inSize.Y, inSize.Z),
		alpha: alpha,
	}
	return &LayerStruct{
		Layer: newLayer,
	}
}

// SetCustomWeights - Set user's weights (make it carefully)
func (lrelu *LeakyReLULayer) SetCustomWeights(t *[]t.Tensor) {
	fmt.Println("There are no weights for ReLU layer")
}

// OutSize - Return output size (dimensions)
func (lrelu *LeakyReLULayer) OutSize() t.Point {
	return (*lrelu).Out.Size
}

// GetInputSize - Return input size (dimensions)
func (lrelu *LeakyReLULayer) GetInputSize() t.Point {
	return (*lrelu).In.Size
}

// GetOutput - Return Leaky ReLU layer's output
func (lrelu *LeakyReLULayer) GetOutput() t.Tensor {
	return (*lrelu).Out
}

// GetWeights - Return Leaky ReLU layer's weights
func (lrelu *LeakyReLULayer) GetWeights() []t.Tensor {
	fmt.Println("There are no weights for ReLU layer")
	return []t.Tensor{}
}

// GetGradients - Return Leaky ReLU layer's gradients
func (lrelu *LeakyReLULayer) GetGradients() t.Tensor {
	return (*lrelu).InputGradientsWeights
}

// FeedForward - Feed data to Leaky ReLU layer
func (lrelu *LeakyReLULayer) FeedForward(t *t.Tensor) {
	(*lrelu).In = (*t)
	(*lrelu).DoActivation()
}

// DoActivation - Leaky ReLU layer's output activation
func (lrelu *LeakyReLULayer) DoActivation() {
	for i := 0; i < (*lrelu).In.Size.X; i++ {
		for j := 0; j < (*lrelu).In.Size.Y; j++ {
			for z := 0; z < (*lrelu).In.Size.Z; z++ {
				v := (*lrelu).In.Get(i, j, z)
				if v < 0 {
					v = (*lrelu).alpha * v
				}
				(*lrelu).Out.Set(i, j, z, v)
			}
		}
	}
}

// CalculateGradients - Calculate Leaky ReLU layer's gradients
func (lrelu *LeakyReLULayer) CalculateGradients(nextLayerGrad *t.Tensor) {
	for i := 0; i < (*lrelu).In.Size.X; i++ {
		for j := 0; j < (*lrelu).In.Size.Y; j++ {
			for z := 0; z < (*lrelu).In.Size.Z; z++ {
				if (*lrelu).In.Get(i, j, z) < 0 {
					(*lrelu).InputGradientsWeights.Set(i, j, z, (*lrelu).alpha)
				} else {
					(*lrelu).InputGradientsWeights.Set(i, j, z, 1.0*nextLayerGrad.Get(i, j, z))
				}
			}
		}
	}
}

// UpdateWeights - Just to point, that Leaky ReLU layer does NOT updating weights
func (lrelu *LeakyReLULayer) UpdateWeights() {
	/*
		Empty
		Need for layer interface.
	*/
}

// PrintOutput - Pretty print Leaky ReLU layer's output
func (lrelu *LeakyReLULayer) PrintOutput() {
	fmt.Println("Printing Leaky ReLU Layer output...")
	(*lrelu).Out.Print()
}

// PrintWeights - Just to point, that Leaky ReLU layer has not weights
func (lrelu *LeakyReLULayer) PrintWeights() {
	fmt.Println("There are no weights for Leaky ReLU layer")
}

// PrintGradients - Print Leaky ReLU layer's local gradients
func (lrelu *LeakyReLULayer) PrintGradients() {
	fmt.Println("Printing Leaky ReLU Layer gradients...")
	(*lrelu).InputGradientsWeights.Print()
}

// SetActivationFunc - Set activation function for layer
func (lrelu *LeakyReLULayer) SetActivationFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set activation function for Leaky ReLU layer")
}

// SetActivationDerivativeFunc - Set derivative of activation function
func (lrelu *LeakyReLULayer) SetActivationDerivativeFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set derivative of activation function for Leaky ReLU layer")
}

// GetStride - Return stride of layer
func (lrelu *LeakyReLULayer) GetStride() int {
	return 0
}

// GetKernelSize - Return kernel size
func (lrelu *LeakyReLULayer) GetKernelSize() int {
	return 0
}

// GetType - Return "leaky_relu" as layer's type
func (lrelu *LeakyReLULayer) GetType() string {
	return "leaky_relu"
}
