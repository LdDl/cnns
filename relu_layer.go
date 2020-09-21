package cnns

import (
	"fmt"

	"github.com/LdDl/cnns/tensor"
	"gonum.org/v1/gonum/mat"
)

// ReLULayer Rectified Linear Unit layer (activation: max(0, x))
/*
	Oj - Input data
	Ok - Output data
	LocalDelta - Incoming gradients*weights (backpropagation)
*/
type ReLULayer struct {
	Oj         *mat.Dense
	Ok         *mat.Dense
	LocalDelta *mat.Dense

	OutputSize *tensor.TDsize
	trainMode  bool
}

// NewReLULayer - Constructor for new ReLU layer. You need to specify input size
/*
	inSize - input layer's size
*/
func NewReLULayer(inSize *tensor.TDsize) Layer {
	newLayer := &ReLULayer{
		Oj:         mat.NewDense(inSize.X*inSize.Z, inSize.Y, nil),
		Ok:         mat.NewDense(inSize.X*inSize.Z, inSize.Y, nil),
		LocalDelta: mat.NewDense(inSize.X*inSize.Z, inSize.Y, nil),
		OutputSize: &tensor.TDsize{X: inSize.X, Y: inSize.Y, Z: inSize.Z},
		trainMode:  false,
	}
	return newLayer
}

// SetCustomWeights Set user's weights for ReLU layer (make it carefully)
func (relu *ReLULayer) SetCustomWeights(t []*mat.Dense) {
	fmt.Println("There are no weights for ReLU layer")
}

// GetOutputSize Returns output size (dimensions) of ReLU layer
func (relu *ReLULayer) GetOutputSize() *tensor.TDsize {
	return relu.OutputSize
}

// GetActivatedOutput Returns ReLU layer's output
func (relu *ReLULayer) GetActivatedOutput() *mat.Dense {
	return relu.Ok
}

// GetWeights Returns ReLU layer's weights
func (relu *ReLULayer) GetWeights() []*mat.Dense {
	fmt.Println("There are no weights for ReLU layer")
	return nil
}

// GetGradients Returns ReLU layer's gradients
func (relu *ReLULayer) GetGradients() *mat.Dense {
	return relu.LocalDelta
}

// FeedForward - Feed data to ReLU layer
func (relu *ReLULayer) FeedForward(t *mat.Dense) error {
	relu.Oj = t
	relu.doActivation()
	return nil
}

// doActivation ReLU layer's output activation
func (relu *ReLULayer) doActivation() {
	rawOj := relu.Oj.RawMatrix().Data
	rawOk := relu.Ok.RawMatrix().Data
	for j := range rawOj {
		if rawOj[j] < 0 {
			rawOk[j] = 0
		} else {
			rawOk[j] = rawOj[j]
		}
	}
}

// CalculateGradients Evaluate ReLU layer's gradients
func (relu *ReLULayer) CalculateGradients(errorsDense *mat.Dense) error {
	raw := relu.Oj.RawMatrix().Data
	rawDelta := relu.LocalDelta.RawMatrix().Data
	rawErrors := errorsDense.RawMatrix().Data
	for i := range raw {
		if raw[i] < 0 {
			rawDelta[i] = 0
		} else {
			rawDelta[i] = rawErrors[i]
		}
	}
	return nil
}

// UpdateWeights Just to point, that ReLU layer does NOT updating weights
func (relu *ReLULayer) UpdateWeights() {
	// There are no weights to update for ReLU layer
}

// PrintOutput Pretty print ReLU layer's output
func (relu *ReLULayer) PrintOutput() {
	fmt.Println("Printing ReLU Layer output...")
}

// PrintWeights Just to point, that ReLU layer has not weights
func (relu *ReLULayer) PrintWeights() {
	fmt.Println("There are no weights for ReLU layer")
}

// SetActivationFunc Set activation function for layer
func (relu *ReLULayer) SetActivationFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set activation function for ReLU layer")
}

// SetActivationDerivativeFunc Set derivative of activation function
func (relu *ReLULayer) SetActivationDerivativeFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set derivative of activation function for ReLU layer")
}

// GetStride Returns stride of layer
func (relu *ReLULayer) GetStride() int {
	return 0
}

// GetType Returns "relu" as layer's type
func (relu *ReLULayer) GetType() string {
	return "relu"
}
