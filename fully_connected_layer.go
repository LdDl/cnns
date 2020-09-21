package cnns

import (
	"fmt"
	"math/rand"

	"github.com/LdDl/cnns/tensor"
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// FullyConnectedLayer FC is simple layer structure (so this layer can be used for simple neural networks like XOR problem)
/*
	Oj - O{j}, activated output from previous layer for j-th neuron (in other words: previous summation input)
	Ok - O{k}, activated output from current layer for k-th node (in other words: activated summation input)
	SumInput - non-activated output for current layer for k-th node (in other words: summation input)
	LocalDelta - δ{k}, delta for current layer for k-th neuron
	NextDeltaWeightSum - SUM(δ{k}*w{j,k}), summation component for evaluating δ{j} for previous layer for j-th neuron
	Weights - w{j,k}, weight from j-th node of previous layer to k-th node of current layer
*/
type FullyConnectedLayer struct {
	Oj                   *mat.Dense
	Ok                   *mat.Dense
	NextDeltaWeightSum   *mat.Dense
	Weights              *mat.Dense
	PreviousWeightsState *mat.Dense
	LocalDelta           *mat.Dense
	SumInput             *mat.Dense
	ActivationFunc       func(v float64) float64
	ActivationDerivative func(v float64) float64
	OutputSize           *tensor.TDsize

	trainMode bool
}

// NewFullyConnectedLayer Constructor for fully-connected layer. You need to specify input size and output size
func NewFullyConnectedLayer(inSize *tensor.TDsize, outSize int) Layer {
	newLayer := &FullyConnectedLayer{
		OutputSize:           &tensor.TDsize{X: outSize, Y: 1, Z: 1},
		Ok:                   &mat.Dense{},
		Oj:                   mat.NewDense(outSize, 1, nil),
		SumInput:             mat.NewDense(outSize, 1, nil),
		Weights:              mat.NewDense(outSize, inSize.Total(), nil),
		PreviousWeightsState: mat.NewDense(outSize, inSize.Total(), nil),
		ActivationFunc:       ActivationTanh,           // Default Activation function is TanH
		ActivationDerivative: ActivationTanhDerivative, // Default derivative of activation function is 1 - TanH(x)*TanH(x)
		trainMode:            false,
	}
	newLayer.PreviousWeightsState.Zero()
	for i := 0; i < outSize; i++ {
		for h := 0; h < inSize.Total(); h++ {
			newLayer.Weights.Set(i, h, rand.Float64()-0.5)
		}
	}
	return newLayer
}

// SetCustomWeights Set user's weights for fully-connected layer (make it carefully)
func (fc *FullyConnectedLayer) SetCustomWeights(weights []*mat.Dense) {
	if len(weights) != 1 {
		fmt.Println("You can provide array of length 1 only (for fully-connected layer)")
		return
	}
	r, c := weights[0].Dims()
	fc.Weights = mat.NewDense(r, c, nil)
	fc.Weights.CloneFrom(weights[0])
}

// GetOutputSize Returns output size (dimensions) of fully-connected layer
func (fc *FullyConnectedLayer) GetOutputSize() *tensor.TDsize {
	return fc.OutputSize
}

// GetActivatedOutput Returns fully-connected layer's output
func (fc *FullyConnectedLayer) GetActivatedOutput() *mat.Dense {
	return fc.Ok // ACTIVATED values
}

// GetWeights Returns fully-connected layer's weights.
func (fc *FullyConnectedLayer) GetWeights() []*mat.Dense {
	return []*mat.Dense{fc.Weights}
}

// GetGradients Returns fully-connected layer's gradients dense
func (fc *FullyConnectedLayer) GetGradients() *mat.Dense {
	return fc.NextDeltaWeightSum
}

// FeedForward Feed data to fully-connected layer
func (fc *FullyConnectedLayer) FeedForward(input *mat.Dense) error {
	r, _ := input.Dims()
	_, weightsCC := fc.Weights.Dims()
	if r != weightsCC {
		// Try to reshape input Dense to match matrix multiplication
		temp, err := Reshape(input, weightsCC, 1)
		if err != nil {
			return errors.Wrap(err, "Can't call FeedForward() on fully-connected layer [1]")
		}
		fc.Oj = temp
	} else {
		fc.Oj.CloneFrom(input)
	}
	err := fc.doActivation()
	if err != nil {
		return errors.Wrap(err, "Can't call FeedForward() on fully-connected layer [2]")
	}
	return nil
}

// doActivation fully-connected layer's output activation
func (fc *FullyConnectedLayer) doActivation() error {
	if fc.Oj == nil {
		return fmt.Errorf("Can't call doActivation() on FC layer")
	}
	fc.Ok.Mul(fc.Weights, fc.Oj)
	fc.SumInput.Copy(fc.Ok)
	rawMatrix := fc.Ok.RawMatrix().Data
	for i := range rawMatrix {
		rawMatrix[i] = fc.ActivationFunc(rawMatrix[i])
	}

	return nil
}

// CalculateGradients Evaluate fully-connected layer's gradients
func (fc *FullyConnectedLayer) CalculateGradients(errorsDense *mat.Dense) error {
	// Evaluate ΔO{k}/ΔΣ(k)
	rawMatrix := fc.SumInput.RawMatrix().Data
	for i := range rawMatrix {
		rawMatrix[i] = fc.ActivationDerivative(rawMatrix[i])
	}

	// Evaluate ΔE{k}/ΔO{k} * ΔO{k}/ΔΣ(k)
	fc.LocalDelta = &mat.Dense{}
	fc.LocalDelta.MulElem(errorsDense, fc.SumInput)

	// Evaluate ΔE{k}/ΔO{k} for next layers in backpropagation direction
	fc.NextDeltaWeightSum = &mat.Dense{}
	fc.NextDeltaWeightSum.Mul(fc.Weights.T(), fc.LocalDelta)

	return nil
}

// UpdateWeights Update fully-connected layer's weights
func (fc *FullyConnectedLayer) UpdateWeights() {
	// Evaluate ΔΣ(k)/Δw{j}{k}
	Δw := &mat.Dense{}
	Δw.Mul(fc.LocalDelta, fc.Oj.T())

	Δw.Scale(-1.0*lp.LearningRate, Δw)

	// Inertia (as separated Scale() call)
	// @todo - this should be optional.
	Δw.Scale(1.0-lp.Momentum, Δw)

	fc.PreviousWeightsState.Scale(lp.Momentum, fc.PreviousWeightsState)

	Δw.Add(Δw, fc.PreviousWeightsState)
	fc.PreviousWeightsState.CloneFrom(Δw)

	// Update weights: w = w + Δw
	fc.Weights.Add(fc.Weights, Δw)
}

// PrintOutput Pretty prrint fully-connected layer's output
func (fc *FullyConnectedLayer) PrintOutput() {
	fmt.Println("Printing fully-connected Layer output...")
	rows, _ := fc.Ok.Dims()
	for r := 0; r < rows; r++ {
		fmt.Printf("\t%v\n", fc.Ok.RawRowView(r))
	}
}

// PrintWeights Pretty print fully-connected layer's weights
func (fc *FullyConnectedLayer) PrintWeights() {
	fmt.Println("Printing fully-connected Layer weights...")
	rows, _ := fc.Weights.Dims()
	for r := 0; r < rows; r++ {
		fmt.Printf("\t%v\n", fc.Weights.RawRowView(r))
	}
}

// SetActivationFunc Set activation function for fully-connected layer. You need to specify function: func(v float64) float64
func (fc *FullyConnectedLayer) SetActivationFunc(f func(v float64) float64) {
	fc.ActivationFunc = f
}

// SetActivationDerivativeFunc Set derivative of activation function for fully-connected layer. You need to specify function: func(v float64) float64
func (fc *FullyConnectedLayer) SetActivationDerivativeFunc(f func(v float64) float64) {
	fc.ActivationDerivative = f
}

// GetStride Returns stride of fully-connected layer
func (fc *FullyConnectedLayer) GetStride() int {
	return 0
}

// GetType Returns "fc" as layer's type
func (fc *FullyConnectedLayer) GetType() string {
	return "fc"
}
