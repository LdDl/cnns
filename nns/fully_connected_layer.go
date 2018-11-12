package nns

import (
	"fmt"
	"math/rand"
)

// FullConnectedLayer is simple layer structure (so this layer can be used for simple neural networks like XOR problem)
type FullConnectedLayer struct {
	In                             Tensor
	Out                            Tensor
	DeltaComponentForPreviousLayer Tensor
	Weights                        Tensor
	PreviousIterationWeights       Tensor
	LocalDelta                     []Gradient
	Input                          []float64
	ActivationFunc                 func(v float64) float64
	ActivationDerivative           func(v float64) float64
}

// NewFullConnectedLayer - constructor for new fully connected layer. You need to specify input size and output size
func NewFullConnectedLayer(inSize TDsize, outSize int) *LayerStruct {
	newLayer := &FullConnectedLayer{
		In:  NewTensor(inSize.X, inSize.Y, inSize.Z),
		Out: NewTensor(outSize, 1, 1),
		DeltaComponentForPreviousLayer: NewTensor(inSize.X, inSize.Y, inSize.Z),
		Weights:                  NewTensor(inSize.X*inSize.Y*inSize.Z, outSize, 1),
		PreviousIterationWeights: NewTensor(inSize.X*inSize.Y*inSize.Z, outSize, 1),
		Input:                make([]float64, outSize),
		LocalDelta:           make([]Gradient, outSize),
		ActivationFunc:       ActivationTanh,           // Default Activation function is TanH
		ActivationDerivative: ActivationTanhDerivative, // Default derivative of activation function is 1 - TanH(x)*TanH(x)
	}
	for i := 0; i < outSize; i++ {
		for h := 0; h < inSize.X*inSize.Y*inSize.Z; h++ {
			newLayer.Weights.Set(h, i, 0, rand.Float64()-0.5)
		}
	}
	return &LayerStruct{
		Layer: newLayer,
	}
}

// SetCustomWeights - set user's weights (make it carefully)
func (fc *FullConnectedLayer) SetCustomWeights(t *[]Tensor) {
	if len(*t) != 1 {
		fmt.Println("You can provide array of length 1 only (for fully-connected layer)")
		return
	}
	for i := 0; i < (*fc).Weights.Size.Y; i++ {
		for h := 0; h < (*fc).Weights.Size.X; h++ {
			// fmt.Println((*fc).Weights.Get(h, i, 0))
			(*fc).Weights.Set(h, i, 0, (*t)[0].Get(h, i, 0))
		}
	}
}

// OutSize - returns output size (dimensions)
func (fc *FullConnectedLayer) OutSize() Point {
	return (*fc).Out.Size
}

// GetInputSize - returns input size (dimensions)
func (fc *FullConnectedLayer) GetInputSize() Point {
	return (*fc).In.Size
}

// GetOutput - returns fully connected layer's output
func (fc *FullConnectedLayer) GetOutput() Tensor {
	return (*fc).Out // Here we outputing ACTIVATED values
}

// GetWeights - returns convolutional layer's weights.
func (fc *FullConnectedLayer) GetWeights() []Tensor {
	return []Tensor{(*fc).Weights}
}

// GetGradients - returns SUM(next layer grad * weights) as gradients
func (fc *FullConnectedLayer) GetGradients() Tensor {
	return (*fc).DeltaComponentForPreviousLayer
}

// FeedForward - feed data to fully connected layer
func (fc *FullConnectedLayer) FeedForward(t *Tensor) {
	(*fc).In = (*t)
	(*fc).DoActivation()
}

// DoActivation - fully connected layer's output activation
func (fc *FullConnectedLayer) DoActivation() {
	for n := 0; n < (*fc).Out.Size.X; n++ {
		inputv := 0.0
		for i := 0; i < (*fc).In.Size.X; i++ {
			for j := 0; j < (*fc).In.Size.Y; j++ {
				for z := 0; z < (*fc).In.Size.Z; z++ {
					m := fc.mapToInput(i, j, z)
					inputv += (*fc).In.Get(i, j, z) * (*fc).Weights.Get(m, n, 0)
				}
			}
		}
		(*fc).Input[n] = inputv
		(*fc).Out.Set(n, 0, 0, (*fc).ActivationFunc(inputv))
	}
}

// CalculateGradients - calculate fully connected layer's gradients
/*
	i - current layer
	(i + 1) - next layer (actually previous in term of backpropagation)
	w{i, j} - weight for i-th neuron of current layer from j-th neuron of previous (in term of feedforward) layer.

	k - index of current layer
	k + 1 - index of next layer (in term of feed forward)
	k - 1 - index of previous layer (in term of feed forward)

	i - index of neuron on current layer
	n - index of neuron on other layer (does not matter which one: next or previous, just "other", but connected of course)

	w{n, i} - weight for n-th neuron on layer for i-th neuron on previous (in term of feed forward) layer


	Last layer:
		δ{i, k} = (out{i, k} - target{i, k}) * derivative(input{i, k}) = LocalDelta

	Hidden layer:
		δ{i, k} = derivative(input{i, k-1}) * [sum(δ{n, k+1} * w{n, i}, for n=0 to len(num of neurons on k+1 layer))] =
				= [sum(δ{n, k+1} * w{n, i} * derivative(input{i, k-1}), for n=0 to len(num of neurons on k+1 layer))]
				= [sum(LocalDelta{n} * w{n, i}), for n=0 to len(num of neurons on k+1 layer))]

*/
func (fc *FullConnectedLayer) CalculateGradients(nextLayerGradients *Tensor) {
	for i := 0; i < (*fc).DeltaComponentForPreviousLayer.Size.X*(*fc).DeltaComponentForPreviousLayer.Size.Y*(*fc).DeltaComponentForPreviousLayer.Size.Z; i++ {
		(*fc).DeltaComponentForPreviousLayer.Data[i] = 0.0
	}

	for n := 0; n < (*fc).Out.Size.X; n++ {
		(*fc).LocalDelta[n].Grad = (*nextLayerGradients).Get(n, 0, 0) * (*fc).ActivationDerivative((*fc).Input[n])
		for i := 0; i < (*fc).In.Size.X; i++ {
			for j := 0; j < (*fc).In.Size.Y; j++ {
				for z := 0; z < (*fc).In.Size.Z; z++ {
					m := fc.mapToInput(i, j, z)
					v := (*fc).LocalDelta[n].Grad * (*fc).Weights.Get(m, n, 0)
					// fmt.Printf("D: %v * %v\n", (*fc).LocalDelta[n].Grad, (*fc).Weights.Get(m, n, 0))
					(*fc).DeltaComponentForPreviousLayer.SetAdd(i, j, z, v)
				}
			}
		}
	}

}

// UpdateWeights - update fully connected layer's weights
/*
	Δw{n, i} - change of weight for n-th neuron on layer for i-th neuron on previous (in term of feed forward) layer

	η - learning rate
	input{n} - activated output of previous layer

	Δw{n, i} =  -(η * ΔE/Δw{n, i}) = -(η)*δ{i}*input{n}
*/
func (fc *FullConnectedLayer) UpdateWeights() {
	for n := 0; n < (*fc).Out.Size.X; n++ {
		grad := (*fc).LocalDelta[n]
		// log.Println("G:", grad)
		for i := 0; i < (*fc).In.Size.X; i++ {
			for j := 0; j < (*fc).In.Size.Y; j++ {
				for z := 0; z < (*fc).In.Size.Z; z++ {
					m := fc.mapToInput(i, j, z)
					// fmt.Printf("%v * %v = %v\n", grad.Grad, fc.In.Get(i, j, z), grad.Grad*lp.LearningRate*(*fc).In.Get(i, j, z))

					/*
						Without inertia
					*/
					// dw := -1.0 * (lp.LearningRate * grad.Grad * (*fc).In.Get(i, j, z)) // delta-Weight

					/*
						With inertia (notice, that η has to be < 0 and we are multiplying η by -1.0)
						See reference: https://en.wikipedia.org/wiki/Backpropagation#Inertia
					*/
					dw := (1.0-lp.Momentum)*(-1.0*(lp.LearningRate*grad.Grad*(*fc).In.Get(i, j, z))) +
						lp.Momentum*(*fc).PreviousIterationWeights.Get(m, n, 0)

					(*fc).PreviousIterationWeights.Set(m, n, 0, dw)

					// w{n,i} = w{n,i} + Δw{n, i}
					(*fc).Weights.SetAdd(m, n, 0, dw)
				}
			}
		}
		// UpdateGradient(&fc.LocalDelta[n])
		//(*fc).LocalDelta[n].Update()
	}
}

// PrintOutput - print fully connected layer's output
func (fc *FullConnectedLayer) PrintOutput() {
	fmt.Println("Printing Fully Connected Layer output...")
	(*fc).Out.Print()
}

// PrintWeights - print fully connected layer's weights
func (fc *FullConnectedLayer) PrintWeights() {
	fmt.Println("Printing Fully Connected Layer weights...")
	(*fc).Weights.Print()
}

// PrintGradients - print fully connected layer's gradients
func (fc *FullConnectedLayer) PrintGradients() {
	fmt.Println("Printing Fully Connected Layer gradients-weights...")
	(*fc).DeltaComponentForPreviousLayer.Print()
}

// SetActivationFunc sets activation function for fully connected layer. You need to specify function: func(v float64) float64
func (fc *FullConnectedLayer) SetActivationFunc(f func(v float64) float64) {
	(*fc).ActivationFunc = f
}

// SetActivationDerivativeFunc sets derivative of activation function for fully connected layer. You need to specify function: func(v float64) float64
func (fc *FullConnectedLayer) SetActivationDerivativeFunc(f func(v float64) float64) {
	(*fc).ActivationDerivative = f
}

// GetStride - get stride of layer
func (fc *FullConnectedLayer) GetStride() int {
	return 0
}

// GetKernelSize - return "conv" as layer's type
func (fc *FullConnectedLayer) GetKernelSize() int {
	return 0
}

// GetType - return "fc" as layer's type
func (fc *FullConnectedLayer) GetType() string {
	return "fc"
}

// func (fc *FullConnectedLayer) mapToInput(d Point) int {
// 	return d.Z*(*fc).In.Size.X*(fc).In.Size.Y + d.Y*(*fc).In.Size.X + d.X
// }

func (fc *FullConnectedLayer) mapToInput(i, j, k int) int {
	return k*(*fc).In.Size.X*(fc).In.Size.Y + j*(*fc).In.Size.X + i
}
