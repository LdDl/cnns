package nns

import (
	"fmt"
	"math"
	"math/rand"
)

// FullConnectedLayer is simple layer structure (so this layer can be used for simple neural networks like XOR problem), where
// In - input data
// Out - output data (need for derivative)
// Out - output activated data
// Weights - array of weights of neuron
// SumDeltaWeights - array of delta weights of neuron
// LocalGradients - local gradients for neurons of current layer
// ActivationFunc       - activation function. You can set custom func(v float64) float64, see SetActivationFunc
// ActivationDerivative - derivative of activation function. You can set custom func(v float64) float64, see SetActivationDerivativeFunc
// IsLastLayer - identify if layers is last (this affects deltas' calculating)
type FullConnectedLayer struct {
	In                     *Tensor
	Out                    *Tensor
	OutActivated           *Tensor
	Weights                *Tensor
	PreviousDeltaWeights   *Tensor
	SumDeltaWeights        *Tensor
	PreviousLocalGradients *Tensor
	LocalGradients         *Tensor
	GradientsWeights       *Tensor
	ActivationFunc         func(v float64) float64
	ActivationDerivative   func(v float64) float64
	IsLastLayer            bool
	HasBias                bool
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
func NewFullConnectedLayer(width, height, depth int, outputSize int, hasBias bool, isLast bool) *FullConnectedLayer {
	newLayer := &FullConnectedLayer{
		In:                     NewTensorEmpty(width, height, depth),
		Out:                    NewTensorEmpty(outputSize, 1, 1),
		OutActivated:           NewTensorEmpty(outputSize, 1, 1),
		Weights:                NewTensorEmpty(width*height*depth, outputSize, 1),
		PreviousDeltaWeights:   NewTensorEmpty(width, height, depth),
		SumDeltaWeights:        NewTensorEmpty(width, height, depth),
		PreviousLocalGradients: NewTensorEmpty(outputSize, 1, 1),
		LocalGradients:         NewTensorEmpty(outputSize, 1, 1),
		ActivationFunc:         ActivationSygmoid,           // Default Activation function is Sygmoid
		ActivationDerivative:   ActivationSygmoidDerivative, // Default derivative of activation function is Sygmoid*(1-Sygmoid)
		IsLastLayer:            isLast,
		HasBias:                hasBias,
	}
	if isLast == true && hasBias == true {
		// do not add bias anyways
	}
	var addBias = 0
	if hasBias == true {
		addBias = 1
		newLayer.Weights = NewTensorEmpty(width*height*depth+1, outputSize, 1) // bias
		newLayer.In = NewTensorEmpty(width+1, height, depth)                   // bias
		newLayer.PreviousDeltaWeights = NewTensorEmpty(width+1, height, depth) // bias
	}
	for i := 0; i < (width*height*depth)+addBias; i++ { //bias
		for j := 0; j < outputSize; j++ {
			newLayer.Weights.SetValue(i, j, 0, rand.Float64())
			// newLayer.Weights.SetValue(i, j, 0, -1+rand.Float64()*2)
		}
	}
	return newLayer
}

// PrintWeights - print fully connected layer's weights
func (fc *FullConnectedLayer) PrintWeights() {
	fmt.Println("Printing Fully Connected Layer weights...")
	(*fc).Weights.Print()
}

// PrintOutput - print fully connected layer's output
func (fc *FullConnectedLayer) PrintOutput() {
	fmt.Println("Printing Fully Connected Layer output...")
	(*fc).OutActivated.Print()
}

// PrintGradients - print fully connected layer's gradients
func (fc *FullConnectedLayer) PrintGradients() {
	fmt.Println("Printing Fully Connected Layer gradients-weights...")
	(*fc).LocalGradients.Print()
}

// GetOutput - get fully connected layer's output
func (fc *FullConnectedLayer) GetOutput() *Tensor {
	return (*fc).OutActivated // Here we outputing ACTIVATED values
}

// FeedForward - feed data to fully connected layer
func (fc *FullConnectedLayer) FeedForward(t *Tensor) {
	if (*fc).HasBias {
		(*fc).InputWithBiases(t)
	} else {
		(*fc).In = t
	}
	(*fc).DoActivation()
}

// GetGradients - get sum (next layer grad * weights)
func (fc *FullConnectedLayer) GetGradients() *Tensor {
	return (*fc).SumDeltaWeights
}

// CalculateGradients - calculate fully connected layer's gradients
func (fc *FullConnectedLayer) CalculateGradients(nextLayerGradients *Tensor) {
	for k := 0; k < (*fc).SumDeltaWeights.Z; k++ {
		for j := 0; j < (*fc).SumDeltaWeights.Y; j++ {
			for i := 0; i < (*fc).SumDeltaWeights.X; i++ {
				(*fc).SumDeltaWeights.SetValue(i, j, k, 0)
			}
		}
	}
	for out := 0; out < (*fc).Out.X; out++ {
		/*
			δ{k} = O{k}*(1-O{k})*(O{k}-t{k}),
				where
					O{k}*(1-O{k}) - derevative of sygmoid activation function
					(O{k}-t{k}) - difference between Output and
			delta_W = norm * δ{k} * Input
		*/
		/*
			δ{j-1} = O{j-1}*(1-O{j-1}) * SUM[δ{j}*w{j-1,j}],
					where
						O{j-1}*(1-O{j-1}) - derevative of sygmoid activation function
						SUM[δ{j}*w{j-1,j}] - product of next layer local gradients and connected weights (can be obtained from nextLayerGradients)
		*/
		(*fc).LocalGradients.SetValue(out, 0, 0, (*fc).ActivationDerivative((*fc).Out.GetValue(out, 0, 0))*nextLayerGradients.GetValue(out, 0, 0))
		localGradient := (*fc).LocalGradients.GetValue(out, 0, 0)
		// fmt.Printf("Output error: %v\n", (*fc).LocalGradients.GetValue(out, 0, 0))

		// Calculate SUM[δ{j}*w{j-1,j}] for calculating local gradients for [current-1]-th layer
		for k := 0; k < (*fc).SumDeltaWeights.Z; k++ {
			for j := 0; j < (*fc).SumDeltaWeights.Y; j++ {
				for i := 0; i < (*fc).SumDeltaWeights.X; i++ {
					mappedIndex := (*fc).SumDeltaWeights.GetIndex(i, j, k)
					weightVal := (*fc).Weights.GetValue(mappedIndex, out, 0)
					(*fc).SumDeltaWeights.AddValue(i, j, k, localGradient*weightVal)
				}
			}
		}
	}
}

var (
	// LearningRate ...
	LearningRate = -0.1
	// Momentum ...
	Momentum = 0.6
	// alpha
	a = 0.99
	// beta
	b = 1.01
	// gamma
	y = 1.01
)

// UpdateWeights - update fully connected layer's weights
func (fc *FullConnectedLayer) UpdateWeights() {
	for out := 0; out < (*fc).Out.X; out++ {
		localGradient := (*fc).LocalGradients.GetValue(out, 0, 0)
		prevLocalGradient := (*fc).PreviousLocalGradients.GetValue(out, 0, 0)
		for k := 0; k < (*fc).In.Z; k++ {
			for j := 0; j < (*fc).In.Y; j++ {
				for i := 0; i < (*fc).In.X; i++ {
					mappedIndex := (*fc).In.GetIndex(i, j, k)
					layerVal := (*fc).In.GetValue(i, j, k)
					previousDelta := (*fc).PreviousDeltaWeights.GetValue(i, j, k)
					errorVal := localGradient * layerVal
					prevErrorVal := localGradient - prevLocalGradient
					if (errorVal - y*prevErrorVal) > 0 {
						// LearningRate = b * LearningRate
					} else {
						// LearningRate = a * LearningRate
					}
					_ = previousDelta
					// fmt.Printf("update weights: %v * %v *%v = %v\n", LearningRate, localGradient, layerVal, LearningRate*localGradient*layerVal)
					deltaWeight := LearningRate*errorVal + Momentum*previousDelta
					(*fc).PreviousDeltaWeights.SetValue(i, j, k, deltaWeight)
					(*fc).Weights.AddValue(mappedIndex, out, 0, deltaWeight)
				}
			}
		}
		(*fc).PreviousLocalGradients.SetValue(out, 0, 0, localGradient)
	}
}

// InputWithBiases - set up biases
func (fc *FullConnectedLayer) InputWithBiases(t *Tensor) {
	x := (*t).X
	y := (*t).Y
	z := (*t).Z
	for k := 0; k < z; k++ {
		for j := 0; j < y; j++ {
			for i := 0; i < x; i++ {
				(*fc).In.SetValue(i, j, k, (*t).GetValue(i, j, k))
			}
		}
	}
	for k := 0; k < (*fc).In.Z; k++ {
		for j := 0; j < (*fc).In.Y; j++ {
			(*fc).In.SetValue((*fc).In.X-1, j, k, 1)
		}
	}
}

// DoActivation - fully connected layer's output activation
func (fc *FullConnectedLayer) DoActivation() {
	for out := 0; out < (*fc).Out.X; out++ {
		sum := 0.0
		for k := 0; k < (*fc).In.Z; k++ {
			for j := 0; j < (*fc).In.Y; j++ {
				for i := 0; i < (*fc).In.X; i++ {
					inputVal := (*fc).In.GetValue(i, j, k)
					mappedIndex := (*fc).In.GetIndex(i, j, k)
					weightVal := (*fc).Weights.GetValue(mappedIndex, out, 0)
					// fmt.Printf("%v * %v\n", inputVal, weightVal)
					// fmt.Printf("weight index: %v\n", (*fc).Weights.GetIndex(mappedIndex, out, 0))
					sum += inputVal * weightVal
				}
			}
		}
		(*fc).Out.SetValue(out, 0, 0, sum)
		(*fc).OutActivated.SetValue(out, 0, 0, (*fc).ActivationFunc(sum))
	}
}
