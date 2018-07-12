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
// Weights - array of neurons
// NewGradients - new values of gradients
// Gradients - old values of gradients
// GradientsWeights - gradients
// ActivationFunc       - activation function. You can set custom func(v float64) float64, see SetActivationFunc
// ActivationDerivative - derivative of activation function. You can set custom func(v float64) float64, see SetActivationDerivativeFunc
// IsLastLayer - identify if layers is last (this affects deltas' calculating)
type FullConnectedLayer struct {
	In                   *Tensor
	Out                  *Tensor
	OutActivated         *Tensor
	Weights              *Tensor
	NewGradients         *Tensor
	Gradients            *Tensor
	GradientsWeights     *Tensor
	ActivationFunc       func(v float64) float64
	ActivationDerivative func(v float64) float64
	IsLastLayer          bool
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
func NewFullConnectedLayer(width, height, depth int, outputSize int, isLast bool) *FullConnectedLayer {
	newLayer := &FullConnectedLayer{
		In:                   NewTensorEmpty(width, height, depth),
		Out:                  NewTensorEmpty(outputSize, 1, 1),
		OutActivated:         NewTensorEmpty(outputSize, 1, 1),
		Weights:              NewTensorEmpty(width*height*depth, outputSize, 1),
		Gradients:            NewTensorEmpty(outputSize, 1, 1),
		NewGradients:         NewTensorEmpty(outputSize, 1, 1),
		GradientsWeights:     NewTensorEmpty(width, height, depth),
		ActivationFunc:       ActivationSygmoid,           // Default Activation function is Sygmoid
		ActivationDerivative: ActivationSygmoidDerivative, // Default derivative of activation function is Sygmoid*(1-Sygmoid)
		IsLastLayer:          isLast,
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
	fmt.Println("Printing Fully Connected Layer weights...")
	(*fc).Weights.Print()
}

// PrintOutput - print fully connected layer's output
func (fc *FullConnectedLayer) PrintOutput() {
	fmt.Println("Printing Fully Connected Layer output...")
	(*fc).Out.Print()
}

// PrintGradients - print fully connected layer's gradients
func (fc *FullConnectedLayer) PrintGradients() {
	fmt.Println("Printing Fully Connected Layer gradients-weights...")
	(*fc).GradientsWeights.Print()
}

// GetOutput - get fully connected layer's output
func (fc *FullConnectedLayer) GetOutput() *Tensor {
	return (*fc).OutActivated // Here we outputing ACTIVATED values
}

// FeedForward - feed data to fully connected layer
func (fc *FullConnectedLayer) FeedForward(t *Tensor) {
	(*fc).In = t
	(*fc).DoActivation()
}

// GetGradients - get fully connected layer's gradients
func (fc *FullConnectedLayer) GetGradients() *Tensor {
	return (*fc).GradientsWeights
}

// CalculateGradients - calculate fully connected layer's gradients
func (fc *FullConnectedLayer) CalculateGradients(difference *Tensor) {
	if (*fc).IsLastLayer {
		/*
			δ{k} = O{k}*(1-O{k})*(O{k}-t{k}),
				where
					O{k}*(1-O{k}) - derevative of sygmoid activation function
					(O{k}-t{k}) - difference between Output and
		*/
		for out := 0; out < (*fc).Out.X; out++ {
			(*fc).Gradients.SetValue(out, 0, 0, (*fc).ActivationDerivative((*fc).Out.GetValue(out, 0, 0))*difference.GetValue(out, 0, 0))
			fmt.Printf("Output error: %v\n", (*fc).Gradients.GetValue(out, 0, 0))
		}
		return
	}

	/*
		δ{j-1} = O{j-1}*(1-O{j-1}) * SUM[δ{j}*w{j-1,j}],
				where
					O{j-1}*(1-O{j-1}) - derevative of sygmoid activation function
					SUM[δ{j}*w{j-1,j}] - product of next layer local gradients and connected weights
	*/

	// for k := 0; k < (*fc).GradientsWeights.Z; k++ {
	// 	for j := 0; j < (*fc).GradientsWeights.Y; j++ {
	// 		for i := 0; i < (*fc).GradientsWeights.X; i++ {
	// 			(*fc).GradientsWeights.SetValue(i, j, k, 0)
	// 		}
	// 	}
	// }

	// for out := 0; out < (*fc).Out.X; out++ {
	// 	// (*fc).NewGradients.SetValue(out, 0, 0, (*fc).ActivationDerivative((*fc).Out.GetValue(out, 0, 0))*difference.GetValue(out, 0, 0)) // δ
	// 	// deltaPrevious := (*fc).NewGradients.GetValue(out, 0, 0)
	// 	// fmt.Printf("neuron #%v %v\n", out, deltaPrevious)
	// 	// fmt.Printf("%v * %v\n", (*fc).ActivationDerivative((*fc).Out.GetValue(out, 0, 0)), nextLayerGrad.GetValue(out, 0, 0))
	// 	for k := 0; k < (*fc).In.Z; k++ {
	// 		for j := 0; j < (*fc).In.Y; j++ {
	// 			for i := 0; i < (*fc).In.X; i++ {
	// 				// mappedIndex := (*fc).In.GetIndex(i, j, k)
	// 				// weightVal := (*fc).Weights.GetValue(mappedIndex, out, 0)
	// 				// // fmt.Printf("%v * %v\n", deltaPrevious, weightVal)
	// 				// (*fc).GradientsWeights.AddValue(i, j, k, deltaPrevious*weightVal)
	// 			}
	// 		}
	// 	}
	// }
}

const (
	// LearningRate ...
	LearningRate = 0.3
	// Momentum ...
	Momentum = 0.6
)

// UpdateWeights - update fully connected layer's weights
func (fc *FullConnectedLayer) UpdateWeights() {
	for out := 0; out < (*fc).Out.X; out++ {
		// newGrad := (*fc).NewGradients.GetValue(out, 0, 0)
		// grad := (*fc).Gradients.GetValue(out, 0, 0)
		// m := (newGrad + grad*Momentum)
		for k := 0; k < (*fc).In.Z; k++ {
			for j := 0; j < (*fc).In.Y; j++ {
				for i := 0; i < (*fc).In.X; i++ {
					// mappedIndex := (*fc).In.GetIndex(i, j, k)
					// layerVal := (*fc).In.GetValue(i, j, k)
					// weightVal := (*fc).Weights.GetValue(mappedIndex, out, 0)
					// // fmt.Printf("%v * %v * %v = %v\n", LearningRate, newGrad, layerVal, LearningRate*newGrad*layerVal)
					// deltaWeight := weightVal - LearningRate*newGrad*layerVal
					// (*fc).Weights.SetValue(mappedIndex, out, 0, deltaWeight)
				}
			}
		}
		// (*fc).Gradients.SetValue(out, 0, 0, m)
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
					//fmt.Printf("%v * %v\n", inputVal, weightVal)
					sum += inputVal * weightVal
				}
			}
		}
		(*fc).Out.SetValue(out, 0, 0, sum)
		(*fc).OutActivated.SetValue(out, 0, 0, (*fc).ActivationFunc(sum))
	}
}
