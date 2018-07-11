package nns

import (
	"fmt"
	"log"
	"math"
)

// MaxPoolingLayer is Max Pooling layer structure
// In - input data
// Out - output data
// Stride - striding step
// FilterSize - size of neuron (kernel). 3x3, 4x4, 5x5,... and etc.
// GradsIn - gradients
type MaxPoolingLayer struct {
	In             *Tensor
	Out            *Tensor
	InputGradients *Tensor
	StrideWidth    int
	StrideHeight   int
}

// NewMaxPoolingLayer - constructor for new MaxPooling layer.
func NewMaxPoolingLayer(
	inputWidth, inputHeight, inputDepth int,
	poolWidth, poolHeight int,
) *MaxPoolingLayer {

	if math.Mod(float64(inputWidth), float64(poolWidth)) != 0 {
		inputWidth++
	}
	if math.Mod(float64(inputHeight), float64(poolHeight)) != 0 {
		inputHeight++
	}
	newLayer := &MaxPoolingLayer{
		In:             NewTensorEmpty(inputWidth, inputHeight, inputDepth),
		Out:            NewTensorEmpty(inputWidth/poolWidth, inputHeight/poolHeight, 1),
		InputGradients: NewTensorEmpty(inputWidth, inputHeight, inputDepth),
		StrideWidth:    poolWidth,
		StrideHeight:   poolHeight,
	}
	log.Println(newLayer.In.X, newLayer.In.Y)
	log.Println(newLayer.Out.X, newLayer.Out.Y)
	return newLayer
}

// PrintWeights - just to point, that max pooling layer has not gradients
func (maxpool *MaxPoolingLayer) PrintWeights() {
	fmt.Println("No weights for Max Pool")
}

// PrintOutput - print max pooling layer's output
func (maxpool *MaxPoolingLayer) PrintOutput() {
	fmt.Println("Printing Max Pooling Layer output...")
	(*maxpool).Out.Print()
}

// GetOutput - get max pooling layer's output
func (maxpool *MaxPoolingLayer) GetOutput() *Tensor {
	return (*maxpool).Out
}

// FeedForward - feed data to max pooling layer
func (maxpool *MaxPoolingLayer) FeedForward(t *Tensor) {
	//	Using loop instead of "(*maxpool).In = t", because of mod(inputWidth/poolWidth) can be not equal to zero
	x := (*t).X
	y := (*t).Y
	z := (*t).Z
	for k := 0; k < z; k++ {
		for j := 0; j < y; j++ {
			for i := 0; i < x; i++ {
				(*maxpool).In.SetValue(i, j, k, (*t).GetValue(i, j, k))
			}
		}
	}
	(*maxpool).DoActivation()
}

// PrintGradients - print max pooling layer's gradients
func (maxpool *MaxPoolingLayer) PrintGradients() {
	fmt.Println("Printing Max Pooling Layer gradients-weights...")
	(*maxpool).InputGradients.Print()
}

// GetGradients - get max pooling layer's gradients
func (maxpool *MaxPoolingLayer) GetGradients() *Tensor {
	return (*maxpool).InputGradients
}

// CalculateGradients - calculate max pooling layer's gradients
func (maxpool *MaxPoolingLayer) CalculateGradients(nextLayerGrad *Tensor) {
}

// UpdateWeights - just to point, that max pooling layer does NOT updating weights
func (maxpool *MaxPoolingLayer) UpdateWeights() {
	/*
		Empty
		Need for layer interface.
	*/
}

// DoActivation - max pooling layer's output activation
func (maxpool *MaxPoolingLayer) DoActivation() {
	Pool((*maxpool).In, (*maxpool).Out, (*maxpool).StrideWidth, (*maxpool).StrideHeight, "max")
}
