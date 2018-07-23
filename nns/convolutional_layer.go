package nns

import (
	"fmt"
	"math/rand"
)

// ConvLayer is convolutional layer structure, where
// In - input data
// Out - output data
// Kernels - array of neurons
// Stride - striding step
// Kernels - size of neuron. 3x3, 4x2, 5x9,... and etc.
// InputGradients - gradients
type ConvLayer struct {
	In             *Tensor
	Out            *Tensor
	Kernels        []Tensor
	KernelWidth    int
	KernelHeight   int
	StrideWidth    int
	StrideHeight   int
	PaddingWidth   int
	PaddingHeight  int
	InputGradients *Tensor
}

// NewConvLayer - constructor for new convolutional layer. You need to specify striding step, size (square) of kernel, amount of kernels, input size.
func NewConvLayer(
	kernelWidth, kernelHeight int,
	strideWidth, strideHeight int,
	paddingWidth, paddingHeight int,
	kernelsNumber int,
	width, height, depth int,
) *ConvLayer {

	newLayer := &ConvLayer{
		In: NewTensorEmpty(width, height, depth),
		Out: NewTensorEmpty(
			(width-kernelWidth+2*paddingWidth)/strideWidth+1,
			(height-kernelHeight+2*paddingHeight)/strideHeight+1,
			depth),
		Kernels:        make([]Tensor, kernelsNumber),
		KernelWidth:    kernelWidth,
		KernelHeight:   kernelHeight,
		StrideWidth:    strideWidth,
		StrideHeight:   strideHeight,
		PaddingWidth:   paddingWidth,
		PaddingHeight:  paddingHeight,
		InputGradients: NewTensorEmpty(width, height, depth),
	}

	for f := range (*newLayer).Kernels {
		kernelTensor := NewTensorEmpty(kernelWidth, kernelHeight, depth)
		for k := 0; k < depth; k++ {
			for j := 0; j < kernelHeight; j++ {
				for i := 0; i < kernelWidth; i++ {
					kernelTensor.SetValue(i, j, k, rand.Float64()-0.5)
				}
			}
		}
		(*newLayer).Kernels[f] = *kernelTensor
	}
	return newLayer
}

// PrintWeights - print convolutional layer's weights
func (con *ConvLayer) PrintWeights() {
	fmt.Println("Printing Convolutional Layer kernels...")
	for i := range (*con).Kernels {
		fmt.Printf("Kernel #%v\n", i)
		(*con).Kernels[i].Print()
	}
}

// PrintOutput - print convolutional layer's output
func (con *ConvLayer) PrintOutput() {
	fmt.Println("Printing Convolutional Layer output...")
	(*con).Out.Print()
}

// GetOutput - get convolutional layer's output
func (con *ConvLayer) GetOutput() *Tensor {
	return (*con).Out
}

// FeedForward - feed data to convolutional layer
func (con *ConvLayer) FeedForward(t *Tensor) {
	(*con).In = t
	(*con).DoActivation()
}

// PrintGradients - print convolutional layer's gradients
func (con *ConvLayer) PrintGradients() {
	fmt.Println("Printing Convolutional Layer gradients-weights...")
	(*con).InputGradients.Print()
}

// PrintSumGradWeights - print convolutional layer's summ of grad*weight
func (con *ConvLayer) PrintSumGradWeights() {
}

// GetGradients - get convolutional layer's gradients
func (con *ConvLayer) GetGradients() *Tensor {
	return (*con).InputGradients
}

// CalculateGradients - calculate convolutional layer's gradients
func (con *ConvLayer) CalculateGradients(nextLayerGrad *Tensor) {
}

// UpdateWeights - update convolutional layer's weights
func (con *ConvLayer) UpdateWeights() {
}

// DoActivation - convolutional layer's output activation
func (con *ConvLayer) DoActivation() {
	(*con).Out = Convolve2D((*con).In, &(*con).Kernels[0], (*con).StrideWidth, (*con).StrideHeight, (*con).PaddingWidth, (*con).PaddingHeight)
}
