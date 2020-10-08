package cnns

import (
	"fmt"
	"math/rand"

	"github.com/LdDl/cnns/tensor"
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// ConvLayer Convolutional layer structure
// Oj - O{j}, activated output from previous layer for j-th neuron (in other words: previous summation input)
// Ok - O{k}, activated output from current layer for k-th node (in other words: activated summation input)
// SumInput - non-activated output for current layer for k-th node (in other words: summation input)
type ConvLayer struct {
	Oj                        *mat.Dense
	Ok                        *mat.Dense
	Kernels                   []*mat.Dense
	PreviousDeltaKernelsState []*mat.Dense

	LocalDeltas        []*mat.Dense
	NextDeltaWeightSum *mat.Dense

	Stride     int
	KernelSize int

	OutputSize *tensor.TDsize
	inputSize  *tensor.TDsize

	inChannels int
	trainMode  bool
}

// NewConvLayer Constructor for convolutional layer. You need to specify striding step, size (square) of kernel, amount of kernels, input size.
/*
	inSize - size of input (width/height and number of channels)
	stride - step on convolve operation
	kernelSize - width==height of kernel
	numberFilters - number of kernels
*/
func NewConvLayer(inSize *tensor.TDsize, stride, kernelSize, numberFilters int) Layer {
	newLayer := &ConvLayer{
		inputSize:                 inSize,
		Stride:                    stride,
		KernelSize:                kernelSize,
		Ok:                        &mat.Dense{},
		Kernels:                   make([]*mat.Dense, numberFilters),
		PreviousDeltaKernelsState: make([]*mat.Dense, numberFilters),
		LocalDeltas:               make([]*mat.Dense, numberFilters),
		NextDeltaWeightSum:        &mat.Dense{},
		OutputSize:                &tensor.TDsize{X: (inSize.X-kernelSize)/stride + 1, Y: (inSize.Y-kernelSize)/stride + 1, Z: numberFilters},
		inChannels:                inSize.Z,
		trainMode:                 false,
	}
	for f := 0; f < numberFilters; f++ {
		if inSize.Z == 1 {
			newLayer.Kernels[f] = mat.NewDense(kernelSize, kernelSize, nil)
			for i := 0; i < kernelSize; i++ {
				for h := 0; h < kernelSize; h++ {
					newLayer.Kernels[f].Set(i, h, rand.Float64()-0.5)
				}
			}
			newLayer.PreviousDeltaKernelsState[f] = mat.NewDense(kernelSize, kernelSize, nil)
			newLayer.PreviousDeltaKernelsState[f].Zero()
			continue
		}
		newLayer.Kernels[f] = mat.NewDense(kernelSize*kernelSize, inSize.Z, nil)
		for i := 0; i < kernelSize*kernelSize; i++ {
			for h := 0; h < inSize.Z; h++ {
				newLayer.Kernels[f].Set(i, h, rand.Float64()-0.5)
			}
		}
		newLayer.PreviousDeltaKernelsState[f] = mat.NewDense(kernelSize*kernelSize, inSize.Z, nil)
		newLayer.PreviousDeltaKernelsState[f].Zero()
	}
	return newLayer
}

// SetCustomWeights Set user's weights for convolutional layer (make it carefully)
/*
	kernels - slice of kernels
*/
func (conv *ConvLayer) SetCustomWeights(kernels []*mat.Dense) {
	if len(conv.Kernels) != len(kernels) {
		fmt.Println("Amount of custom filters has to be equal to layer's amount of filters. Skipping...")
		return
	}
	for i := range kernels {
		conv.Kernels[i].CloneFrom(kernels[i])
		tr, tc := kernels[i].Dims()
		conv.PreviousDeltaKernelsState[i] = mat.NewDense(tr, tc, nil)
		conv.PreviousDeltaKernelsState[i].Zero()
	}
}

// GetInputSize Returns dimensions of incoming data for convolutional layer
func (conv *ConvLayer) GetInputSize() *tensor.TDsize {
	return conv.inputSize
}

// GetOutputSize Returns output size (dimensions) of convolutional layer
func (conv *ConvLayer) GetOutputSize() *tensor.TDsize {
	return conv.OutputSize
}

// GetActivatedOutput Returns convolutional layer's output
func (conv *ConvLayer) GetActivatedOutput() *mat.Dense {
	return conv.Ok
}

// GetWeights Returns convolutional layer's weights.
func (conv *ConvLayer) GetWeights() []*mat.Dense {
	return conv.Kernels
}

// GetGradients Returns convolutional layer's gradients dense
func (conv *ConvLayer) GetGradients() *mat.Dense {
	return conv.NextDeltaWeightSum
}

// FeedForward Feed data to convolutional layer
func (conv *ConvLayer) FeedForward(input *mat.Dense) error {
	conv.Oj = input
	err := conv.doActivation()
	if err != nil {
		return errors.Wrap(err, "Can't call FeedForward() on convolutional layer")
	}
	return nil
}

// doActivation Convolutional layer's output activation
func (conv *ConvLayer) doActivation() error {
	resultMatrix := &mat.Dense{}
	for i := range conv.Kernels {
		feature, err := Convolve2D(conv.Oj, conv.Kernels[i], conv.inChannels, conv.Stride)
		if err != nil {
			return errors.Wrap(err, "Can't call doActivation() on Convolutional Layer")
		}
		if resultMatrix.IsEmpty() {
			resultMatrix = feature
		} else {
			t := &mat.Dense{}
			t.Stack(resultMatrix, feature)
			resultMatrix = t
		}
	}
	conv.Ok = resultMatrix
	return nil
}

// CalculateGradients Evaluate convolutional layer's gradients
func (conv *ConvLayer) CalculateGradients(lossGradients *mat.Dense) error {

	channels := conv.inChannels
	features := conv.OutputSize.Z
	errRows, errCols := lossGradients.Dims()
	inputRows, inputCols := conv.Oj.Dims()

	for f := 0; f < features; f++ {
		partialErrors := ExtractChannel(lossGradients, errRows, errCols, features, f)
		channelsStack := &mat.Dense{}
		for c := 0; c < channels; c++ {
			partialMatrix := ExtractChannel(conv.Oj, inputRows, inputCols, channels, c)
			// dL/dF = Convolution(Input, LossGradient dL/dO)
			partialLocalDeltas, err := Convolve2D(partialMatrix, partialErrors, 1, conv.Stride)
			if err != nil {
				return errors.Wrap(err, "Can't call CalculateGradients() while calculate Convolution(Input, LossGradient dL/dO)")
			}
			if channelsStack.IsEmpty() {
				channelsStack = partialLocalDeltas
			} else {
				t := &mat.Dense{}
				t.Stack(channelsStack, partialLocalDeltas)
				channelsStack = t
			}
		}
		conv.LocalDeltas[f] = channelsStack
	}

	conv.NextDeltaWeightSum = &mat.Dense{}
	for f := 0; f < features; f++ {

		// Add padding for each incoming loss gradient
		partialErrors := ExtractChannel(lossGradients, errRows, errCols, features, f)

		padded := ZeroPadding(partialErrors, conv.KernelSize-1)

		// Rotate each kernel by 180 degrees and do full convolution
		kernelR, kernelC := conv.Kernels[f].Dims()
		channelStacked := &mat.Dense{}
		for c := 0; c < channels; c++ {
			partialKernel := ExtractChannel(conv.Kernels[f], kernelR, kernelC, channels, c)
			partialRotatedKernel := Rot2D180(partialKernel)

			// error = dL/dX = FullConvolution(LossGradient dL/dO, rot180(kernel))
			dLdX, err := Convolve2D(padded, partialRotatedKernel, 1, conv.Stride)
			if err != nil {
				return errors.Wrap(err, "Can't call CalculateGradients() while calculate FullConvolution(LossGradient dL/dO, rot180(kernel))")
			}

			// Stack channels for each feature
			if channelStacked.IsEmpty() {
				channelStacked = dLdX
			} else {
				t := &mat.Dense{}
				t.Stack(channelStacked, dLdX)
				channelStacked = t
			}
		}

		// Sum channels for each feature
		if conv.NextDeltaWeightSum.IsEmpty() {
			conv.NextDeltaWeightSum = channelStacked
		} else {
			t := &mat.Dense{}
			t.Add(conv.NextDeltaWeightSum, channelStacked)
			conv.NextDeltaWeightSum = t
		}
	}

	return nil
}

// UpdateWeights Update convolutional layer's weights
func (conv *ConvLayer) UpdateWeights(lp *LearningParams) {
	features := len(conv.Kernels)

	for f := 0; f < features; f++ {
		kernel := conv.Kernels[f]
		previousDeltaWeights := conv.PreviousDeltaKernelsState[f]
		partialErrors := conv.LocalDeltas[f]

		// Evaluate ΔΣ(k)/Δw{j}{k}
		// In FC layer we do: Δw.Mul(fc.LocalDelta, fc.Oj.T()), but fc.Oj.T() = 1.0 in case of convolutional layer. So we can skip this step
		Δw := &mat.Dense{}
		Δw.Scale(-1.0*lp.LearningRate, partialErrors)

		// Inertia (as separated Scale() call)
		// @todo - this should be optional
		Δw.Scale(1.0-lp.Momentum, Δw)

		previousDeltaWeights.Scale(lp.Momentum, previousDeltaWeights)
		Δw.Add(Δw, previousDeltaWeights)
		conv.PreviousDeltaKernelsState[f].CloneFrom(Δw)

		// Update weights: w = w + Δw
		kernel.Add(kernel, Δw)
	}
}

// PrintOutput Pretty print convolutional layer's output
func (conv *ConvLayer) PrintOutput() {
	fmt.Println("Printing Convolutional Layer output...")
}

// PrintWeights Pretty print convolutional layer's weights
func (conv *ConvLayer) PrintWeights() {
	fmt.Println("Printing Convolutional Layer kernels...")
	features := len(conv.Kernels)
	for f := 0; f < features; f++ {
		kernelRows, kernelCols := conv.Kernels[f].Dims()
		fmt.Printf("\tKernel #%d:\n", f)
		for c := 0; c < conv.inChannels; c++ {
			partialKernel := ExtractChannel(conv.Kernels[f], kernelRows, kernelCols, conv.inChannels, c)
			fmt.Printf("\tChannel #%d:\n", c)
			rows, _ := partialKernel.Dims()
			for r := 0; r < rows; r++ {
				fmt.Printf("\t\t%v\n", partialKernel.RawRowView(r))
			}
		}
	}
}

// SetActivationFunc Set activation function for layer
func (conv *ConvLayer) SetActivationFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set activation function for convolutional layer")
}

// SetActivationDerivativeFunc Set derivative of activation function
func (conv *ConvLayer) SetActivationDerivativeFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set derivative of activation function for convolutional layer")
}

// GetType Returns "conv" as layer's type
func (conv *ConvLayer) GetType() string {
	return "conv"
}

// GetStride Returns stride of layer
func (conv *ConvLayer) GetStride() int {
	return conv.Stride
}
