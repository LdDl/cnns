package nns

import (
	"cnns_vika/utils/u"
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
	In                    *Tensor
	Out                   *Tensor
	Kernels               []Tensor
	KernelWidth           int
	KernelHeight          int
	StrideWidth           int
	StrideHeight          int
	PaddingWidth          int
	PaddingHeight         int
	InputGradients        *Tensor
	DeltaWeights          []*Tensor
	PreviuousDeltaWeights []*Tensor
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
			kernelsNumber),
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

		newLayer.DeltaWeights = append(newLayer.DeltaWeights, NewTensorEmpty(kernelWidth, kernelHeight, depth))
		newLayer.PreviuousDeltaWeights = append(newLayer.PreviuousDeltaWeights, NewTensorEmpty(kernelWidth, kernelHeight, depth))

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
	for k := 0; k < len((*con).DeltaWeights); k++ {
		for i := 0; i < (*con).KernelWidth; i++ {
			for j := 0; j < (*con).KernelHeight; j++ {
				for z := 0; z < (*con).In.Z; z++ {
					(*con).DeltaWeights[k].SetValue(i, j, z, 0)
				}
			}
		}
	}
	for x := 0; x < (*con).In.X; x++ {
		for y := 0; y < (*con).In.Y; y++ {
			rn := (*con).SameAsOuput(x, y)
			for z := 0; z < (*con).In.Z; z++ {
				sumError := 0.0
				for i := rn.MinX; i <= rn.MaxX; i++ {
					minx := i * (*con).StrideWidth
					for j := rn.MinY; j <= rn.MaxY; j++ {
						miny := j * (*con).StrideHeight
						for k := rn.MinZ; k <= rn.MaxZ; k++ {
							weightApplied := (*con).Kernels[k].GetValue(x-minx, y-miny, z)
							sumError += weightApplied * nextLayerGrad.GetValue(i, j, k)
							(*con).DeltaWeights[k].AddValue(x-minx, y-miny, z, (*con).In.GetValue(x, y, z)*nextLayerGrad.GetValue(i, j, k))
							// fmt.Printf("delta weights index (i,j): (%v, %v)\n\t", x-minx, y-miny)
							// fmt.Printf("In val * Grad val: %v * %v = %v\n", (*con).In.GetValue(x, y, z), nextLayerGrad.GetValue(i, j, k), (*con).In.GetValue(x, y, z)*nextLayerGrad.GetValue(i, j, k))
							// fmt.Printf("\tCurrent summation: %v\n", (*con).DeltaWeights[k].GetValue(x-minx, y-miny, z))
						}
					}
				}
				(*con).InputGradients.SetValue(x, y, z, sumError)
				// fmt.Printf("conv grad: %v , corresponding value: %v \n", sumError, (*con).In.GetValue(x, y, z))
			}
		}
	}
	// fmt.Println("New deltaWeights")
	// (*con).DeltaWeights[0].Print()
}

// SameAsOuput - reshape convolutional layer's output
func (con *ConvLayer) SameAsOuput(x, y int) RangeP {
	a := float64(x)
	b := float64(y)
	return RangeP{
		MinX: u.NormalizeRange((a-float64((*con).KernelWidth)+1.0)/float64((*con).StrideWidth), (*con).Out.X, true),
		MinY: u.NormalizeRange((b-float64((*con).KernelHeight)+1.0)/float64((*con).StrideHeight), (*con).Out.Y, true),
		MinZ: 0,
		MaxX: u.NormalizeRange(a/float64((*con).StrideWidth), (*con).Out.X, false),
		MaxY: u.NormalizeRange(b/float64((*con).StrideHeight), (*con).Out.Y, false),
		MaxZ: len((*con).Kernels) - 1,
	}
}

// RangeP is struct for reshaping convolution arrays
type RangeP struct {
	MinX, MinY, MinZ int
	MaxX, MaxY, MaxZ int
}

// UpdateWeights - update convolutional layer's weights
func (con *ConvLayer) UpdateWeights() {
	for f := 0; f < len((*con).Kernels); f++ {
		for i := 0; i < (*con).KernelWidth; i++ {
			for j := 0; j < (*con).KernelHeight; j++ {
				for z := 0; z < (*con).In.Z; z++ {
					// weightFilter := (*con).Kernels[f].GetValue(i, j, z)
					previousDelta := (*con).PreviuousDeltaWeights[f].GetValue(i, j, z)
					deltaWeight := (*con).DeltaWeights[f].GetValue(i, j, z)
					// oldGrad := (*con).OldFilterGrads[f].GetValue(i, j, z)

					_ = previousDelta
					deltaWeight = LearningRate * deltaWeight // + Momentum*previousDelta
					(*con).Kernels[f].AddValue(i, j, z, deltaWeight)
					// (*con).Filters[f].SetSingle(i, j, z, UpdateWeight(weightFilter, &oldGrad, &newGrad, 1.0))
					// (*con).OldFilterGrads[f].SetSingle(i, j, z, UpdateGradient(&oldGrad, &newGrad))
				}
			}
		}
	}
}

// DoActivation - convolutional layer's output activation
func (con *ConvLayer) DoActivation() {
	(*con).Out = Convolve2D((*con).In, &(*con).Kernels[0], (*con).StrideWidth, (*con).StrideHeight, (*con).PaddingWidth, (*con).PaddingHeight)
}
