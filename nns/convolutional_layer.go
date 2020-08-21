package nns

import (
	"fmt"
	"math/rand"

	t "github.com/LdDl/cnns/nns/tensor"
	"github.com/LdDl/cnns/utils/u"
)

// ConvLayer is convolutional layer structure
type ConvLayer struct {
	DeltaWeightsComponent t.Tensor
	In                    t.Tensor
	Out                   t.Tensor
	Kernels               []t.Tensor
	PreviousKernelsDeltas []t.Tensor
	LocalDeltas           []TensorGradient
	Stride                int
	KernelSize            int
}

// NewConvLayer - constructor for new convolutional layer. You need to specify striding step, size (square) of kernel, amount of kernels, input size.
func NewConvLayer(stride, kernelSize, numberFilters int, inSize t.TDsize) *LayerStruct {
	newLayer := &ConvLayer{
		DeltaWeightsComponent: t.NewTensor(inSize.X, inSize.Y, inSize.Z),
		In:                    t.NewTensor(inSize.X, inSize.Y, inSize.Z),
		Out:                   t.NewTensor((inSize.X-kernelSize)/stride+1, (inSize.Y-kernelSize)/stride+1, numberFilters),
		Stride:                stride,
		KernelSize:            kernelSize,
	}
	for a := 0; a < numberFilters; a++ {
		tmp := t.NewTensor(kernelSize, kernelSize, inSize.Z)
		for i := 0; i < kernelSize; i++ {
			for j := 0; j < kernelSize; j++ {
				for z := 0; z < inSize.Z; z++ {
					tmp.Set(i, j, z, rand.Float64()-0.5)
				}
			}
		}
		newLayer.Kernels = append(newLayer.Kernels, tmp)

		tt := t.NewTensor(kernelSize, kernelSize, inSize.Z)
		for i := 0; i < kernelSize; i++ {
			for j := 0; j < kernelSize; j++ {
				for z := 0; z < inSize.Z; z++ {
					tt.Set(i, j, z, 0)
				}
			}
		}
		newLayer.PreviousKernelsDeltas = append(newLayer.PreviousKernelsDeltas, tt)

		for i := 0; i < numberFilters; i++ {
			t := NewTensorGradient(kernelSize, kernelSize, inSize.Z)
			newLayer.LocalDeltas = append(newLayer.LocalDeltas, t)
		}
	}
	return &LayerStruct{
		Layer: newLayer,
	}
}

// SetCustomWeights - set user's weights (make it carefully)
func (con *ConvLayer) SetCustomWeights(t *[]t.Tensor) {
	if len((*con).Kernels) != len(*t) {
		fmt.Println("Amount of custom filters has to be equal to layer's amount of filters. Skipping...")
		return
	}
	for i := range (*con).Kernels {
		(*con).Kernels[i] = (*t)[i]
	}
}

// OutSize - returns output size (dimensions)
func (con *ConvLayer) OutSize() t.TDsize {
	return (*con).Out.Size
}

// GetInputSize - returns input size (dimensions)
func (con *ConvLayer) GetInputSize() t.TDsize {
	return (*con).In.Size
}

// GetOutput - returns convolutional layer's output
func (con *ConvLayer) GetOutput() t.Tensor {
	return (*con).Out
}

// GetWeights - returns convolutional layer's weights
func (con *ConvLayer) GetWeights() []t.Tensor {
	return (*con).Kernels
}

// GetGradients - returns convolutional layer's gradients
func (con *ConvLayer) GetGradients() t.Tensor {
	return (*con).DeltaWeightsComponent
}

// FeedForward - feed data to convolutional layer
func (con *ConvLayer) FeedForward(t *t.Tensor) {
	(*con).In = (*t)
	(*con).DoActivation()
}

// DoActivation - convolutional layer's output activation
func (con *ConvLayer) DoActivation() {
	for filter := 0; filter < len((*con).Kernels); filter++ {
		filterData := (*con).Kernels[filter]
		// filterData = filterData.Rot2D180()
		// (*con).Out = (*con).In.Conv2D(filterData, [2]int{1, 1}, [2]int{0, 0})
		for x := 0; x < (*con).Out.Size.X; x++ {
			for y := 0; y < (*con).Out.Size.Y; y++ {
				// mappedX, mappedY, _ := con.mapToInput(x, y, 0)
				mappedX, mappedY := x*(*con).Stride, y*(*con).Stride
				sum := 0.0
				for i := 0; i < (*con).KernelSize; i++ {
					for j := 0; j < (*con).KernelSize; j++ {
						for z := 0; z < (*con).In.Size.Z; z++ {
							f := filterData.Get(i, j, z)
							v := (*con).In.Get(mappedX+i, mappedY+j, z)
							sum += f * v
						}
					}
				}
				(*con).Out.Set(x, y, filter, sum)
			}
		}
	}
}

// CalculateGradients - calculate convolutional layer's gradients
func (con *ConvLayer) CalculateGradients(nextLayerGrad *t.Tensor) {
	for k := 0; k < len((*con).LocalDeltas); k++ {
		for i := 0; i < (*con).KernelSize; i++ {
			for j := 0; j < (*con).KernelSize; j++ {
				for z := 0; z < (*con).In.Size.Z; z++ {
					(*con).LocalDeltas[k].SetGrad(i, j, z, 0.0)
				}
			}
		}
	}
	for x := 0; x < (*con).In.Size.X; x++ {
		for y := 0; y < (*con).In.Size.Y; y++ {
			rn := con.sameAsOuput(x, y)
			for z := 0; z < (*con).In.Size.Z; z++ {
				sumError := 0.0
				for i := rn.MinX; i <= rn.MaxX; i++ {
					minX := i * (*con).Stride
					for j := rn.MinY; j <= rn.MaxY; j++ {
						minY := j * (*con).Stride
						for k := rn.MinZ; k <= rn.MaxZ; k++ {
							weightApplied := (*con).Kernels[k].Get(x-minX, y-minY, z)
							sumError += weightApplied * (*nextLayerGrad).Get(i, j, k)
							(*con).LocalDeltas[k].AddToGrad(x-minX, y-minY, z, (*con).In.Get(x, y, z)*(*nextLayerGrad).Get(i, j, k))
						}
					}
				}
				(*con).DeltaWeightsComponent.Set(x, y, z, sumError)
			}
		}
	}
}

// UpdateWeights - update convolutional layer's weights
func (con *ConvLayer) UpdateWeights() {
	for a := 0; a < len((*con).Kernels); a++ {
		for i := 0; i < (*con).KernelSize; i++ {
			for j := 0; j < (*con).KernelSize; j++ {
				for z := 0; z < (*con).In.Size.Z; z++ {
					grad := con.LocalDeltas[a].Get(i, j, z)

					prevDW := (*con).PreviousKernelsDeltas[a].Get(i, j, z)
					dw := (1.0-lp.Momentum)*(-1.0*(lp.LearningRate*grad.Grad*1.0)) + lp.Momentum*prevDW

					(*con).PreviousKernelsDeltas[a].Set(i, j, z, dw)
					(*con).Kernels[a].SetAdd(i, j, z, dw)

					con.LocalDeltas[a].Set(i, j, z, grad)
				}
			}
		}
	}
}

// PrintOutput - print convolutional layer's output
func (con *ConvLayer) PrintOutput() {
	fmt.Println("Printing Convolutional Layer output...")
	(*con).Out.Print()
}

// PrintWeights - print convolutional layer's weights
func (con *ConvLayer) PrintWeights() {
	fmt.Println("Printing Convolutional Layer kernels...")
	for i := range (*con).Kernels {
		fmt.Printf("Kernel #%v\n", i)
		(*con).Kernels[i].Print()
	}
}

// PrintGradients - print convolutional layer's gradients
func (con *ConvLayer) PrintGradients() {
	fmt.Println("Printing Convolutional Layer gradients-weights...")
	(*con).DeltaWeightsComponent.Print()
}

// SetActivationFunc - sets activation function for layer
func (con *ConvLayer) SetActivationFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set activation function for convolutional layer")
}

// SetActivationDerivativeFunc sets derivative of activation function
func (con *ConvLayer) SetActivationDerivativeFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set derivative of activation function for convolutional layer")
}

// GetType - return "conv" as layer's type
func (con *ConvLayer) GetType() string {
	return "conv"
}

// GetStride - get stride of layer
func (con *ConvLayer) GetStride() int {
	return con.Stride
}

// GetKernelSize - return kernel size
func (con *ConvLayer) GetKernelSize() int {
	return con.KernelSize
}

func (con *ConvLayer) mapToInput(i, j, k int) (x int, y int, z int) {
	return i * (*con).Stride, j * (*con).Stride, k
}

// Range - struct for reshaping data indecies
type Range struct {
	MinX, MaxX int
	MinY, MaxY int
	MinZ, MaxZ int
}

// sameAsOuput - reshape convolutional layer's output
func (con *ConvLayer) sameAsOuput(x, y int) Range {
	a := float64(x)
	b := float64(y)
	return Range{
		MinX: u.NormalizeRange((a-float64((*con).KernelSize)+1.0)/float64((*con).Stride), (*con).Out.Size.X, true),
		MinY: u.NormalizeRange((b-float64((*con).KernelSize)+1.0)/float64((*con).Stride), (*con).Out.Size.Y, true),
		MinZ: 0,
		MaxX: u.NormalizeRange(a/float64((*con).Stride), (*con).Out.Size.X, false),
		MaxY: u.NormalizeRange(b/float64((*con).Stride), (*con).Out.Size.Y, false),
		MaxZ: len((*con).Kernels) - 1,
	}
}
