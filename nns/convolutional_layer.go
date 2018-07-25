package nns

import (
	"cnns_vika/utils/u"
	"fmt"
	"math/rand"
)

// ConvLayer is convolutional layer structure
type ConvLayer struct {
	InputGradientsWeights Tensor
	In                    Tensor
	Out                   Tensor
	Kernels               []Tensor
	KernelsGradients      []TensorGradient
	Stride                int
	KernelSize            int
}

// NewConvLayer - constructor for new convolutional layer. You need to specify striding step, size (square) of kernel, amount of kernels, input size.
func NewConvLayer(stride, kernelSize, numberFilters int, inSize TDsize) *ConvLayer {
	newLayer := &ConvLayer{
		InputGradientsWeights: NewTensor(inSize.X, inSize.Y, inSize.Z),
		In:         NewTensor(inSize.X, inSize.Y, inSize.Z),
		Out:        NewTensor((inSize.X-kernelSize)/stride+1, (inSize.Y-kernelSize)/stride+1, numberFilters),
		Stride:     stride,
		KernelSize: kernelSize,
	}
	for a := 0; a < numberFilters; a++ {
		t := NewTensor(kernelSize, kernelSize, inSize.Z)
		for i := 0; i < kernelSize; i++ {
			for j := 0; j < kernelSize; j++ {
				for z := 0; z < inSize.Z; z++ {
					t.Set(i, j, z, rand.Float64()-0.5)
				}
			}
		}

		// hardcoded weights for testing purposes
		hcweights := [][][]float64{
			[][]float64{
				[]float64{0.10466029, -0.06228581, -0.43436298},
				[]float64{0.44050909, -0.07536250, -0.34348075},
				[]float64{0.16456005, 0.18682307, -0.40303048},
			},
		}
		t.CopyFrom(hcweights)

		newLayer.Kernels = append(newLayer.Kernels, t)

		for i := 0; i < numberFilters; i++ {
			t := NewTensorGradient(kernelSize, kernelSize, inSize.Z)
			newLayer.KernelsGradients = append(newLayer.KernelsGradients, t)
		}
	}
	return newLayer
}

func (con *ConvLayer) mapToInput(out Point, z int) Point {
	return Point{
		X: out.X * (*con).Stride,
		Y: out.Y * (*con).Stride,
		Z: z,
	}
}

type Range struct {
	MinX, MaxX int
	MinY, MaxY int
	MinZ, MaxZ int
}

// SameAsOuput - reshape convolutional layer's output
func (con *ConvLayer) SameAsOuput(x, y int) Range {
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
func (con *ConvLayer) GetOutput() Tensor {
	return (*con).Out
}

// FeedForward - feed data to convolutional layer
func (con *ConvLayer) FeedForward(t *Tensor) {
	(*con).In = (*t)
	(*con).DoActivation()
}

// PrintGradients - print convolutional layer's gradients
func (con *ConvLayer) PrintGradients() {
	fmt.Println("Printing Convolutional Layer gradients-weights...")
	(*con).InputGradientsWeights.Print()
}

// PrintSumGradWeights - print convolutional layer's summ of grad*weight
func (con *ConvLayer) PrintSumGradWeights() {
}

// GetGradients - get convolutional layer's gradients
func (con *ConvLayer) GetGradients() Tensor {
	return (*con).InputGradientsWeights
}

// CalculateGradients - calculate convolutional layer's gradients
func (con *ConvLayer) CalculateGradients(nextLayerGrad *Tensor) {
	for k := 0; k < len((*con).KernelsGradients); k++ {
		for i := 0; i < (*con).KernelSize; i++ {
			for j := 0; j < (*con).KernelSize; j++ {
				for z := 0; z < (*con).In.Size.Z; z++ {
					(*con).KernelsGradients[k].SetGrad(i, j, z, 0.0)
				}
			}
		}
	}

	for x := 0; x < (*con).In.Size.X; x++ {
		for y := 0; y < (*con).In.Size.Y; y++ {
			rn := con.SameAsOuput(x, y)
			for z := 0; z < (*con).In.Size.Z; z++ {
				sumError := 0.0
				for i := rn.MinX; i <= rn.MaxX; i++ {
					minX := i * (*con).Stride
					for j := rn.MinY; j <= rn.MaxY; j++ {
						minY := j * (*con).Stride
						for k := rn.MinZ; k <= rn.MaxZ; k++ {
							weightApplied := (*con).Kernels[k].Get(x-minX, y-minY, z)
							sumError += weightApplied * (*nextLayerGrad).Get(i, j, k)
							(*con).KernelsGradients[k].AddToGrad(x-minX, y-minY, z, (*con).In.Get(x, y, z)*(*nextLayerGrad).Get(i, j, k))
						}
					}
				}
				(*con).InputGradientsWeights.Set(x, y, z, sumError)
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
					w := (*con).Kernels[a].Get(i, j, z)
					grad := con.KernelsGradients[a].Get(i, j, z)
					w = UpdateWeight(w, &grad, 1.0)
					(*con).Kernels[a].Set(i, j, z, w)
					// con.KernelsGradients[a].Print()
					UpdateGradient(&grad)
					con.KernelsGradients[a].Set(i, j, z, grad)
				}
			}
		}
	}
}

// DoActivation - convolutional layer's output activation
func (con *ConvLayer) DoActivation() {
	for filter := 0; filter < len((*con).Kernels); filter++ {
		filterData := (*con).Kernels[filter]
		for x := 0; x < (*con).Out.Size.X; x++ {
			for y := 0; y < (*con).Out.Size.Y; y++ {
				mapped := con.mapToInput(Point{X: x, Y: y, Z: 0}, 0)
				sum := 0.0
				for i := 0; i < (*con).KernelSize; i++ {
					for j := 0; j < (*con).KernelSize; j++ {
						for z := 0; z < (*con).In.Size.Z; z++ {
							f := filterData.Get(i, j, z)
							v := (*con).In.Get(mapped.X+i, mapped.Y+j, z)
							sum += f * v
						}
					}
				}
				(*con).Out.Set(x, y, filter, sum)
			}
		}
	}
}
