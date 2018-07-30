package nns

import (
	"cnns_vika/utils/u"
	"fmt"
	"math"
)

// MaxPoolingLayer is Max Pooling layer structure
// In - input data
// Out - output data
// PooledIndecies - indecies of pooled values
// Stride - striding step
// InputGradientsWeights - gradients
type MaxPoolingLayer struct {
	In                    Tensor
	Out                   Tensor
	PooledIndecies        Tensor
	InputGradientsWeights Tensor
	Stride                int
	ExtendFilter          int
}

func (maxpool *MaxPoolingLayer) SetActivationFunc(f func(v float64) float64) {
	//
}

func (maxpool *MaxPoolingLayer) SetActivationDerivativeFunc(f func(v float64) float64) {
	//
}

// NewMaxPoolingLayer - constructor for new MaxPooling layer.
func NewMaxPoolingLayer(stride, extendFilter int, inSize TDsize) *LayerStruct {
	newLayer := &MaxPoolingLayer{
		In:  NewTensor(inSize.X, inSize.Y, inSize.Z),
		Out: NewTensor((inSize.X-extendFilter)/stride+1, (inSize.Y-extendFilter)/stride+1, inSize.Z),
		InputGradientsWeights: NewTensor(inSize.X, inSize.Y, inSize.Z),
		Stride:                stride,
		ExtendFilter:          extendFilter,
	}
	return &LayerStruct{
		Layer: newLayer,
	}
}

func (maxpool *MaxPoolingLayer) OutSize() Point {
	return (*maxpool).Out.Size
}

func (maxpool *MaxPoolingLayer) mapToInput(out Point, z int) Point {
	return Point{
		X: out.X * (*maxpool).Stride,
		Y: out.Y * (*maxpool).Stride,
		Z: z,
	}
}

// SameAsOuput - reshape convolutional layer's output
func (maxpool *MaxPoolingLayer) SameAsOuput(x, y int) Range {
	a := float64(x)
	b := float64(y)
	return Range{
		MinX: u.NormalizeRange((a-float64((*maxpool).ExtendFilter)+1.0)/float64((*maxpool).Stride), (*maxpool).Out.Size.X, true),
		MinY: u.NormalizeRange((b-float64((*maxpool).ExtendFilter)+1.0)/float64((*maxpool).Stride), (*maxpool).Out.Size.Y, true),
		MinZ: 0,
		MaxX: u.NormalizeRange(a/float64((*maxpool).Stride), (*maxpool).Out.Size.X, false),
		MaxY: u.NormalizeRange(b/float64((*maxpool).Stride), (*maxpool).Out.Size.Y, false),
		MaxZ: (*maxpool).Out.Size.Z - 1,
	}
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
func (maxpool *MaxPoolingLayer) GetOutput() Tensor {
	return (*maxpool).Out
}

// FeedForward - feed data to max pooling layer
func (maxpool *MaxPoolingLayer) FeedForward(t *Tensor) {
	(*maxpool).In = (*t)
	(*maxpool).DoActivation()
}

// PrintGradients - print max pooling layer's gradients
func (maxpool *MaxPoolingLayer) PrintGradients() {
	fmt.Println("Printing Max Pooling Layer local gradients...")
	(*maxpool).InputGradientsWeights.Print()
}

// PrintSumGradWeights - print maxpool layer's summ of grad*weight
func (maxpool *MaxPoolingLayer) PrintSumGradWeights() {

}

// GetGradients - get max pooling layer's gradients
func (maxpool *MaxPoolingLayer) GetGradients() Tensor {
	return (*maxpool).InputGradientsWeights
}

// CalculateGradients - calculate max pooling layer's gradients
func (maxpool *MaxPoolingLayer) CalculateGradients(nextLayerGrad *Tensor) {
	for x := 0; x < (*maxpool).In.Size.X; x++ {
		for y := 0; y < (maxpool).In.Size.Y; y++ {
			rn := maxpool.SameAsOuput(x, y)
			for z := 0; z < (*maxpool).In.Size.Z; z++ {
				sumError := 0.0
				for i := rn.MinX; i <= rn.MaxX; i++ {
					minX := i * (*maxpool).Stride
					_ = minX
					for j := rn.MinY; j <= rn.MaxY; j++ {
						minY := j * (*maxpool).Stride
						_ = minY
						if (*maxpool).In.Get(x, y, z) == (*maxpool).Out.Get(i, j, z) {
							sumError += (*nextLayerGrad).Get(i, j, z)
						} else {
							sumError += 0
						}
					}
				}
				(*maxpool).InputGradientsWeights.Set(x, y, z, sumError)
			}
		}
	}
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
	for x := 0; x < (*maxpool).Out.Size.X; x++ {
		for y := 0; y < (*maxpool).Out.Size.Y; y++ {
			for z := 0; z < (*maxpool).In.Size.Z; z++ {
				mapped := maxpool.mapToInput(Point{X: x, Y: y, Z: 0}, 0)
				mval := -1.0 * math.MaxFloat64
				for i := 0; i < (*maxpool).ExtendFilter; i++ {
					for j := 0; j < (*maxpool).ExtendFilter; j++ {
						v := (*maxpool).In.Get(mapped.X+i, mapped.Y+j, z)
						if v > mval {
							mval = v
						}
					}
				}
				(*maxpool).Out.Set(x, y, z, mval)
			}
		}
	}
}
