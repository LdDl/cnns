package nns

import (
	"fmt"
	"math"

	"github.com/LdDl/cnns/utils/u"
)

// MaxPoolingLayer is Max Pooling layer structure
// In - Input data
// Out - Output data
// Stride - Striding step
// InputGradientsWeights - Gradients
type MaxPoolingLayer struct {
	In                    Tensor
	Out                   Tensor
	InputGradientsWeights Tensor
	Stride                int
	ExtendFilter          int
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

// SetCustomWeights - set user's weights (make it carefully)
func (maxpool *MaxPoolingLayer) SetCustomWeights(t *[]Tensor) {
	fmt.Println("There are no weights for pooling layer")
}

// OutSize - returns output size (dimensions)
func (maxpool *MaxPoolingLayer) OutSize() Point {
	return (*maxpool).Out.Size
}

// GetInputSize - returns input size (dimensions)
func (maxpool *MaxPoolingLayer) GetInputSize() Point {
	return (*maxpool).In.Size
}

// GetOutput - returns max pooling layer's output
func (maxpool *MaxPoolingLayer) GetOutput() Tensor {
	return (*maxpool).Out
}

// GetWeights - returns pooling layer's weights
func (maxpool *MaxPoolingLayer) GetWeights() []Tensor {
	fmt.Println("There are no weights for pooling layer")
	return []Tensor{}
}

// GetGradients - returns max pooling layer's gradients
func (maxpool *MaxPoolingLayer) GetGradients() Tensor {
	return (*maxpool).InputGradientsWeights
}

// FeedForward - feed data to max pooling layer
func (maxpool *MaxPoolingLayer) FeedForward(t *Tensor) {
	(*maxpool).In = (*t)
	(*maxpool).DoActivation()
}

// DoActivation - max pooling layer's output activation
func (maxpool *MaxPoolingLayer) DoActivation() {
	for x := 0; x < (*maxpool).Out.Size.X; x++ {
		for y := 0; y < (*maxpool).Out.Size.Y; y++ {
			for z := 0; z < (*maxpool).In.Size.Z; z++ {
				// mappedX, mappedY, _ := maxpool.mapToInput(x, y, 0)
				mappedX, mappedY := x*(*maxpool).Stride, y*(*maxpool).Stride
				mval := -1.0 * math.MaxFloat64
				for i := 0; i < (*maxpool).ExtendFilter; i++ {
					for j := 0; j < (*maxpool).ExtendFilter; j++ {
						v := (*maxpool).In.Get(mappedX+i, mappedY+j, z)
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

// CalculateGradients - calculate max pooling layer's gradients
func (maxpool *MaxPoolingLayer) CalculateGradients(nextLayerGrad *Tensor) {
	for x := 0; x < (*maxpool).In.Size.X; x++ {
		for y := 0; y < (maxpool).In.Size.Y; y++ {
			rn := maxpool.sameAsOuput(x, y)
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

// PrintOutput - print max pooling layer's output
func (maxpool *MaxPoolingLayer) PrintOutput() {
	fmt.Println("Printing Max Pooling Layer output...")
	(*maxpool).Out.Print()
}

// PrintWeights - just to point, that max pooling layer has not gradients
func (maxpool *MaxPoolingLayer) PrintWeights() {
	fmt.Println("There are no weights for pooling layer")
}

// PrintGradients - print max pooling layer's gradients
func (maxpool *MaxPoolingLayer) PrintGradients() {
	fmt.Println("Printing Max Pooling Layer local gradients...")
	(*maxpool).InputGradientsWeights.Print()
}

// SetActivationFunc - sets activation function for layer
func (maxpool *MaxPoolingLayer) SetActivationFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set activation function for pooling layer")
}

// SetActivationDerivativeFunc sets derivative of activation function
func (maxpool *MaxPoolingLayer) SetActivationDerivativeFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set derivative of activation function for pooling layer")
}

// GetStride - get stride of layer
func (maxpool *MaxPoolingLayer) GetStride() int {
	return maxpool.Stride
}

// GetKernelSize - return "conv" as layer's type
func (maxpool *MaxPoolingLayer) GetKernelSize() int {
	return maxpool.ExtendFilter
}

// GetType - return "maxpool" as layer's type
func (maxpool *MaxPoolingLayer) GetType() string {
	return "pool"
}

func (maxpool *MaxPoolingLayer) mapToInput(i, j, k int) (x int, y int, z int) {
	return i * (*maxpool).Stride, j * (*maxpool).Stride, k
}

// sameAsOuput - reshape convolutional layer's output
func (maxpool *MaxPoolingLayer) sameAsOuput(x, y int) Range {
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
