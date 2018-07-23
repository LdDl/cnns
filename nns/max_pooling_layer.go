package nns

import (
	"cnns_vika/utils/u"
	"fmt"
	"math"
	"sort"
)

// MaxPoolingLayer is Max Pooling layer structure
// In - input data
// Out - output data
// PooledIndecies - indecies of pooled values
// Stride - striding step
// LocalGradients - gradients
type MaxPoolingLayer struct {
	In             *Tensor
	Out            *Tensor
	PooledIndecies *Tensor
	LocalGradients *Tensor
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
		LocalGradients: NewTensorEmpty(inputWidth, inputHeight, inputDepth),
		StrideWidth:    poolWidth,
		StrideHeight:   poolHeight,
	}
	// log.Println(newLayer.In.X, newLayer.In.Y)
	// log.Println(newLayer.Out.X, newLayer.Out.Y)
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
	fmt.Println("Printing Max Pooling Layer local gradients...")
	(*maxpool).LocalGradients.Print()
}

// PrintSumGradWeights - print maxpool layer's summ of grad*weight
func (maxpool *MaxPoolingLayer) PrintSumGradWeights() {

}

// GetGradients - get max pooling layer's gradients
func (maxpool *MaxPoolingLayer) GetGradients() *Tensor {
	return (*maxpool).LocalGradients
}

// CalculateGradients - calculate max pooling layer's gradients
func (maxpool *MaxPoolingLayer) CalculateGradients(nextLayerGrad *Tensor) {
	sort.Sort((*maxpool).PooledIndecies)
	for i := 0; i < (*maxpool).PooledIndecies.Size(); i++ {
		mappedIndex := (*maxpool).PooledIndecies.Data[i]
		sumweightgrad := nextLayerGrad.GetValue(i, 0, 0)
		(*maxpool).LocalGradients.SetValue(int(mappedIndex), 0, 0, sumweightgrad)
	}
	(*maxpool).LocalGradients.Print()
}

// SameAsOuput - reshape max pooling layer's output
func (maxpool *MaxPoolingLayer) SameAsOuput(x, y int) RangeP {
	a := float64(x)
	b := float64(y)
	return RangeP{
		MinX: u.NormalizeRange((a-float64((*maxpool).StrideWidth)+1.0)/float64((*maxpool).StrideWidth), (*maxpool).Out.X, true),
		MinY: u.NormalizeRange((b-float64((*maxpool).StrideHeight)+1.0)/float64((*maxpool).StrideHeight), (*maxpool).Out.Y, true),
		MinZ: 0,
		MaxX: u.NormalizeRange(a/float64((*maxpool).StrideWidth), (*maxpool).Out.X, false),
		MaxY: u.NormalizeRange(b/float64((*maxpool).StrideHeight), (*maxpool).Out.Y, false),
		MaxZ: (*maxpool).Out.Z - 1,
	}
}

// RangeP is struct for reshaping arrays
type RangeP struct {
	MinX, MinY, MinZ int
	MaxX, MaxY, MaxZ int
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
	(*maxpool).PooledIndecies = Pool((*maxpool).In, (*maxpool).Out, (*maxpool).StrideWidth, (*maxpool).StrideHeight, "max")
}
