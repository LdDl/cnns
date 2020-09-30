package cnns

import (
	"fmt"
	"math"
	"strings"

	"github.com/LdDl/cnns/tensor"
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

type poolingType int

const (
	poolMAX = iota + 1
	poolMIN
	poolAVG
)

func (pt poolingType) String() string {
	switch pt {
	case poolMAX:
		return "max"
	case poolMIN:
		return "min"
	case poolAVG:
		return "avg"
	default:
		return fmt.Sprintf("Pooling type #%d is not defined", pt)
	}
}

type zeroPaddingType int

const (
	poolVALID = iota + 1
	poolSAME
)

func (zpt zeroPaddingType) String() string {
	switch zpt {
	case poolVALID:
		return "valid"
	case poolSAME:
		return "same"
	default:
		return fmt.Sprintf("Zero padding type #%d is not defined", zpt)
	}
}

// PoolingLayer Pooling layer structure
/*
	Oj - Input data
	Ok - Output data
	LocalDelta - Gradients
*/
type PoolingLayer struct {
	Oj           *mat.Dense
	Ok           *mat.Dense
	Masks        *mat.Dense
	Stride       int
	ExtendFilter int
	masksIndices [][][2]int

	OutputSize *tensor.TDsize
	inputSize  *tensor.TDsize

	PoolingType poolingType
	ZeroPadding zeroPaddingType
	trainMode   bool
}

// NewPoolingLayer Constructor for pooling layer.
func NewPoolingLayer(inSize *tensor.TDsize, stride, extendFilter int, poolingType string, zeroPad string) Layer {
	newLayer := &PoolingLayer{
		inputSize:    inSize,
		Oj:           mat.NewDense(inSize.X, inSize.Y, nil),
		Ok:           &mat.Dense{},
		Masks:        mat.NewDense(inSize.X, inSize.Y, nil),
		Stride:       stride,
		ExtendFilter: extendFilter,
		trainMode:    false,
	}

	switch strings.ToLower(zeroPad) {
	case "same":
		newLayer.ZeroPadding = poolSAME
		// If zero padding truly needed?
		if (inSize.X-extendFilter)%stride != 0 {
			newLayer.OutputSize = &tensor.TDsize{
				X: int(math.Ceil(float64(inSize.X-extendFilter)/float64(stride) + 1)),
				Y: int(math.Ceil(float64(inSize.Y-extendFilter)/float64(stride) + 1)),
				Z: inSize.Z,
			}
		} else {
			// Ignore predefined zeroPad value.
			newLayer.ZeroPadding = poolVALID
			newLayer.OutputSize = &tensor.TDsize{
				X: (inSize.X-extendFilter)/stride + 1,
				Y: (inSize.Y-extendFilter)/stride + 1,
				Z: inSize.Z,
			}
		}
		break
	default: // Default is 'VALID'
		newLayer.ZeroPadding = poolVALID
		newLayer.OutputSize = &tensor.TDsize{
			X: (inSize.X-extendFilter)/stride + 1,
			Y: (inSize.Y-extendFilter)/stride + 1,
			Z: inSize.Z,
		}
		break
	}
	switch strings.ToLower(poolingType) {
	case "max":
		newLayer.PoolingType = poolMAX
		break
	case "min":
		newLayer.PoolingType = poolMIN
		break
	case "avg":
		newLayer.PoolingType = poolAVG
		break
	default:
		fmt.Printf("Warning: type '%s' for pooling layer is not supported. Use 'max', 'min' or 'avg'\n", poolingType)
		newLayer.PoolingType = poolMAX
		break
	}

	return newLayer
}

// SetCustomWeights Set user's weights (make it carefully) for pooling layer
func (pool *PoolingLayer) SetCustomWeights(t []*mat.Dense) {
	fmt.Println("There are no weights for pooling layer")
}

// GetInputSize Returns dimensions of incoming data for pooling layer
func (pool *PoolingLayer) GetInputSize() *tensor.TDsize {
	return pool.inputSize
}

// GetOutputSize Returns output size (dimensions) of pooling layer
func (pool *PoolingLayer) GetOutputSize() *tensor.TDsize {
	return pool.OutputSize
}

// GetActivatedOutput Returns pooling layer's output
func (pool *PoolingLayer) GetActivatedOutput() *mat.Dense {
	return pool.Ok
}

// GetWeights Returns pooling layer's weights
func (pool *PoolingLayer) GetWeights() []*mat.Dense {
	fmt.Println("There are no weights for pooling layer")
	return nil
}

// GetGradients Returns pooling layer's gradients
func (pool *PoolingLayer) GetGradients() *mat.Dense {
	return pool.Masks
}

// FeedForward Feed data to pooling layer
func (pool *PoolingLayer) FeedForward(input *mat.Dense) error {
	pool.Oj = input
	if pool.ZeroPadding == poolSAME {
		matrixR, matrixC := pool.Oj.Dims()
		stacked := &mat.Dense{}
		for c := 0; c < pool.OutputSize.Z; c++ {
			// Add padding for each channel
			partialMatrix := ExtractChannel(pool.Oj, matrixR, matrixC, pool.OutputSize.Z, c) //pool.Oj.Slice(c*matrixC, matrixR/pool.OutputSize.Z+c*matrixC, 0, matrixC).(*mat.Dense)
			padded := ZeroPadding(partialMatrix, 1)
			if stacked.IsEmpty() {
				stacked = padded
			} else {
				t := &mat.Dense{}
				t.Stack(stacked, padded)
				stacked = t
			}
		}
		pool.Oj = stacked
	}
	pool.doActivation()
	return nil
}

// DoActivation Pooling layer's output activation
func (pool *PoolingLayer) doActivation() {

	pool.Ok, pool.Masks, pool.masksIndices = Pool2D(pool.Oj, pool.OutputSize.X, pool.OutputSize.Y, pool.OutputSize.Z, pool.ExtendFilter, pool.Stride, pool.PoolingType, true)
}

// CalculateGradients Evaluate pooling layer's gradients
func (pool *PoolingLayer) CalculateGradients(errorsDense *mat.Dense) error {
	errorsReshaped := errorsDense
	var err error
	okR, okC := pool.Ok.Dims()
	errR, errC := errorsDense.Dims()
	if okR != errR || okC != errC {
		errorsReshaped, err = Reshape(errorsDense, okR, okC)
		if err != nil {
			return errors.Wrap(err, "Can't call CalculateGradients() on pooling layer while reshaping incoming gradients")
		}
	}
	stride := pool.Stride
	windowSize := pool.ExtendFilter
	channels := pool.OutputSize.Z
	errorsRows, errorsCols := errorsReshaped.Dims()
	maskR, maskC := pool.Masks.Dims()
	maskIndicesSplit := len(pool.masksIndices) / channels
	for c := 0; c < channels; c++ {
		partialErrors := ExtractChannel(errorsReshaped, errorsRows, errorsCols, channels, c)
		partialErrRows, partialErrCols := partialErrors.Dims()
		partialMask := ExtractChannel(pool.Masks, maskR, maskC, channels, c)
		partialMaskIndices := pool.masksIndices[c*maskIndicesSplit : maskIndicesSplit+c*maskIndicesSplit]
		for y := 0; y < partialErrRows; y++ {
			startYi := y * stride
			startYj := startYi + windowSize
			for x := 0; x < partialErrCols; x++ {
				startX := x * stride
				part := partialMask.Slice(startYi, startYj, startX, startX+windowSize).(*mat.Dense)
				// fmt.Println(len(partialMaskIndices), y, x)
				part.Set(partialMaskIndices[y][x][0], partialMaskIndices[y][x][1], partialErrors.At(y, x))
			}
		}
	}
	return nil
}

// UpdateWeights Just to point, that pooling layer does NOT updating weights
func (pool *PoolingLayer) UpdateWeights() {
	// "There are no weights to update for pooling layer"
}

// PrintOutput Pretty print pooling layer's output
func (pool *PoolingLayer) PrintOutput() {
	fmt.Println("Printing Pooling Layer output...")
}

// PrintWeights Just to point, that pooling layer has not gradients
func (pool *PoolingLayer) PrintWeights() {
	fmt.Println("There are no weights for pooling layer")
}

// SetActivationFunc Set activation function for layer
func (pool *PoolingLayer) SetActivationFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set activation function for pooling layer")
}

// SetActivationDerivativeFunc Set derivative of activation function
func (pool *PoolingLayer) SetActivationDerivativeFunc(f func(v float64) float64) {
	// Nothing here. Just for interface.
	fmt.Println("You can not set derivative of activation function for pooling layer")
}

// GetStride Returns stride of layer
func (pool *PoolingLayer) GetStride() int {
	return pool.Stride
}

// GetType Returns "pool" as layer's type
func (pool *PoolingLayer) GetType() string {
	return "pool"
}
