package cnns

import (
	"math"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// Pool2D Pooling of matrix with defined window: windowSize/stride/pooling_type. See ref. https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
/*
	matrix - source matrix
	outRows - number of output rows
	outCols - number of output columns
	channels - number of input channels
	windowsSize - size of "kernel"
	stride - step
	ptype - type of pooling (max/min/avg)
	returnMasks - return masks ??? (for training mode)
*/
func Pool2D(matrix *mat.Dense, outRows, outCols, channels, windowSize, stride int, ptype poolingType, returnMasks bool) (*mat.Dense, *mat.Dense, [][][2]int) {
	sourceR, sourceC := matrix.Dims()
	flattenSlice := []float64{}

	if !returnMasks {
		for c := 0; c < channels; c++ {
			partialSlice := make([]float64, outRows*outCols)
			tmpMatrix := matrix.Slice(c*sourceC, sourceR/channels+c*sourceC, 0, sourceC).(*mat.Dense)
			wg := sync.WaitGroup{}
			for y := 0; y < outRows; y++ {
				startYi := y * stride
				startYj := startYi + windowSize
				wg.Add(1)
				go pool2D(tmpMatrix, partialSlice, y, startYi, startYj, outCols, windowSize, stride, ptype, &wg)
			}
			wg.Wait()
			flattenSlice = append(flattenSlice, partialSlice...)
		}
		return mat.NewDense(outRows*channels, outCols, flattenSlice), nil, nil
	}

	masks := &mat.Dense{}
	masksIndices := [][][2]int{}
	for c := 0; c < channels; c++ {
		partialSlice := make([]float64, outRows*outCols)
		tmpMatrix := matrix.Slice(c*sourceC, sourceR/channels+c*sourceC, 0, sourceC).(*mat.Dense)
		tmpR, tmpC := tmpMatrix.Dims()
		partialMasks := mat.NewDense(tmpR, tmpC, nil)
		partialMasks.Zero()

		partialMasksIndices := make([][][2]int, outRows)
		wg := sync.WaitGroup{}
		for y := 0; y < outRows; y++ {
			wg.Add(1)
			startYi := y * stride
			startYj := startYi + windowSize
			go pool2DWithMasks(tmpMatrix, partialMasks, partialMasksIndices, partialSlice, y, startYi, startYj, outCols, windowSize, stride, ptype, &wg)
		}
		wg.Wait()

		if masks.IsEmpty() {
			masks = partialMasks
		} else {
			tmp := &mat.Dense{}
			tmp.Stack(masks, partialMasks)
			masks = tmp
		}
		flattenSlice = append(flattenSlice, partialSlice...)
		masksIndices = append(masksIndices, partialMasksIndices...)
	}

	return mat.NewDense(outRows*channels, outCols, flattenSlice), masks, masksIndices
}

// pool2D See ref. Pool2D()
func pool2D(matrix *mat.Dense, flattenMatrix []float64, y, startYi, startYj, outCols, windowSize, stride int, ptype poolingType, wg *sync.WaitGroup) {
	for x := 0; x < outCols; x++ {
		startX := x * stride
		part := matrix.Slice(startYi, startYj, startX, startX+windowSize)
		switch ptype {
		case poolMAX:
			flattenMatrix[y*outCols+x] = maxPool(part)
			break
		case poolMIN:
			panic("poolMIN is not implemented")
		case poolAVG:
			panic("poolAVG is not implemented")
		default:
			panic("default behaviour for pool_%TYPE% is not implemented")
		}
	}
	wg.Done()
}

func pool2DWithMasks(matrix, masks *mat.Dense, partialMasksIndices [][][2]int, flattenMatrix []float64, y, startYi, startYj, outCols, windowSize, stride int, ptype poolingType, wg *sync.WaitGroup) {
	partialMasks := make([][2]int, outCols)
	for x := 0; x < outCols; x++ {
		startX := x * stride
		part := matrix.Slice(startYi, startYj, startX, startX+windowSize).(*mat.Dense)
		partMask := masks.Slice(startYi, startYj, startX, startX+windowSize).(*mat.Dense)
		switch ptype {
		case poolMAX:
			maxX, maxY, k := maxPoolIdx(part)
			partMask.Set(maxX, maxY, 1)
			partialMasks[x] = [2]int{maxX, maxY}
			flattenMatrix[y*outCols+x] = k
			break
		case poolMIN:
			panic("poolMIN is not implemented (with masks)")
		case poolAVG:
			panic("poolAVG is not implemented (with masks)")
		default:
			panic("default behaviour for pool_%TYPE% is not implemented (with masks)")
		}
	}
	partialMasksIndices[y] = partialMasks
	wg.Done()
}

func maxPoolIdx(m mat.Matrix) (int, int, float64) {
	max := math.Inf(-1)
	maxi := -1
	maxj := -1
	rows, cols := m.Dims()
	for x := 0; x < rows; x++ {
		for y := 0; y < cols; y++ {
			val := m.At(x, y)
			if val > max {
				max = val
				maxi = x
				maxj = y
			}
		}
	}
	return maxi, maxj, max
}

func minPoolIdx(m *mat.Dense) (int, float64) {
	min := math.Inf(1)
	mini := -1
	raw := m.RawMatrix().Data
	for i := 0; i < len(raw); i++ {
		if raw[i] < min {
			min = raw[i]
			mini = i
		}
	}
	return mini, min
}

func maxPool(m mat.Matrix) float64 {
	return mat.Max(m)
}

func minPool(m mat.Matrix) float64 {
	return mat.Min(m)
}

func avgPool(m mat.Matrix) float64 {
	r, c := m.Dims()
	return mat.Sum(m) / float64(r*c)
}
