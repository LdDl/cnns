package cnns

import (
	"gonum.org/v1/gonum/mat"
)

// Im2Col Convert image to column-based vector. See ref. http://cs231n.stanford.edu/slides/2016/winter1516_lecture11.pdf -> Slide "Implementing Convolutions: im2col"
/*
	matrix - source matrix
	kernelRows - kernel's height
	kernelCols - kernel's width
	stride - step
*/
func Im2Col(matrix *mat.Dense, kernelRows, kernelCols, stride int) *mat.Dense {
	colSize := kernelRows * kernelCols
	r, c := matrix.Dims()
	rows := (r-kernelRows)/stride + 1
	cols := (c-kernelCols)/stride + 1
	flattenMatrix := make([]float64, colSize*rows*cols)

	idx := 0
	for y := 0; y < rows; y++ {
		startY := y * stride
		makeCol(matrix, kernelRows, kernelCols, stride, startY, startY+kernelRows, cols, idx, flattenMatrix)
		idx += colSize * cols
	}
	return mat.NewDense(rows*cols, colSize, flattenMatrix)
}

// makeCol Slice matrix for Im2Col(). See ref. Im2Col()
func makeCol(matrix *mat.Dense, kernelSizeR, kernelSizeC, stride, startY, shiftY, cols, colIdx int, newFlattenMatrix []float64) {
	for x := 0; x < cols; x++ {
		startX := x * stride
		part := matrix.Slice(startY, shiftY, startX, startX+kernelSizeC).(*mat.Dense)
		for i := 0; i < kernelSizeR; i++ {
			for j := 0; j < kernelSizeC; j++ {
				newFlattenMatrix[colIdx] = part.At(i, j)
				colIdx++
			}
		}
	}
}
