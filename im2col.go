package cnns

import "gonum.org/v1/gonum/mat"

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
	res := make([]float64, colSize*rows*cols)

	idx := 0
	// var wg sync.WaitGroup
	for y := 0; y < rows; y++ {
		// 	wg.Add(1)
		// go makeCol(matrix, k, stride, y, cols, idx, res, &wg)
		makeCol(matrix, kernelRows, kernelCols, stride, y, cols, idx, res)
		idx += colSize * cols
	}
	// wg.Wait()
	return mat.NewDense(rows*cols, colSize, res)
}

// makeCol Slice matrix for Im2Col(). See ref. Im2Col()
func makeCol(matrix *mat.Dense, kernelSizeR, kernelSizeC, s, y, cols, sIdx int, res []float64) {
	for x := 0; x < cols; x++ {
		startY := y * s
		startX := x * s
		part := matrix.Slice(startY, startY+kernelSizeR, startX, startX+kernelSizeC)
		for i := 0; i < kernelSizeR; i++ {
			for j := 0; j < kernelSizeC; j++ {
				res[sIdx] = part.At(i, j)
				sIdx++
			}
		}
	}
}
