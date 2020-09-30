package cnns

import (
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// Convolve2D Convolution between a kernel and a matrix. See ref. https://en.wikipedia.org/wiki/Kernel_(image_processing)#Convolution
/*
	matrix - source matrix
	kernel - kernel itself
	channels - number of input channels
	stride - step
*/
func Convolve2D(matrix, kernel *mat.Dense, channels, stride int) (*mat.Dense, error) {
	kernelR, kernelC := kernel.Dims()
	matrixR, matrixC := matrix.Dims()
	out := &mat.Dense{}
	for c := 0; c < channels; c++ {

		partialMatrix := ExtractChannel(matrix, matrixR, matrixC, channels, c)
		partialKernel := ExtractChannel(kernel, kernelR, kernelC, channels, c)
		partialConvolve, err := convolve2D(partialMatrix, partialKernel, stride)
		if err != nil {
			return nil, errors.Wrap(err, "Can't call Convolve2D()")
		}
		if out.IsEmpty() {
			out = partialConvolve
		} else {
			out.Add(out, partialConvolve)
		}
	}
	return out, nil
}

// convolve2D Convolution between a kernel and a matrix. See ref. Convolve2D()
func convolve2D(matrix, kernel *mat.Dense, stride int) (*mat.Dense, error) {
	kernelR, kernelC := kernel.Dims()
	matrixR, matrixC := matrix.Dims()
	outRows := (matrixR-kernelR)/stride + 1
	outCols := (matrixC-kernelC)/stride + 1
	im2col := Im2Col(matrix, kernelR, kernelC, stride)
	kernelAsVector := Flatten(kernel)
	out := &mat.Dense{}
	out.Mul(kernelAsVector, im2col.T())
	out, err := Reshape(out, outRows, outCols)
	if err != nil {
		return nil, errors.Wrap(err, "Can't call unexported convolve2D()")
	}
	return out, nil
}
