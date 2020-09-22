package cnns

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

var (
	benchSingleChannelImage = mat.NewDense(9, 8, []float64{
		-0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
		-0.9, -0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
		-0.17, 0.18, -0.19, 0.20, 0.21, 0.22, 0.23, 0.24,
		-0.25, 0.26, 0.27, -0.28, 0.29, 0.30, 0.31, 0.32,
		-0.33, 0.34, 0.35, 0.36, -0.37, 0.38, 0.39, 0.40,
		-0.41, 0.42, 0.43, 0.44, 0.45, -0.46, 0.47, 0.48,
		-0.49, 0.50, 0.51, 0.52, 0.53, 0.54, -0.55, 0.56,
		-0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, -0.64,
		-0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72,
	})

	benchKernelSingle = mat.NewDense(3, 3, []float64{
		0.10466029, -0.06228581, -0.43436298,
		0.44050909, -0.07536250, -0.34348075,
		0.16456005, 0.18682307, -0.40303048,
	})

	benchNumFiltersSingle = 1
	benchStrideSingle     = 1
)

func BenchmarkConvolve2DSingle(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := Convolve2D(benchSingleChannelImage, benchKernelSingle, benchNumFiltersSingle, benchStrideSingle)
		if err != nil {
			panic(err)
		}
	}
}