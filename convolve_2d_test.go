package cnns

import (
	"math"
	"math/rand"
	"testing"

	"github.com/LdDl/cnns/tensor"
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

	oldTensorSingle       = tensor.NewConvolveLayer(8, 9, 1, 1, 3, 1)
	benchNumFiltersSingle = 1
	benchStrideSingle     = 1
)

func BenchmarkConvolve2DSingleSmall(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := Convolve2D(benchSingleChannelImage, benchKernelSingle, benchNumFiltersSingle, benchStrideSingle)
		if err != nil {
			panic(err)
		}
	}
}

func BenchmarkNaiveConvolveSmall(b *testing.B) {
	oldTensorSingle.In.Data = benchSingleChannelImage.RawMatrix().Data
	oldTensorSingle.Kernels[0].Data = benchKernelSingle.RawMatrix().Data
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		oldTensorSingle.NaiveConv()
	}
}

func BenchmarkConvolve2DSingleBig(b *testing.B) {
	width, height := 512, 512
	kernelSize := 7
	data := make([]float64, width*height)
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			data[i*width+j] = rand.Float64() - 0.5
		}
	}
	bigMat := mat.NewDense(height, width, data)

	kernelData := make([]float64, kernelSize*kernelSize)
	for i := 0; i < kernelSize; i++ {
		for j := 0; j < kernelSize; j++ {
			kernelData[i*kernelSize+j] = rand.Float64() - 0.5
		}
	}
	bigKernel := mat.NewDense(kernelSize, kernelSize, kernelData)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := Convolve2D(bigMat, bigKernel, benchNumFiltersSingle, benchStrideSingle)
		if err != nil {
			panic(err)
		}
	}
}

func BenchmarkNaiveConvolveBig(b *testing.B) {
	width, height := 512, 512
	kernelSize := 7
	data := make([]float64, width*height)
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			data[i*width+j] = rand.Float64() - 0.5
		}
	}
	bigMat := tensor.NewConvolveLayer(width, height, 1, 1, kernelSize, 1)

	kernelData := make([]float64, kernelSize*kernelSize)
	for i := 0; i < kernelSize; i++ {
		for j := 0; j < kernelSize; j++ {
			kernelData[i*kernelSize+j] = rand.Float64() - 0.5
		}
	}

	bigMat.In.Data = data
	bigMat.Kernels[0].Data = kernelData
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bigMat.NaiveConv()
	}
}

func TestConvAndNaiveSmall(t *testing.T) {
	data := tensor.NewConvolveLayer(8, 9, 1, 1, 3, 1)
	data.In.Data = benchSingleChannelImage.RawMatrix().Data
	data.Kernels[0].Data = benchKernelSingle.RawMatrix().Data
	data.NaiveConv()

	outConv2d, err := Convolve2D(benchSingleChannelImage, benchKernelSingle, benchNumFiltersSingle, benchStrideSingle)
	if err != nil {
		t.Error(err)
	}

	dataConv2d := outConv2d.RawMatrix().Data

	for i := range data.Out.Data {
		if (data.Out.Data[i] - dataConv2d[i]) >= 0.000000000001 {
			t.Errorf("Pos #%d. Value naive: %f, Value gonum: %f, Difference: %f", i, data.Out.Data[i], dataConv2d[i], math.Abs(data.Out.Data[i]-dataConv2d[i]))
		}
	}
}

func TestConvAndNaiveBig(t *testing.T) {
	width, height := 512, 512
	kernelSize := 7

	data := make([]float64, width*height)
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			data[i*width+j] = rand.Float64() - 0.5
		}
	}
	kernelData := make([]float64, kernelSize*kernelSize)
	for i := 0; i < kernelSize; i++ {
		for j := 0; j < kernelSize; j++ {
			kernelData[i*kernelSize+j] = rand.Float64() - 0.5
		}
	}

	oldWay := tensor.NewConvolveLayer(width, height, 1, 1, kernelSize, 1)
	oldWay.In.Data = data
	oldWay.Kernels[0].Data = kernelData
	oldWay.NaiveConv()

	newWay, err := Convolve2D(mat.NewDense(height, width, data), mat.NewDense(kernelSize, kernelSize, kernelData), benchNumFiltersSingle, benchStrideSingle)
	if err != nil {
		t.Error(err)
	}

	dataNewWay := newWay.RawMatrix().Data

	for i := range oldWay.Out.Data {
		if (oldWay.Out.Data[i] - dataNewWay[i]) >= 0.000000000001 {
			t.Errorf("Pos #%d. Value naive: %f, Value gonum: %f, Difference: %f", i, oldWay.Out.Data[i], dataNewWay[i], math.Abs(oldWay.Out.Data[i]-dataNewWay[i]))
		}
	}
}

func TestConvolve2DRGB(t *testing.T) {
	imgRows := 5
	imgCols := 5
	kernelSize := 3

	redChannel := mat.NewDense(imgRows, imgCols, []float64{
		1, 0, 1, 0, 2,
		1, 1, 3, 2, 1,
		1, 1, 0, 1, 1,
		2, 3, 2, 1, 3,
		0, 2, 0, 1, 0,
	})
	greenChannel := mat.NewDense(imgRows, imgCols, []float64{
		1, 0, 0, 1, 0,
		2, 0, 1, 2, 0,
		3, 1, 1, 3, 0,
		0, 3, 0, 3, 2,
		1, 0, 3, 2, 1,
	})
	blueChannel := mat.NewDense(imgRows, imgCols, []float64{
		2, 0, 1, 2, 1,
		3, 3, 1, 3, 2,
		2, 1, 1, 1, 0,
		3, 1, 3, 2, 0,
		1, 1, 2, 1, 1,
	})

	kernel1R := mat.NewDense(kernelSize, kernelSize, []float64{
		0, 1, 0,
		0, 0, 2,
		0, 1, 0,
	})
	kernel1G := mat.NewDense(kernelSize, kernelSize, []float64{
		2, 1, 0,
		0, 0, 0,
		0, 3, 0,
	})
	kernel1B := mat.NewDense(kernelSize, kernelSize, []float64{
		1, 0, 0,
		1, 0, 0,
		0, 0, 2,
	})

	stride := 1
	channels := 3

	correct := [][]float64{
		[]float64{19, 13, 15},
		[]float64{28, 16, 20},
		[]float64{23, 18, 25},
	}

	img2 := &mat.Dense{}
	img2.Stack(redChannel, greenChannel)
	imageRGB := &mat.Dense{}
	imageRGB.Stack(img2, blueChannel)

	kernel1 := &mat.Dense{}
	kernel1.Stack(kernel1R, kernel1G)
	kernelRGB := &mat.Dense{}
	kernelRGB.Stack(kernel1, kernel1B)

	outMatrix, err := Convolve2D(imageRGB, kernelRGB, channels, stride)
	if err != nil {
		t.Error(err)
		return
	}

	rows, _ := outMatrix.Dims()
	if rows != len(correct) {
		t.Errorf("'rows' must be %d, but got %d after Convolve2D()", len(correct), rows)
		return
	}

	for r := 0; r < rows; r++ {
		rowView := outMatrix.RawRowView(r)
		if len(rowView) != len(correct[r]) {
			t.Errorf("'cols' in single row should be %d, but got %d after Convolve2D()", len(correct[r]), len(rowView))
			return
		}
		for c := range rowView {
			if rowView[c] != correct[r][c] {
				t.Errorf("Element in position r = %d, c = %d should be %f, but got %f", r, c, correct[r][c], rowView[c])
				return
			}
		}
	}

}

func RoundPlaces(v float64, places int) float64 {
	shift := math.Pow(10, float64(places))
	return Round(v*shift) / shift
}

func Round(v float64) float64 {
	if v >= 0 {
		return math.Floor(v + 0.5)
	}
	return math.Ceil(v - 0.5)
}
