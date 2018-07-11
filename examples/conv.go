package examples

import (
	"cnns_vika/nns"
	"fmt"
)

var (
	height = 9
	width  = 8
	depth  = 1
)

// CheckConvolutional - проверка операции свёртки
func CheckConvolutional() {
	var matrix = [][][]float64{
		[][]float64{
			[]float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
			[]float64{9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
			[]float64{17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0},
			[]float64{25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0},
			[]float64{33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0},
			[]float64{41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0},
			[]float64{49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0},
			[]float64{57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0},
			[]float64{65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0},
		},
	}

	var image = nns.NewTensorEmpty(8, 9, 1) // w,h,d
	image.Set(&matrix)
	image.Print()

	var kernel = nns.NewTensorEmpty(3, 3, 1)
	var kernelMatrix = [][][]float64{
		[][]float64{
			[]float64{0.10466028797962, 0.440509088045012, 0.16456005321849},
			[]float64{-0.06228581281302, -0.075362502928734, 0.186823072867109},
			[]float64{-0.434362980782524, -0.343480745267209, -0.403030481085515},
		},
	}

	kernel.Set(&kernelMatrix)
	kernel.Print()

	fmt.Println("Convolve:")
	nns.Convolve2D(image, kernel, 1, 1, 0, 0)
}
