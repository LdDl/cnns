package examples

import (
	"fmt"
	"log"

	"github.com/LdDl/cnns/nns"
)

// ImportNet - example of how ImportFromFile(fname string) works
func ImportNet() {
	jsonName := "datasets/conv_net.json"
	var net nns.WholeNet
	err := net.ImportFromFile(jsonName, false)
	if err != nil {
		log.Panicln(err)
	}
	fmt.Printf("Layers:\n")
	for i := range net.Layers {
		fmt.Printf("%v weights:\n", net.Layers[i].GetType())
		net.Layers[i].PrintWeights()
	}

	// Result of output should be same as in "examples/conv.go" file
	var matrix = [][][]float64{
		[][]float64{
			[]float64{-0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
			[]float64{-0.9, -0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16},
			[]float64{-0.17, 0.18, -0.19, 0.20, 0.21, 0.22, 0.23, 0.24},
			[]float64{-0.25, 0.26, 0.27, -0.28, 0.29, 0.30, 0.31, 0.32},
			[]float64{-0.33, 0.34, 0.35, 0.36, -0.37, 0.38, 0.39, 0.40},
			[]float64{-0.41, 0.42, 0.43, 0.44, 0.45, -0.46, 0.47, 0.48},
			[]float64{-0.49, 0.50, 0.51, 0.52, 0.53, 0.54, -0.55, 0.56},
			[]float64{-0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, -0.64},
			[]float64{-0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72},
		},
	}
	var image = nns.NewTensor(8, 9, 1)
	image.SetData(matrix)
	fmt.Println("Image:")
	image.Print()
	fmt.Println("Weights before training:")
	net.Layers[0].PrintWeights()
	net.Layers[len(net.Layers)-1].PrintWeights()
	net.FeedForward(&image)
	var desired = nns.NewTensor(3, 1, 1)
	desired.SetData([][][]float64{
		[][]float64{
			[]float64{0.32, 0.45, 0.96},
		},
	})
	net.Backpropagate(&desired)
	fmt.Println("Weights after training:")
	net.Layers[0].PrintWeights()
	net.Layers[len(net.Layers)-1].PrintWeights()
}
