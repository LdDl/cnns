package main

import (
	"fmt"

	"github.com/LdDl/cnns"
	t "github.com/LdDl/cnns/tensor"
)

// ExampleConv - Check feedworward and backpropagate operations
/*
	Using here 4 layers: convolutional, ReLU, pooling (max) and fc
		Convolutinal features:
			Single kernel: 3x3
			Input size: 8x9x1 (width, height, depth)
		ReLU features:
			Input size: Convolutinal.OutputSize
		Pooling (max) features:
			Window size: 3x3
			Input size: ReLU.OutputSize
		FC:
			Input size: Pooling.OutputSize
			Outputsize: 3 (actually 3x1x1)
	Using custom weights (for testing purposes) also. You can check "step_by_step_cnss.xlsx" file.
*/
func ExampleConv() {
	conv := cnns.NewConvLayer(1, 3, 1, t.TDsize{X: 8, Y: 9, Z: 1})
	relu := cnns.NewReLULayer(conv.OutSize())
	maxpool := cnns.NewMaxPoolingLayer(2, 2, relu.OutSize())
	fullyconnected := cnns.NewFullConnectedLayer(maxpool.OutSize(), 3)

	convCustomWeights := t.NewTensor(3, 3, 1)
	convCustomWeights.SetData(3, 3, 1, []float64{
		0.10466029, -0.06228581, -0.43436298,
		0.44050909, -0.07536250, -0.34348075,
		0.16456005, 0.18682307, -0.40303048,
	})
	conv.SetCustomWeights(&[]t.Tensor{convCustomWeights})

	fcCustomWeights := t.NewTensor(maxpool.OutSize().X*maxpool.OutSize().Y*maxpool.OutSize().Z, 3, 1)
	fcCustomWeights.SetData(maxpool.OutSize().X*maxpool.OutSize().Y*maxpool.OutSize().Z, 3, 1, []float64{
		-0.19908814, 0.01521263, 0.31363996, -0.28573613, -0.11934281, -0.18194183, -0.03111016, -0.21696585, -0.20689814,
		0.17908468, -0.28144695, -0.29681312, -0.13912858, 0.07067328, 0.36249144, -0.20688576, -0.20291744, 0.25257304,
		-0.29341734, 0.36533501, 0.19671917, 0.02382031, -0.47169692, -0.34167172, 0.10725344, 0.47524162, -0.42054638,
	})
	fullyconnected.SetCustomWeights(&[]t.Tensor{fcCustomWeights})

	var net cnns.WholeNet
	net.Layers = append(net.Layers, conv)
	net.Layers = append(net.Layers, relu)
	net.Layers = append(net.Layers, maxpool)
	net.Layers = append(net.Layers, fullyconnected)
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
	var image = t.NewTensor(8, 9, 1)
	image.SetData3D(matrix)
	// fmt.Println("Image:")
	// image.Print()

	fmt.Println("\nWeights before training:")
	net.Layers[0].PrintWeights()
	net.Layers[len(net.Layers)-1].PrintWeights()

	var desired = t.NewTensor(3, 1, 1)
	for e := 0; e < 3; e++ {
		net.FeedForward(&image)
		desired.SetData3D([][][]float64{
			[][]float64{
				[]float64{0.32, 0.45, 0.96},
			},
		})
		net.Backpropagate(&desired)
	}

	net.FeedForward(&image)
	net.PrintOutput()

	fmt.Println("Weights after training:")
	net.Layers[0].PrintWeights()
	net.Layers[len(net.Layers)-1].PrintWeights()
}

func main() {
	ExampleConv()
}
