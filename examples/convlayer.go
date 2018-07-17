package examples

import (
	"cnns_vika/nns"
	"fmt"
)

// CheckConvLayer - проверка свёрточного слоя
func CheckConvLayer() {
	clayer := nns.NewConvLayer(3, 3, 1, 1, 0, 0, 1, 8, 9, 1)
	conv := &nns.LayerStruct{
		Layer: clayer,
	}
	rlayer := nns.NewReLULayer(clayer.Out.X, clayer.Out.Y, clayer.Out.Z)
	relu := &nns.LayerStruct{
		Layer: rlayer,
	}
	mlayer := nns.NewMaxPoolingLayer(rlayer.Out.X, rlayer.Out.Y, rlayer.Out.Z, 2, 2)
	maxpool := &nns.LayerStruct{
		Layer: mlayer,
	}
	flayer := nns.NewFullConnectedLayer(mlayer.Out.X, mlayer.Out.Y, mlayer.Out.Z, 3, false, true)
	fullyconnected := &nns.LayerStruct{
		Layer: flayer,
	}

	clayer.PrintWeights()
	flayer.PrintWeights()

	var net nns.WholeNet
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

	var image = nns.NewTensorEmpty(8, 9, 1) // w,h,d
	image.Set(&matrix)

	// FeedForward
	net.Layers[0].FeedForward(image)
	net.Layers[0].PrintOutput()

	net.Layers[1].FeedForward(net.Layers[0].GetOutput())
	net.Layers[1].PrintOutput()

	net.Layers[2].FeedForward(net.Layers[1].GetOutput())
	net.Layers[2].PrintOutput()

	net.Layers[3].FeedForward(net.Layers[2].GetOutput())
	net.Layers[3].PrintOutput()

	// Backpropagate
	var desired = nns.NewTensorEmpty(3, 1, 1) // w,h,d
	matrix = [][][]float64{
		[][]float64{
			[]float64{0.32, 0.45, 0.96},
		},
	}
	desired.Set(&matrix)
	difference := net.Layers[3].GetOutput().Sub(desired)
	fmt.Println("Output - Desired:")
	difference.Print()

	net.Layers[3].CalculateGradients(difference)
	// net.Layers[3].PrintGradients()

	// net.Layers[3].PrintWeights()
	net.Layers[3].UpdateWeights()
	// net.Layers[3].PrintWeights()
}
