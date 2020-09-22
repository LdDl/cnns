package main

import (
	"fmt"
	"log"

	"github.com/LdDl/cnns"
	"github.com/LdDl/cnns/tensor"
	"gonum.org/v1/gonum/mat"
)

func main() {
	ExampleConv()
	// ExampleConv2()
}

// ExampleConv Check how convolutional network's layers works with single channel image. Corresponding file is "step_by_step_cnn(dense inertia).xlsx" file.
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
	Using custom weights (for testing purposes) also.
*/
func ExampleConv() {
	conv := cnns.NewConvLayer(tensor.TDsize{X: 8, Y: 9, Z: 1}, 1, 3, 1)
	relu := cnns.NewReLULayer(conv.GetOutputSize())
	maxpool := cnns.NewPoolingLayer(relu.GetOutputSize(), 2, 2, "max", "valid")
	fullyconnected := cnns.NewFullyConnectedLayer(maxpool.GetOutputSize(), 3)

	convCustomWeights := mat.NewDense(3, 3, []float64{
		0.10466029, -0.06228581, -0.43436298,
		0.44050909, -0.07536250, -0.34348075,
		0.16456005, 0.18682307, -0.40303048,
	})
	conv.SetCustomWeights([]*mat.Dense{convCustomWeights})

	fcCustomWeights := mat.NewDense(3, maxpool.GetOutputSize().Total(), []float64{
		-0.19908814, 0.01521263, 0.31363996, -0.28573613, -0.11934281, -0.18194183, -0.03111016, -0.21696585, -0.20689814,
		0.17908468, -0.28144695, -0.29681312, -0.13912858, 0.07067328, 0.36249144, -0.20688576, -0.20291744, 0.25257304,
		-0.29341734, 0.36533501, 0.19671917, 0.02382031, -0.47169692, -0.34167172, 0.10725344, 0.47524162, -0.42054638,
	})
	fullyconnected.SetCustomWeights([]*mat.Dense{fcCustomWeights})

	var net cnns.WholeNet
	net.Layers = append(net.Layers, conv)
	net.Layers = append(net.Layers, relu)
	net.Layers = append(net.Layers, maxpool)
	net.Layers = append(net.Layers, fullyconnected)

	image := mat.NewDense(9, 8, []float64{
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

	fmt.Printf("Layers weights:\n")
	for i := range net.Layers {
		fmt.Printf("%s #%d weights:\n", net.Layers[i].GetType(), i)
		net.Layers[i].PrintWeights()
	}
	fmt.Println("\tDoing training....")
	for e := 0; e < 3; e++ {
		err := net.FeedForward(image)
		if err != nil {
			log.Printf("Feedforward caused error: %s", err.Error())
			return
		}
		desired := mat.NewDense(3, 1, []float64{0.32, 0.45, 0.96})
		err = net.Backpropagate(desired)
		if err != nil {
			log.Printf("Backpropagate caused error: %s", err.Error())
			return
		}

		fmt.Printf("Epoch #%d. New layers weights\n", e)
		for i := range net.Layers {
			fmt.Printf("%s #%d weights on epoch #%d:\n", net.Layers[i].GetType(), i, e)
			net.Layers[i].PrintWeights()
		}
	}
}

// ExampleConv2 Check how convolutional network's layers works with RGB-based image. Corresponding file is "step_by_step_cnn(rgb dense inertia).xlsx" file.
/*
	Using here 4 layers: convolutional, ReLU, pooling (max) and fc
		Convolutinal features:
			Single kernel: 3x3
			Input size: 5x5x3 (width, height, depth)
		ReLU features:
			Input size: Convolutinal.OutputSize
		Pooling (max) features:
			Window size: 3x3
			Input size: ReLU.OutputSize
		FC:
			Input size: Pooling.OutputSize
			Outputsize: 3 (actually 2x1x1)
	Using custom weights (for testing purposes) also.
*/
func ExampleConv2() {
	conv := cnns.NewConvLayer(tensor.TDsize{X: 5, Y: 5, Z: 3}, 1, 3, 2)
	relu := cnns.NewReLULayer(conv.GetOutputSize())
	maxpool := cnns.NewPoolingLayer(relu.GetOutputSize(), 2, 2, "max", "valid")
	fullyconnected := cnns.NewFullyConnectedLayer(maxpool.GetOutputSize(), 2)

	var net cnns.WholeNet
	net.Layers = append(net.Layers, conv)
	net.Layers = append(net.Layers, relu)
	net.Layers = append(net.Layers, maxpool)
	net.Layers = append(net.Layers, fullyconnected)

	redChannel := mat.NewDense(5, 5, []float64{
		1, 0, 1, 0, 2,
		1, 1, 3, 2, 1,
		1, 1, 0, 1, 1,
		2, 3, 2, 1, 3,
		0, 2, 0, 1, 0,
	})
	greenChannel := mat.NewDense(5, 5, []float64{
		1, 0, 0, 1, 0,
		2, 0, 1, 2, 0,
		3, 1, 1, 3, 0,
		0, 3, 0, 3, 2,
		1, 0, 3, 2, 1,
	})
	blueChannel := mat.NewDense(5, 5, []float64{
		2, 0, 1, 2, 1,
		3, 3, 1, 3, 2,
		2, 1, 1, 1, 0,
		3, 1, 3, 2, 0,
		1, 1, 2, 1, 1,
	})

	kernel1R := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 2,
		0, 1, 0,
	})
	kernel1G := mat.NewDense(3, 3, []float64{
		2, 1, 0,
		0, 0, 0,
		0, 3, 0,
	})
	kernel1B := mat.NewDense(3, 3, []float64{
		1, 0, 0,
		1, 0, 0,
		0, 0, 2,
	})

	kernel2R := mat.NewDense(3, 3, []float64{
		0, -1, 0,
		0, 0, 2,
		0, 1, 0,
	})
	kernel2G := mat.NewDense(3, 3, []float64{
		2, 1, 0,
		0, 0, 0,
		0, -3, 0,
	})
	kernel2B := mat.NewDense(3, 3, []float64{
		1, 0, 0,
		1, 0, 0,
		0, 0, -2,
	})

	img2 := &mat.Dense{}
	img2.Stack(redChannel, greenChannel)
	image := &mat.Dense{}
	image.Stack(img2, blueChannel)

	kernel1 := &mat.Dense{}
	kernel1.Stack(kernel1R, kernel1G)
	convCustomWeights1 := &mat.Dense{}
	convCustomWeights1.Stack(kernel1, kernel1B)

	kernel2 := &mat.Dense{}
	kernel2.Stack(kernel2R, kernel2G)
	convCustomWeights2 := &mat.Dense{}
	convCustomWeights2.Stack(kernel2, kernel2B)
	conv.SetCustomWeights([]*mat.Dense{convCustomWeights1, convCustomWeights2})

	fcCustomWeights := mat.NewDense(2, maxpool.GetOutputSize().Total(), []float64{
		-0.19908814, 0.01521263,
		0.17908468, -0.28144695,
	})
	fullyconnected.SetCustomWeights([]*mat.Dense{fcCustomWeights})

	fmt.Printf("Layers weights:\n")
	for i := range net.Layers {
		fmt.Printf("%s #%d weights:\n", net.Layers[i].GetType(), i)
		net.Layers[i].PrintWeights()
	}
	fmt.Println("\tDoing training....")
	for e := 0; e < 1; e++ {
		err := net.FeedForward(image)
		if err != nil {
			log.Printf("Feedforward caused error: %s", err.Error())
			return
		}

		desired := mat.NewDense(2, 1, []float64{0.15, 0.8})
		err = net.Backpropagate(desired)
		if err != nil {
			log.Printf("Backpropagate caused error: %s", err.Error())
			return
		}

		fmt.Printf("Epoch #%d. New layers weights\n", e)
		for i := range net.Layers {
			fmt.Printf("%s #%d weights on epoch #%d:\n", net.Layers[i].GetType(), i, e)
			net.Layers[i].PrintWeights()
		}
	}
}
