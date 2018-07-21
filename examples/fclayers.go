package examples

import (
	"cnns_vika/nns"
	"fmt"
	"log"
)

// CheckFClayer - проверка полносвязного слоя
func CheckFClayer() {
	flayer1 := nns.NewFullConnectedLayer(2, 1, 1, 3, true, false)
	fullyconnected1 := &nns.LayerStruct{
		Layer: flayer1,
	}
	flayer2 := nns.NewFullConnectedLayer(flayer1.Out.X, flayer1.Out.Y, flayer1.Out.Z, 3, true, false)
	fullyconnected2 := &nns.LayerStruct{
		Layer: flayer2,
	}
	flayer3 := nns.NewFullConnectedLayer(flayer2.Out.X, flayer2.Out.Y, flayer2.Out.Z, 1, true, false)
	fullyconnected3 := &nns.LayerStruct{
		Layer: flayer3,
	}

	var net nns.WholeNet
	net.Layers = append(net.Layers, fullyconnected1)
	net.Layers = append(net.Layers, fullyconnected2)
	net.Layers = append(net.Layers, fullyconnected3)

	var matrix = [][][]float64{
		[][]float64{
			[]float64{1, 0},
		},
	}

	var image = nns.NewTensorEmpty(2, 1, 1) // w,h,d
	image.Set(&matrix)

	// net.Layers[0].PrintWeights()
	// net.Layers[1].PrintWeights()
	// net.Layers[2].PrintWeights()

	// FeedForward
	net.Layers[0].FeedForward(image)
	net.Layers[0].PrintOutput()

	net.Layers[1].FeedForward(net.Layers[0].GetOutput())
	net.Layers[1].PrintOutput()

	net.Layers[2].FeedForward(net.Layers[1].GetOutput())
	net.Layers[2].PrintOutput()

	// Backpropagate
	var desired = nns.NewTensorEmpty(1, 1, 1) // w,h,d
	matrix = [][][]float64{
		[][]float64{
			[]float64{0},
		},
	}
	desired.Set(&matrix)
	difference := net.Layers[2].GetOutput().Sub(desired)
	// fmt.Println("Output - Desired:")
	difference.Print()

	net.Layers[2].CalculateGradients(difference)
	// net.Layers[2].PrintGradients()

	net.Layers[1].CalculateGradients(net.Layers[2].GetGradients())
	// net.Layers[1].PrintGradients()

	net.Layers[0].CalculateGradients(net.Layers[1].GetGradients())
	// net.Layers[0].PrintGradients()

	net.Layers[0].UpdateWeights()
	net.Layers[1].UpdateWeights()
	net.Layers[2].UpdateWeights()

	// net.Layers[0].PrintWeights()
	// net.Layers[1].PrintWeights()
	// net.Layers[2].PrintWeights()

	log.Println("<<<<<<<<<<====================>>>>>>>>>>")
	log.Println("Second step")
	log.Println("<<<<<<<<<<====================>>>>>>>>>>")

	image = nns.NewTensorEmpty(2, 1, 1) // w,h,d
	matrix = [][][]float64{
		[][]float64{
			[]float64{1, 0},
		},
	}
	image.Set(&matrix)
	// FeedForward
	net.Layers[0].FeedForward(image)
	//	net.Layers[0].PrintOutput()

	net.Layers[1].FeedForward(net.Layers[0].GetOutput())
	// net.Layers[1].PrintOutput()

	net.Layers[2].FeedForward(net.Layers[1].GetOutput())
	// net.Layers[2].PrintOutput()

	// Backpropagate
	desired = nns.NewTensorEmpty(1, 1, 1) // w,h,d
	matrix = [][][]float64{
		[][]float64{
			[]float64{0},
		},
	}
	desired.Set(&matrix)
	difference = net.Layers[2].GetOutput().Sub(desired)
	fmt.Println("Output - Desired:")
	// difference.Print()

	net.Layers[2].CalculateGradients(difference)
	// net.Layers[2].PrintGradients()

	net.Layers[1].CalculateGradients(net.Layers[2].GetGradients())
	// net.Layers[1].PrintGradients()

	net.Layers[0].CalculateGradients(net.Layers[1].GetGradients())
	// net.Layers[0].PrintGradients()

	net.Layers[2].UpdateWeights()
	net.Layers[2].PrintWeights()
	net.Layers[1].UpdateWeights()
	net.Layers[1].PrintWeights()
	net.Layers[0].UpdateWeights()
	net.Layers[0].PrintWeights()

}
