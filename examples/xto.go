package examples

import (
	"cnns_vika/nns"
	"cnns_vika/utils/u"
	"fmt"
	"log"
	"math/rand"
	"time"
)

// CheckXTO - проверка свёрточного слоя для распознавания символов "X", "T" и "O"
func CheckXTO() {
	rand.Seed(time.Now().UnixNano())
	conv := nns.NewConvLayer(1, 3, 2, nns.TDsize{X: 8, Y: 9, Z: 1}) //
	relu := nns.NewReLULayer(conv.OutSize())
	maxpool := nns.NewMaxPoolingLayer(2, 2, relu.OutSize())
	fullyconnected := nns.NewFullConnectedLayer(maxpool.OutSize(), 3)
	// flayer.SetActivationFunc(ActivationTanh)
	// flayer.SetActivationDerivativeFunc(ActivationTanhDerivative)

	var net nns.WholeNet
	net.Layers = append(net.Layers, conv)
	net.Layers = append(net.Layers, relu)
	net.Layers = append(net.Layers, maxpool)
	net.Layers = append(net.Layers, fullyconnected)

	var xmatrix = [][][]float64{
		[][]float64{
			[]float64{0, 0, 0, 0, 0, 0, 0, 0},
			[]float64{0, 1, 0, 0, 0, 1, 0, 0},
			[]float64{0, 0, 1, 0, 1, 0, 0, 0},
			[]float64{0, 0, 1, 0, 1, 0, 0, 0},
			[]float64{0, 0, 0, 1, 0, 0, 0, 0},
			[]float64{0, 0, 1, 0, 1, 0, 0, 0},
			[]float64{0, 0, 1, 0, 1, 0, 0, 0},
			[]float64{0, 1, 0, 0, 0, 1, 0, 0},
			[]float64{0, 1, 0, 0, 0, 1, 0, 0},
		},
	}

	var tmatrix = [][][]float64{
		[][]float64{
			[]float64{0, 1, 1, 1, 1, 1, 1, 0},
			[]float64{0, 1, 1, 1, 1, 1, 1, 0},
			[]float64{0, 0, 0, 1, 1, 0, 0, 0},
			[]float64{0, 0, 0, 1, 1, 0, 0, 0},
			[]float64{0, 0, 0, 1, 1, 0, 0, 0},
			[]float64{0, 0, 0, 1, 1, 0, 0, 0},
			[]float64{0, 0, 0, 1, 1, 0, 0, 0},
			[]float64{0, 0, 0, 1, 1, 0, 0, 0},
			[]float64{0, 0, 0, 1, 1, 0, 0, 0},
		},
	}

	var omatrix = [][][]float64{
		[][]float64{
			[]float64{0, 0, 0, 0, 0, 0, 0, 0},
			[]float64{0, 1, 1, 1, 1, 1, 1, 0},
			[]float64{0, 1, 0, 0, 0, 0, 1, 0},
			[]float64{0, 1, 0, 0, 0, 0, 1, 0},
			[]float64{0, 1, 0, 0, 0, 0, 1, 0},
			[]float64{0, 1, 0, 0, 0, 0, 1, 0},
			[]float64{0, 1, 0, 0, 0, 0, 1, 0},
			[]float64{0, 1, 0, 0, 0, 0, 1, 0},
			[]float64{0, 0, 1, 1, 1, 1, 0, 0},
		},
	}

	var ximage = nns.NewTensor(8, 9, 1)
	ximage.SetData(xmatrix)
	var timage = nns.NewTensor(8, 9, 1)
	timage.SetData(tmatrix)
	var oimage = nns.NewTensor(8, 9, 1)
	oimage.SetData(omatrix)

	log.Println("Weights before:")
	net.Layers[0].PrintWeights()
	net.Layers[3].PrintWeights()

	// Train
	for i := 0; i < 10000; i++ {
		var rnd = u.RandomInt(0, 3)
		var desired = nns.NewTensor(3, 1, 1)
		desiredMat := [][][]float64{[][]float64{[]float64{0.0, 0.0, 0.0}}}
		desiredMat[0][0][rnd] = 1.0
		desired.SetData(desiredMat)
		var train = nns.NewTensor(8, 9, 1)
		switch rnd {
		case 0:
			train = ximage
			break
		case 1:
			train = timage
			break
		case 2:
			train = oimage
			break
		default:
			break
		}
		// Forward
		net.FeedForward(&train)
		// Backward
		net.Backpropagate(&desired)
	}
	log.Println("Weights after:")
	net.Layers[0].PrintWeights()
	net.Layers[3].PrintWeights()

	// Test
	xmatrix = [][][]float64{
		[][]float64{
			[]float64{0, 0, 0, 0, 0, 0, 0, 0},
			[]float64{0, 1, 0, 0, 0, 1, 0, 0},
			[]float64{0, 0, 1, 0, 1, 0, 0, 0},
			[]float64{0, 0, 1, 0, 1, 0, 0, 0},
			[]float64{0, 0, 0, 0.8, 0, 0, 0, 0},
			[]float64{0, 0, 1, 0, 1, 0, 0, 0},
			[]float64{0, 0, 1, 0, 0.5, 0, 0, 0},
			[]float64{0, 0.4, 0, 0, 0, 1, 0, 0},
			[]float64{0, 1, 0, 0, 0, 1, 0, 0},
		},
	}
	tmatrix = [][][]float64{
		[][]float64{
			[]float64{0, 1, 1, 1, 1, 0.7, 0.5, 0},
			[]float64{0, 1, 1, 1, 1, 1, 1, 0},
			[]float64{0, 0, 0, 0, 1, 0, 0, 0},
			[]float64{0, 0, 0, 0, 1, 0, 0, 0},
			[]float64{0, 0, 0, 1, 1, 0, 0, 0},
			[]float64{0, 0, 0, 1, 1, 0, 0, 0},
			[]float64{0, 0, 0, 1, 1, 0, 0, 0},
			[]float64{0, 0, 0, 1, 0, 0, 0, 0},
			[]float64{0, 0, 0, 1, 1, 0, 0, 0},
		},
	}
	omatrix = [][][]float64{
		[][]float64{
			[]float64{0, 0, 0, 0, 0.6, 0, 0, 0},
			[]float64{0, 1, 1, 0.5, 1, 1, 1, 0},
			[]float64{0, 1, 0, 0.5, 0, 0, 1, 0},
			[]float64{0, 1, 0, 0, 0, 0, 1, 0},
			[]float64{0, 1, 0, 0, 0, 0, 1, 0},
			[]float64{0, 1, 0, 0, 0, 0, 1, 0},
			[]float64{0, 0.9, 0, 0, 0, 0, 1, 0},
			[]float64{0, 1, 0, 0, 0, 0, 1, 0},
			[]float64{0, 0, 1, 0.8, 1, 1, 0, 0},
		},
	}
	ximage.SetData(xmatrix)
	timage.SetData(tmatrix)
	oimage.SetData(omatrix)

	fmt.Println("For X should be: [1, 0, 0], Got:")
	net.FeedForward(&ximage)
	net.PrintOutput()

	fmt.Println("For T should be: [0, 1, 0], Got:")
	net.FeedForward(&timage)
	net.PrintOutput()

	fmt.Println("For O should be: [0, 0, 1], Got:")
	net.FeedForward(&oimage)
	net.PrintOutput()
}
