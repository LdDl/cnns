package examples

import (
	"cnns_vika/nns"
	"cnns_vika/utils/u"
	"fmt"
	"math/rand"
	"time"
)

// CheckXTO - проверка свёрточного слоя для распознавания символов "X", "T" и "O"
func CheckXTO() {
	rand.Seed(time.Now().UnixNano())
	clayer := nns.NewConvLayer(1, 3, 1, nns.TDsize{X: 8, Y: 9, Z: 1})
	conv := &nns.LayerStruct{
		Layer: clayer,
	}
	rlayer := nns.NewReLULayer(clayer.Out.Size)
	relu := &nns.LayerStruct{
		Layer: rlayer,
	}
	mlayer := nns.NewMaxPoolingLayer(2, 2, rlayer.Out.Size)
	maxpool := &nns.LayerStruct{
		Layer: mlayer,
	}
	flayer := nns.NewFullConnectedLayer(mlayer.Out.Size, 3)
	// flayer.SetActivationFunc(ActivationTanh)
	// flayer.SetActivationDerivativeFunc(ActivationTanhDerivative)
	fullyconnected := &nns.LayerStruct{
		Layer: flayer,
	}

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
	ximage.CopyFrom(xmatrix)
	var timage = nns.NewTensor(8, 9, 1)
	timage.CopyFrom(tmatrix)
	var oimage = nns.NewTensor(8, 9, 1)
	oimage.CopyFrom(omatrix)

	// log.Println("Weights before:")
	// net.Layers[0].PrintWeights()
	// net.Layers[3].PrintWeights()

	// Train
	for i := 0; i < 10000; i++ {
		var rnd = u.RandomInt(0, 3)
		var desired = nns.NewTensor(3, 1, 1)
		desiredMat := [][][]float64{[][]float64{[]float64{0.0, 0.0, 0.0}}}
		desiredMat[0][0][rnd] = 1.0
		desired.CopyFrom(desiredMat)
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
		// FeedForward
		net.Layers[0].FeedForward(&train)
		for l := 1; l < len(net.Layers); l++ {
			out := net.Layers[l-1].GetOutput()
			net.Layers[l].FeedForward(&out)
		}
		// Backpropagate
		difference := net.Layers[len(net.Layers)-1].GetOutput()
		difference.Sub(&desired)
		net.Layers[len(net.Layers)-1].CalculateGradients(&difference)
		for i := len(net.Layers) - 2; i >= 0; i-- {
			grad := net.Layers[i+1].GetGradients()
			net.Layers[i].CalculateGradients(&grad)
		}
		for i := range net.Layers {
			net.Layers[i].UpdateWeights()
		}
	}
	// log.Println("Weights after:")
	// net.Layers[0].PrintWeights()
	// net.Layers[3].PrintWeights()

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
	ximage.CopyFrom(xmatrix)
	timage.CopyFrom(tmatrix)
	oimage.CopyFrom(omatrix)

	fmt.Println("For X should be: [1, 0, 0], Got:")
	net.Layers[0].FeedForward(&ximage)
	for l := 1; l < len(net.Layers); l++ {
		out := net.Layers[l-1].GetOutput()
		net.Layers[l].FeedForward(&out)
	}
	net.Layers[len(net.Layers)-1].PrintOutput()

	fmt.Println("For T should be: [0, 1, 0], Got:")
	net.Layers[0].FeedForward(&timage)
	for l := 1; l < len(net.Layers); l++ {
		out := net.Layers[l-1].GetOutput()
		net.Layers[l].FeedForward(&out)
	}
	net.Layers[len(net.Layers)-1].PrintOutput()

	fmt.Println("For O should be: [0, 0, 1], Got:")
	net.Layers[0].FeedForward(&oimage)
	for l := 1; l < len(net.Layers); l++ {
		out := net.Layers[l-1].GetOutput()
		net.Layers[l].FeedForward(&out)
	}
	net.Layers[len(net.Layers)-1].PrintOutput()
}
