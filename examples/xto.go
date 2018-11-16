package examples

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/LdDl/cnns/nns"

	"github.com/LdDl/cnns/utils/u"
)

// CheckXTO - recognition of "X", "T" и "O" symbols represented as matrices
func CheckXTO() {
	rand.Seed(time.Now().UnixNano())
	conv := nns.NewConvLayer(1, 3, 2, nns.TDsize{X: 8, Y: 9, Z: 1})
	relu := nns.NewLeakyReLULayer(conv.OutSize(), 0.01)
	maxpool := nns.NewMaxPoolingLayer(2, 2, relu.OutSize())
	fullyconnected := nns.NewFullConnectedLayer(maxpool.OutSize(), 3)

	// You can play with activation function for fully connected layer
	fullyconnected.SetActivationFunc(nns.ActivationSygmoid)
	fullyconnected.SetActivationDerivativeFunc(nns.ActivationSygmoidDerivative)

	var net nns.WholeNet
	net.Layers = append(net.Layers, conv)
	net.Layers = append(net.Layers, relu)
	net.Layers = append(net.Layers, maxpool)
	net.Layers = append(net.Layers, fullyconnected)

	// Init train and test data
	inputs, desired := formTrainDataXTO()
	inputsTests, desiredTests := formTestDataXTO()

	// Start traing process
	numOfEpochs := 50
	trainErr, testErr, err := net.Train(&inputs, &desired, &inputsTests, &desiredTests, numOfEpochs)
	if err != nil {
		log.Fatalln(err)
	}

	fmt.Printf("Error on training data: %v\nError on test data: %v\n", trainErr, testErr)

}

func formTrainDataXTO() ([]nns.Tensor, []nns.Tensor) {
	numExamples := 10000

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
	ximage.SetData3D(xmatrix)
	var timage = nns.NewTensor(8, 9, 1)
	timage.SetData3D(tmatrix)
	var oimage = nns.NewTensor(8, 9, 1)
	oimage.SetData3D(omatrix)

	inputs := make([]nns.Tensor, numExamples)
	desired := make([]nns.Tensor, numExamples)
	for i := 0; i < numExamples; i++ {
		var rnd = u.RandomInt(0, 3)

		input := nns.NewTensor(8, 9, 1)
		switch rnd {
		case 0:
			input = ximage
			break
		case 1:
			input = timage
			break
		case 2:
			input = oimage
			break
		default:
			break
		}

		desiredMat := [][][]float64{[][]float64{[]float64{0.0, 0.0, 0.0}}}
		desiredMat[0][0][rnd] = 1.0

		target := nns.NewTensor(3, 1, 1)
		target.SetData3D(desiredMat)

		inputs[i] = input
		desired[i] = target
	}
	return inputs, desired
}

func formTestDataXTO() ([]nns.Tensor, []nns.Tensor) {
	inputs := make([]nns.Tensor, 0, 3)
	desired := make([]nns.Tensor, 0, 3)

	input := nns.NewTensor(8, 9, 1)
	target := nns.NewTensor(3, 1, 1)

	// X = [1, 0, 0]
	input.SetData(8, 9, 1, []float64{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 1, 0, 0,
		0, 0, 1, 0, 1, 0, 0, 0,
		0, 0, 1, 0, 1, 0, 0, 0,
		0, 0, 0, 0.8, 0, 0, 0, 0,
		0, 0, 1, 0, 1, 0, 0, 0,
		0, 0, 1, 0, 0.5, 0, 0, 0,
		0, 0.4, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0, 0,
	})
	target.SetData(3, 1, 1, []float64{1, 0, 0})
	inputs = append(inputs, input)
	desired = append(desired, target)

	// T = [0, 1, 0]
	input = nns.NewTensor(8, 9, 1)
	input.SetData(8, 9, 1, []float64{
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
	})
	target = nns.NewTensor(3, 1, 1)
	target.SetData(3, 1, 1, []float64{0, 1, 0})
	inputs = append(inputs, input)
	desired = append(desired, target)

	// O = [0, 0, 1]
	input = nns.NewTensor(8, 9, 1)
	input.SetData(8, 9, 1, []float64{
		0, 0, 0, 0, 0.6, 0, 0, 0,
		0, 1, 1, 0.5, 1, 1, 1, 0,
		0, 1, 0, 0.5, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 0.9, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 0, 1, 0.8, 1, 1, 0, 0,
	})
	target = nns.NewTensor(3, 1, 1)
	target.SetData(3, 1, 1, []float64{0, 0, 1})
	inputs = append(inputs, input)
	desired = append(desired, target)

	return inputs, desired
}
