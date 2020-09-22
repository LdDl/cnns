package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/LdDl/cnns"
	"github.com/LdDl/cnns/tensor"
	"github.com/LdDl/cnns/utils/u"
	"gonum.org/v1/gonum/mat"
)

var (
	imgWidth    = 8
	imgHeight   = 9
	numExamples = 10000
	numOfEpochs = 50
)

func main() {
	rand.Seed(time.Now().UnixNano())
	conv := cnns.NewConvLayer(tensor.TDsize{X: 8, Y: 9, Z: 1}, 1, 3, 1)
	relu := cnns.NewReLULayer(conv.GetOutputSize())
	maxpool := cnns.NewPoolingLayer(relu.GetOutputSize(), 2, 2, "max", "valid")
	fullyconnected := cnns.NewFullyConnectedLayer(maxpool.GetOutputSize(), 3)

	// You can play with activation function for fully connected layer
	fullyconnected.SetActivationFunc(cnns.ActivationSygmoid)
	fullyconnected.SetActivationDerivativeFunc(cnns.ActivationSygmoidDerivative)

	var net cnns.WholeNet
	net.Layers = append(net.Layers, conv)
	net.Layers = append(net.Layers, relu)
	net.Layers = append(net.Layers, maxpool)
	net.Layers = append(net.Layers, fullyconnected)

	// Init train and test data
	inputs, desired := formTrainDataXTO()
	inputsTests, desiredTests := formTestDataXTO()

	// Start traing process
	trainErr, testErr, err := net.Train(inputs, desired, inputsTests, desiredTests, numOfEpochs)
	if err != nil {
		log.Printf("Can't train network due the error: %s", err.Error())
		return
	}

	for i := range inputsTests {
		in := inputsTests[i]
		target := desiredTests[i]
		err := net.FeedForward(in)
		if err != nil {
			log.Printf("Feedforward (testing) caused error: %s", err.Error())
			return
		}
		out := net.GetOutput()
		fmt.Println("\n>>>Out:")
		fmt.Println("\t", out.RawMatrix().Data)
		fmt.Println(">>>Desired:")
		fmt.Println("\t", target.RawMatrix().Data)
	}

	fmt.Printf("Error on training data: %v\nError on test data: %v\n", trainErr, testErr)
}

func formTrainDataXTO() ([]*mat.Dense, []*mat.Dense) {

	ximage := mat.NewDense(imgHeight, imgWidth, []float64{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 1, 0, 0,
		0, 0, 1, 0, 1, 0, 0, 0,
		0, 0, 1, 0, 1, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 1, 0, 1, 0, 0, 0,
		0, 0, 1, 0, 1, 0, 0, 0,
		0, 1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0, 0,
	})
	timage := mat.NewDense(imgHeight, imgWidth, []float64{
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
	})
	oimage := mat.NewDense(imgHeight, imgWidth, []float64{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 0, 0, 0, 1, 0,
		0, 0, 1, 1, 1, 1, 0, 0,
	})

	inputs := make([]*mat.Dense, numExamples)
	desired := make([]*mat.Dense, numExamples)

	for i := 0; i < numExamples; i++ {
		var rnd = u.RandomInt(0, 3)

		input := mat.NewDense(imgHeight, imgWidth, nil)
		switch rnd {
		case 0:
			input.CloneFrom(ximage)
			break
		case 1:
			input.CloneFrom(timage)
			break
		case 2:
			input.CloneFrom(oimage)
			break
		default:
			break
		}

		desiredMat := []float64{0.0, 0.0, 0.0}
		desiredMat[rnd] = 1.0
		target := mat.NewDense(3, 1, desiredMat)

		inputs[i] = input
		desired[i] = target
	}

	return inputs, desired
}

func formTestDataXTO() ([]*mat.Dense, []*mat.Dense) {

	inputs := make([]*mat.Dense, 3)
	desired := make([]*mat.Dense, 3)

	// X = [1, 0, 0]
	ximage := mat.NewDense(imgHeight, imgWidth, []float64{
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
	inputs[0] = ximage
	desired[0] = mat.NewDense(3, 1, []float64{1, 0, 0})

	// T = [0, 1, 0]
	timage := mat.NewDense(imgHeight, imgWidth, []float64{
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
	inputs[1] = timage
	desired[1] = mat.NewDense(3, 1, []float64{0, 1, 0})

	// O = [0, 0, 1]
	oimage := mat.NewDense(imgHeight, imgWidth, []float64{
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
	inputs[2] = oimage
	desired[2] = mat.NewDense(3, 1, []float64{0, 0, 1})

	return inputs, desired
}
