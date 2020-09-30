package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/LdDl/cnns"
	"github.com/LdDl/cnns/tensor"
	"gonum.org/v1/gonum/mat"

	"github.com/LdDl/cnns/utils/u"
)

func main() {
	CheckOR()
}

// CheckOR - solve "OR" problem
func CheckOR() {
	rand.Seed(time.Now().UnixNano())
	// fully-connected layer with 3 output neurons
	fullyconnected1 := cnns.NewFullyConnectedLayer(&tensor.TDsize{X: 2, Y: 1, Z: 1}, 2)
	// There is 2 lines of reduntan code below, but it shows how to set definied activation function
	fullyconnected1.SetActivationFunc(cnns.ActivationTanh)
	fullyconnected1.SetActivationDerivativeFunc(cnns.ActivationTanhDerivative)

	// fully-connected layer with 1 output neurons
	fullyconnected2 := cnns.NewFullyConnectedLayer(fullyconnected1.GetOutputSize(), 1)
	// There is 2 lines of reduntan code below, but it shows how to set definied activation function
	fullyconnected2.SetActivationFunc(cnns.ActivationTanh)
	fullyconnected2.SetActivationDerivativeFunc(cnns.ActivationTanhDerivative)

	// Init network
	net := cnns.WholeNet{
		LP: cnns.NewLearningParametersDefault(),
	}
	net.Layers = append(net.Layers, fullyconnected1)
	net.Layers = append(net.Layers, fullyconnected2)

	// Init train and test data
	inputs, desired := formTrainDataOR()
	inputsTests, desiredTests := formTestDataOR()

	// Start traing process
	numOfEpochs := 1
	trainErr, testErr, err := net.Train(inputs, desired, inputsTests, desiredTests, numOfEpochs)
	if err != nil {
		log.Fatalln(err)
	}

	for i := range inputsTests {
		in := inputsTests[i]
		target := desiredTests[i]
		err := net.FeedForward(in)
		if err != nil {
			log.Printf("Feedforward (testing) caused error: %s", err.Error())
			return
		}
		fmt.Println("\n>>>Input:")
		fmt.Println("\t", in.RawMatrix().Data)
		out := net.GetOutput()
		fmt.Println(">>>Out:")
		fmt.Println("\t", out.RawMatrix().Data)
		fmt.Println(">>>Desired:")
		fmt.Println("\t", target.RawMatrix().Data)
	}

	fmt.Printf("Error on training data: %v\nError on test data: %v\n", trainErr, testErr)
}

func formTrainDataOR() ([]*mat.Dense, []*mat.Dense) {
	numExamples := 100000

	inputs := make([]*mat.Dense, numExamples)
	desired := make([]*mat.Dense, numExamples)
	for i := 0; i < numExamples; i++ {
		x := u.RandomInt(0, 2)
		y := u.RandomInt(0, 2)
		z := u.OrINT(x, y)
		input := mat.NewDense(2, 1, []float64{float64(x), float64(y)})
		target := mat.NewDense(1, 1, []float64{float64(z)})
		inputs[i] = input
		desired[i] = target
	}
	return inputs, desired
}

func formTestDataOR() ([]*mat.Dense, []*mat.Dense) {
	inputs := make([]*mat.Dense, 0, 4)
	desired := make([]*mat.Dense, 0, 4)

	input := mat.NewDense(2, 1, []float64{0, 0})
	target := mat.NewDense(1, 1, []float64{0})

	// 0 xor 0 = 0
	inputs = append(inputs, input)
	desired = append(desired, target)

	// 1 xor 0 = 1
	input = mat.NewDense(2, 1, []float64{1, 0})
	target = mat.NewDense(1, 1, []float64{1})
	inputs = append(inputs, input)
	desired = append(desired, target)

	// 0 xor 1 = 1
	input = mat.NewDense(2, 1, []float64{0, 1})
	target = mat.NewDense(1, 1, []float64{1})
	inputs = append(inputs, input)
	desired = append(desired, target)

	// 1 xor 1 = 0
	input = mat.NewDense(2, 1, []float64{1, 1})
	target = mat.NewDense(1, 1, []float64{1})
	inputs = append(inputs, input)
	desired = append(desired, target)

	return inputs, desired
}
