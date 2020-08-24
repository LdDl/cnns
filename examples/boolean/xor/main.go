package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/LdDl/cnns"
	t "github.com/LdDl/cnns/tensor"

	"github.com/LdDl/cnns/utils/u"
)

func main() {
	CheckXOR()
}

// CheckXOR - solve "XOR" problem
func CheckXOR() {
	rand.Seed(time.Now().UnixNano())
	// Fully connected layer with 3 output neurons
	fullyconnected1 := cnns.NewFullyConnectedLayer(&t.TDsize{X: 2, Y: 1, Z: 1}, 2)
	// There is 2 lines of reduntan code below, but it shows how to set definied activation function
	fullyconnected1.SetActivationFunc(cnns.ActivationTanh)
	fullyconnected1.SetActivationDerivativeFunc(cnns.ActivationTanhDerivative)

	// Fully connected layer with 1 output neurons
	fullyconnected2 := cnns.NewFullyConnectedLayer(fullyconnected1.GetOutputSize(), 1)
	// There is 2 lines of reduntan code below, but it shows how to set definied activation function
	fullyconnected2.SetActivationFunc(cnns.ActivationTanh)
	fullyconnected2.SetActivationDerivativeFunc(cnns.ActivationTanhDerivative)

	// Init network
	var net cnns.WholeNet
	net.Layers = append(net.Layers, fullyconnected1)
	net.Layers = append(net.Layers, fullyconnected2)

	// Init train and test data
	inputs, desired := formTrainDataXOR()
	inputsTests, desiredTests := formTestDataXOR()

	// Start traing process
	numOfEpochs := 1
	trainErr, testErr, err := net.Train(&inputs, &desired, &inputsTests, &desiredTests, numOfEpochs)
	if err != nil {
		log.Fatalln(err)
	}

	fmt.Printf("Error on training data: %v\nError on test data: %v\n", trainErr, testErr)
}

func formTrainDataXOR() ([]t.Tensor, []t.Tensor) {
	numExamples := 100000

	inputs := make([]t.Tensor, numExamples)
	desired := make([]t.Tensor, numExamples)
	for i := 0; i < numExamples; i++ {
		x := u.RandomInt(0, 2)
		y := u.RandomInt(0, 2)

		z := u.XorINT(x, y)

		input := t.NewTensor(2, 1, 1)
		input.SetData(2, 1, 1, []float64{float64(x), float64(y)})

		target := t.NewTensor(1, 1, 1)
		target.SetData(1, 1, 1, []float64{float64(z)})

		inputs[i] = input
		desired[i] = target
	}
	return inputs, desired
}

func formTestDataXOR() ([]t.Tensor, []t.Tensor) {
	inputs := make([]t.Tensor, 0, 4)
	desired := make([]t.Tensor, 0, 4)

	input := t.NewTensor(2, 1, 1)
	target := t.NewTensor(1, 1, 1)

	// 0 or 0 = 0
	input.SetData(2, 1, 1, []float64{0, 0})
	target.SetData(1, 1, 1, []float64{0})
	inputs = append(inputs, input)
	desired = append(desired, target)

	// 1 or 0 = 1
	input = t.NewTensor(2, 1, 1)
	input.SetData(2, 1, 1, []float64{1, 0})
	target = t.NewTensor(1, 1, 1)
	target.SetData(1, 1, 1, []float64{1})
	inputs = append(inputs, input)
	desired = append(desired, target)

	// 0 or 1 = 1
	input = t.NewTensor(2, 1, 1)
	input.SetData(2, 1, 1, []float64{0, 1})
	target = t.NewTensor(1, 1, 1)
	target.SetData(1, 1, 1, []float64{1})
	inputs = append(inputs, input)
	desired = append(desired, target)

	// 1 or 1 = 0
	input = t.NewTensor(2, 1, 1)
	input.SetData(2, 1, 1, []float64{1, 1})
	target = t.NewTensor(1, 1, 1)
	target.SetData(1, 1, 1, []float64{0})
	inputs = append(inputs, input)
	desired = append(desired, target)

	return inputs, desired
}
