package examples

import (
	"math/rand"
	"time"

	"github.com/LdDl/cnns/nns"

	"github.com/LdDl/cnns/utils/u"
)

// CheckOR - solve "OR" problem
func CheckOR() {
	rand.Seed(time.Now().UnixNano())
	// Fully connected layer with 3 output neurons
	fullyconnected1 := nns.NewFullConnectedLayer(nns.TDsize{X: 2, Y: 1, Z: 1}, 2)
	// There is 2 lines of reduntan code below, but it shows how to set definied activation function
	fullyconnected1.SetActivationFunc(nns.ActivationTanh)
	fullyconnected1.SetActivationDerivativeFunc(nns.ActivationTanhDerivative)

	// Fully connected layer with 1 output neurons
	fullyconnected2 := nns.NewFullConnectedLayer(fullyconnected1.OutSize(), 1)
	// There is 2 lines of reduntan code below, but it shows how to set definied activation function
	fullyconnected2.SetActivationFunc(nns.ActivationTanh)
	fullyconnected2.SetActivationDerivativeFunc(nns.ActivationTanhDerivative)

	// Init network
	var net nns.WholeNet
	net.Layers = append(net.Layers, fullyconnected1)
	net.Layers = append(net.Layers, fullyconnected2)
	for i := 0; i < 100000; i++ {
		firstInt := u.RandomInt(0, 2)
		secondInt := u.RandomInt(0, 2)
		firstBool := false
		secondBool := false
		if firstInt == 1 {
			firstBool = true
		}
		if secondInt == 1 {
			secondBool = true
		}
		outputBool := (firstBool || secondBool)
		outputInt := 0
		if outputBool == true {
			outputInt = 1
		}
		desired := nns.NewTensor(1, 1, 1)
		desired.SetData3D([][][]float64{[][]float64{[]float64{float64(outputInt)}}})
		input := nns.NewTensor(2, 1, 1)
		input.SetData3D([][][]float64{[][]float64{[]float64{float64(firstInt), float64(secondInt)}}})
		// Forward
		net.FeedForward(&input)
		// Backward
		net.Backpropagate(&desired)
	}

	// 0 * 0
	inputTest := nns.NewTensor(2, 1, 1)
	net.FeedForward(&inputTest)
	net.Layers[1].PrintOutput()

	// 1 * 0
	inputTest.SetData3D([][][]float64{[][]float64{[]float64{1.0, 0}}})
	net.FeedForward(&inputTest)
	net.PrintOutput()

	// 0 * 1
	inputTest.SetData3D([][][]float64{[][]float64{[]float64{0, 1.0}}})
	net.FeedForward(&inputTest)
	net.PrintOutput()

	// 1 * 1
	inputTest.SetData3D([][][]float64{[][]float64{[]float64{1.0, 1.0}}})
	net.FeedForward(&inputTest)
	net.PrintOutput()
}
