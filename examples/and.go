package examples

import (
	"cnns_vika/nns"
	"cnns_vika/utils/u"
	"math"
	"math/rand"
	"time"
)

// ActivationTanh .
func ActivationTanh(v float64) float64 {
	return math.Tanh(v)
}

// ActivationTanhDerivative .
func ActivationTanhDerivative(v float64) float64 {
	return 1 - ActivationTanh(v)*ActivationTanh(v)
}

// CheckAND - проверка полносвязного слоя при решении проблемы AND
func CheckAND() {
	rand.Seed(time.Now().UnixNano())
	// Слой с тремя нейронами
	fullyconnected1 := nns.NewFullConnectedLayer(nns.TDsize{X: 2, Y: 1, Z: 1}, 2)
	fullyconnected1.SetActivationFunc(ActivationTanh)
	fullyconnected1.SetActivationDerivativeFunc(ActivationTanhDerivative)

	// Слой с одним выходным нейроном
	fullyconnected2 := nns.NewFullConnectedLayer(fullyconnected1.OutSize(), 1)
	fullyconnected2.SetActivationFunc(ActivationTanh)
	fullyconnected2.SetActivationDerivativeFunc(ActivationTanhDerivative)

	// Инициализация сети
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
		outputBool := (firstBool && secondBool)
		outputInt := 0
		if outputBool == true {
			outputInt = 1
		}
		desired := nns.NewTensor(1, 1, 1)
		desired.SetData([][][]float64{[][]float64{[]float64{float64(outputInt)}}})
		input := nns.NewTensor(2, 1, 1)
		input.SetData([][][]float64{[][]float64{[]float64{float64(firstInt), float64(secondInt)}}})
		// Forward
		net.FeedForward(&input)
		// Backward
		net.Backpropagate(&desired)
	}

	// 0 * 0
	inputTest := nns.NewTensor(2, 1, 1)
	net.FeedForward(&inputTest)
	net.PrintOutput()

	// 1 * 0
	inputTest.SetData([][][]float64{[][]float64{[]float64{1.0, 0}}})
	net.FeedForward(&inputTest)
	net.PrintOutput()

	// 0 * 1
	inputTest.SetData([][][]float64{[][]float64{[]float64{0, 1.0}}})
	net.FeedForward(&inputTest)
	net.PrintOutput()

	// 1 * 1
	inputTest.SetData([][][]float64{[][]float64{[]float64{1.0, 1.0}}})
	net.FeedForward(&inputTest)
	net.PrintOutput()
}
