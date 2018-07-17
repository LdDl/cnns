package examples

import (
	"cnns_vika/nns"
	"cnns_vika/utils/u"
	"math/rand"
	"time"
)

// CheckOR - проверка полносвязного слоя при решении проблемы OR
func CheckOR() {
	rand.Seed(time.Now().UnixNano())
	// Слой с тремя нейронами
	flayer1 := nns.NewFullConnectedLayer(2, 1, 1, 2, true, false)
	fullyconnected1 := &nns.LayerStruct{
		Layer: flayer1,
	}
	// Слой с одним выходным нейроном
	flayer2 := nns.NewFullConnectedLayer(2, 1, 1, 1, true, true)
	fullyconnected2 := &nns.LayerStruct{
		Layer: flayer2,
	}
	// Инициализация сети
	var net nns.WholeNet
	net.Layers = append(net.Layers, fullyconnected1)
	net.Layers = append(net.Layers, fullyconnected2)

	for i := 0; i < 50000; i++ {
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
		desired := nns.NewTensorEmpty(1, 1, 1)
		desired.Set(&[][][]float64{[][]float64{[]float64{float64(outputInt)}}})
		input := nns.NewTensorEmpty(2, 1, 1)
		input.Set(&[][][]float64{[][]float64{[]float64{float64(firstInt), float64(secondInt)}}})

		// Forward
		net.Layers[0].FeedForward(input)
		net.Layers[1].FeedForward(net.Layers[0].GetOutput())

		//Backward
		difference := net.Layers[1].GetOutput().Sub(desired)
		// fmt.Printf("desired: %v, out: %v, difference: %v\n", desired.Data[0], net.Layers[0].GetOutput().Data[0], difference.Data[0])
		net.Layers[1].CalculateGradients(difference)
		net.Layers[0].CalculateGradients(net.Layers[1].GetGradients())
		net.Layers[1].UpdateWeights()
		net.Layers[0].UpdateWeights()

		// log.Println(firstInt, secondInt, outputInt, net.Layers[1].GetOutput().Data[0])

	}

	inputTest := nns.NewTensorEmpty(2, 1, 1)
	inputTest.Set(&[][][]float64{[][]float64{[]float64{0, 0}}})
	net.Layers[0].FeedForward(inputTest)
	net.Layers[1].FeedForward(net.Layers[0].GetOutput())
	net.Layers[1].PrintOutput()

	inputTest.Set(&[][][]float64{[][]float64{[]float64{1.0, 0}}})
	net.Layers[0].FeedForward(inputTest)
	net.Layers[1].FeedForward(net.Layers[0].GetOutput())
	net.Layers[1].PrintOutput()

	inputTest.Set(&[][][]float64{[][]float64{[]float64{0, 1.0}}})
	net.Layers[0].FeedForward(inputTest)
	net.Layers[1].FeedForward(net.Layers[0].GetOutput())
	net.Layers[1].PrintOutput()

	inputTest.Set(&[][][]float64{[][]float64{[]float64{1.0, 1.0}}})
	net.Layers[0].FeedForward(inputTest)
	net.Layers[1].FeedForward(net.Layers[0].GetOutput())
	net.Layers[1].PrintOutput()
}
