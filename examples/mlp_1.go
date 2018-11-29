package examples

import (
	"fmt"
	"log"

	"github.com/LdDl/cnns/nns"
	t "github.com/LdDl/cnns/nns/tensor"
)

// ExampleOne - Example of MLP
func ExampleOne() {
	jsonName := "datasets/mlp_example1.json"
	var net nns.WholeNet
	err := net.ImportFromFile(jsonName, false)
	if err != nil {
		log.Panicln(err)
	}
	fmt.Printf("Layers:\n")
	for i := range net.Layers {
		fmt.Printf("%v weights:\n", net.Layers[i].GetType())
		net.Layers[i].PrintWeights()
	}

	inputData := t.NewTensor(2, 1, 1)
	inputData.SetData(2, 1, 1, []float64{0.2, 0.5})

	for e := 0; e < 3; e++ {
		net.FeedForward(&inputData)
		desired := t.NewTensor(1, 1, 1)
		desired.SetData(1, 1, 1, []float64{0.4})
		net.Backpropagate(&desired)
		fmt.Printf("Layers (after):\n")
		for i := range net.Layers {
			fmt.Printf("%v weights:\n", net.Layers[i].GetType())
			net.Layers[i].PrintWeights()
		}
	}

	// net.FeedForward(&inputData)
	// net.PrintOutput()
}

// ExampleTwo - Example of MLP
func ExampleTwo() {
	jsonName := "datasets/mlp_example2.json"
	var net nns.WholeNet
	err := net.ImportFromFile(jsonName, false)
	if err != nil {
		log.Panicln(err)
	}
	fmt.Printf("Layers:\n")
	for i := range net.Layers {
		fmt.Printf("%v weights:\n", net.Layers[i].GetType())
		net.Layers[i].PrintWeights()
	}
	net.Layers[0].SetActivationFunc(nns.ActivationSygmoid)
	net.Layers[1].SetActivationFunc(nns.ActivationSygmoid)

	net.Layers[0].SetActivationDerivativeFunc(nns.ActivationSygmoidDerivative)
	net.Layers[1].SetActivationDerivativeFunc(nns.ActivationSygmoidDerivative)

	inputData := t.NewTensor(2, 1, 1)
	inputData.SetData(2, 1, 1, []float64{0.2, 0.5})

	for e := 0; e < 3; e++ {
		net.FeedForward(&inputData)
		desired := t.NewTensor(1, 1, 1)
		desired.SetData(1, 1, 1, []float64{0.4})
		net.Backpropagate(&desired)
		fmt.Printf("Layers (after):\n")
		for i := range net.Layers {
			fmt.Printf("%v weights:\n", net.Layers[i].GetType())
			net.Layers[i].PrintWeights()
		}
	}

	net.FeedForward(&inputData)
	net.PrintOutput()
}
