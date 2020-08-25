package main

import (
	"fmt"
	"log"

	"github.com/LdDl/cnns"
	t "github.com/LdDl/cnns/tensor"
)

func main() {
	ExampleOne()
	// ExampleTwo()
}

// ExampleOne - Example of MLP
func ExampleOne() {
	jsonName := "../datasets/mlp_example1.json"
	var net cnns.WholeNet
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
		net.FeedForward(inputData)
		desired := t.NewTensor(1, 1, 1)
		desired.SetData(1, 1, 1, []float64{0.4})
		net.Backpropagate(desired)
		fmt.Printf("Layers (after):\n")
		for i := range net.Layers {
			fmt.Printf("%v weights:\n", net.Layers[i].GetType())
			net.Layers[i].PrintWeights()
		}
	}

	net.FeedForward(inputData)
	net.PrintOutput()
}

// ExampleTwo - Example of MLP
func ExampleTwo() {
	jsonName := "../datasets/mlp_example2.json"
	var net cnns.WholeNet
	err := net.ImportFromFile(jsonName, false)
	if err != nil {
		log.Panicln(err)
	}
	fmt.Printf("Layers:\n")
	for i := range net.Layers {
		fmt.Printf("%v weights:\n", net.Layers[i].GetType())
		net.Layers[i].PrintWeights()
	}
	net.Layers[0].SetActivationFunc(cnns.ActivationSygmoid)
	net.Layers[1].SetActivationFunc(cnns.ActivationSygmoid)

	net.Layers[0].SetActivationDerivativeFunc(cnns.ActivationSygmoidDerivative)
	net.Layers[1].SetActivationDerivativeFunc(cnns.ActivationSygmoidDerivative)

	inputData := t.NewTensor(2, 1, 1)
	inputData.SetData(2, 1, 1, []float64{0.2, 0.5})

	for e := 0; e < 3; e++ {
		net.FeedForward(inputData)
		desired := t.NewTensor(1, 1, 1)
		desired.SetData(1, 1, 1, []float64{0.4})
		net.Backpropagate(desired)
		fmt.Printf("Layers (after):\n")
		for i := range net.Layers {
			fmt.Printf("%v weights:\n", net.Layers[i].GetType())
			net.Layers[i].PrintWeights()
		}
	}

	net.FeedForward(inputData)
	net.PrintOutput()
}
