package main

import (
	"fmt"
	"log"

	"github.com/LdDl/cnns"
	"gonum.org/v1/gonum/mat"
)

func main() {
	ExampleOne()
	// ExampleTwo()
}

// ExampleOne - Example of MLP. Corresponding file is "step_by_step_mlp(inertia).xlsx"
func ExampleOne() {
	jsonName := "../datasets/mlp_example1.json"
	var net cnns.WholeNet
	err := net.ImportFromFile(jsonName, false)
	if err != nil {
		log.Panicln(err)
	}
	fmt.Printf("Layers weights:\n")
	for i := range net.Layers {
		fmt.Printf("%s #%d weights:\n", net.Layers[i].GetType(), i)
		net.Layers[i].PrintWeights()
	}

	fmt.Println("\tDoing training....")
	inputDense := mat.NewDense(2, 1, []float64{0.2, 0.5})
	for e := 0; e < 3; e++ {
		err := net.FeedForward(inputDense)
		if err != nil {
			log.Printf("Feedforward caused error: %s", err.Error())
			return
		}
		desired := mat.NewDense(1, 1, []float64{0.4})
		err = net.Backpropagate(desired)
		if err != nil {
			log.Printf("Backpropagate caused error: %s", err.Error())
			return
		}

		fmt.Printf("Epoch #%d. New layers weights\n", e)
		for i := range net.Layers {
			fmt.Printf("%s #%d weights on epoch #%d:\n", net.Layers[i].GetType(), i, e)
			net.Layers[i].PrintWeights()
		}
	}

	fmt.Println("Feedforward one more time. Result is:")
	net.FeedForward(inputDense)
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
	fmt.Printf("Layers weights:\n")
	for i := range net.Layers {
		fmt.Printf("%s #%d weights:\n", net.Layers[i].GetType(), i)
		net.Layers[i].PrintWeights()
	}
	net.Layers[0].SetActivationFunc(cnns.ActivationSygmoid)
	net.Layers[1].SetActivationFunc(cnns.ActivationSygmoid)

	net.Layers[0].SetActivationDerivativeFunc(cnns.ActivationSygmoidDerivative)
	net.Layers[1].SetActivationDerivativeFunc(cnns.ActivationSygmoidDerivative)

	fmt.Println("\tDoing training....")
	inputDense := mat.NewDense(2, 1, []float64{0.2, 0.5})
	for e := 0; e < 3; e++ {
		err := net.FeedForward(inputDense)
		if err != nil {
			log.Printf("Feedforward caused error: %s", err.Error())
			return
		}
		desired := mat.NewDense(1, 1, []float64{0.4})
		err = net.Backpropagate(desired)
		if err != nil {
			log.Printf("Backpropagate caused error: %s", err.Error())
			return
		}
		fmt.Printf("Epoch #%d. New layers weights\n", e)
		for i := range net.Layers {
			fmt.Printf("%s #%d weights on epoch #%d:\n", net.Layers[i].GetType(), i, e)
			net.Layers[i].PrintWeights()
		}
	}

	fmt.Println("Feedforward one more time. Result is:")
	net.FeedForward(inputDense)
	net.PrintOutput()
}
