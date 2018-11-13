package examples

import (
	"fmt"
	"log"

	"github.com/LdDl/cnns/nns"
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

	inputData := nns.NewTensor(2, 1, 1)
	inputData.SetData(2, 1, 1, []float64{0.2, 0.5})

	for e := 0; e < 3; e++ {
		net.FeedForward(&inputData)
		// obj := net.GetOutput()
		// log.Println("out")
		// obj.Print()
		desired := nns.NewTensor(1, 1, 1)
		desired.SetData(1, 1, 1, []float64{0.4})
		net.Backpropagate(&desired)
		fmt.Printf("Layers (after):\n")
		for i := range net.Layers {
			fmt.Printf("%v weights:\n", net.Layers[i].GetType())
			net.Layers[i].PrintWeights()
		}
	}
	net.PrintOutput()
}
