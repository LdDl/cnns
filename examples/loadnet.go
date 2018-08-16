package examples

import (
	"cnns_vika/nns"
	"fmt"
	"log"
)

// ImportNet - example of how ImportFromFile(fname string) works
func ImportNet() {
	jsonName := "datasets/conv_net.json"
	var net nns.WholeNet
	err := net.ImportFromFile(jsonName)
	if err != nil {
		log.Panicln(err)
	}
	fmt.Printf("Layers:\n")
	for i := range net.Layers {
		fmt.Printf("%v weights:\n", net.Layers[i].GetType())
		net.Layers[i].PrintWeights()
	}
}
