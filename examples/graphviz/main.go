package main

import (
	"fmt"

	"github.com/LdDl/cnns"
	t "github.com/LdDl/cnns/tensor"
)

func main() {
	var net cnns.WholeNet

	fullyconnected1 := cnns.NewFullyConnectedLayer(&t.TDsize{X: 2, Y: 1, Z: 1}, 2)
	fullyconnected2 := cnns.NewFullyConnectedLayer(fullyconnected1.GetOutputSize(), 5)
	fullyconnected3 := cnns.NewFullyConnectedLayer(fullyconnected2.GetOutputSize(), 3)
	fullyconnected4 := cnns.NewFullyConnectedLayer(fullyconnected3.GetOutputSize(), 1)

	net.Layers = append(net.Layers, fullyconnected1)
	net.Layers = append(net.Layers, fullyconnected2)
	net.Layers = append(net.Layers, fullyconnected3)
	net.Layers = append(net.Layers, fullyconnected4)

	graphvizText := net.GetGraphvizText()
	fmt.Println(graphvizText)
}
