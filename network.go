package cnns

import (
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// WholeNet Neural net itself (slice of layers)
type WholeNet struct {
	Layers []Layer
	LP     *LearningParams
}

// FeedForward Forward pass through the net
func (wh *WholeNet) FeedForward(input *mat.Dense) error {
	wh.Layers[0].FeedForward(input)
	for l := 1; l < len(wh.Layers); l++ {
		outDense := wh.Layers[l-1].GetActivatedOutput()
		err := wh.Layers[l].FeedForward(outDense)
		if err != nil {
			return errors.Wrap(err, "Can't call FeedForward() on neural net")
		}
	}
	return nil
}

// Backpropagate Backward pass through the net (training)
func (wh *WholeNet) Backpropagate(Tk *mat.Dense) error {
	Ok := wh.Layers[len(wh.Layers)-1].GetActivatedOutput()
	/*
		Chain rule for backpropagation is:
			Δw{j}{k} = ΔE{k}/Δw{j}{k}
			Δw{j}{k} = ΔE{k}/ΔO{k} * ΔO{k}/Δw{j}{k}
			Δw{j}{k} = ΔE{k}/ΔO{k} * ΔO{k}/ΔΣ(k) * ΔΣ(k)/Δw{j}{k}
	*/

	/*
		Error on last layer is defined as:
			E{k} = (1/2) * (T{k} - O{k}) ^ 2, where
				T{k} - desired target
				O{k} = activate(Σw{j}{k}*o{j}) - activated output, where o{j} - is input
		Derivative of E{k} with respect to O{k}:
			ΔE{k} / Δo{k} = -(T{k} - O{k}) = (O{k} - T{k})
	*/

	// Evaluate ΔE{k}/ΔO{k}
	Ediff := &mat.Dense{}
	// fmt.Println(Ok, Tk)
	Ediff.Sub(Ok, Tk)

	// Evaluate ΔE{k}/ΔO{k} * ΔO{k}/ΔΣ(k) * ΔΣ(k)/Δw{j}{k}
	err := wh.Layers[len(wh.Layers)-1].CalculateGradients(Ediff)
	if err != nil {
		return errors.Wrap(err, "Can't call CalculateGradients() on last layer of neural net")
	}

	// Do job for every hidden layer
	for i := len(wh.Layers) - 2; i >= 0; i-- {
		gradDense := wh.Layers[i+1].GetGradients()
		err := wh.Layers[i].CalculateGradients(gradDense)
		if err != nil {
			return errors.Wrap(err, "Can't call CalculateGradients() while doing backpropagation")
		}
	}

	// Update weights
	for i := range wh.Layers {
		wh.Layers[i].UpdateWeights(wh.LP)
	}
	return nil
}

// PrintOutput Print net's output (last layer output)
func (wh *WholeNet) PrintOutput() {
	wh.Layers[len(wh.Layers)-1].PrintOutput()
}

// GetOutput Return net's output (last layer output)
func (wh *WholeNet) GetOutput() *mat.Dense {
	return wh.Layers[len(wh.Layers)-1].GetActivatedOutput()
}

// GetGraphvizText Returns Graphviz text-based output
func (wh *WholeNet) GetGraphvizText() string {
	graph := "digraph G {rankdir = LR;splines=false;edge[style=invis];ranksep= 1.4;"

	if len(wh.Layers) == 0 {
		return ""
	}
	inputSize := wh.Layers[0].GetInputSize()
	inputVertices := []string{}
	inputVerticesLabels := []string{}
	inputNodeProperties := "node [shape=circle, color=chartreuse, style=filled, fillcolor=chartreuse]"
	for x := 0; x < inputSize.X; x++ {
		vertex := fmt.Sprintf("x%[1]d [label=<x<sub>%[1]d</sub><sup>(0)</sup>>]", x)
		inputVerticesLabels = append(inputVerticesLabels, vertex)
		inputVertices = append(inputVertices, fmt.Sprintf("x%[1]d", x))
	}
	inputLayerVertices := fmt.Sprintf("{%s}", strings.Join(inputVertices, ";"))
	inputNodeProperties = fmt.Sprintf("{%s;%s;}", inputNodeProperties, strings.Join(inputVerticesLabels, ";"))
	inputRankProperties := fmt.Sprintf("{rank=same;%s;}", strings.Join(inputVertices, "->"))
	inputLayerProperties := fmt.Sprintf("l_input [shape=plaintext, label=\"Input layer\"];")
	inputLayerRankProperties := fmt.Sprintf("{rank=same; l_input;%[1]s};", inputVertices[0])
	graph += inputNodeProperties
	graph += inputRankProperties
	graph += inputLayerProperties
	graph += inputLayerRankProperties

	layersVertices := []string{}

	for l := range wh.Layers {
		nodeProperties := ""

		vertices := []string{}
		verticesLabels := []string{}
		layerType := ""

		size := wh.Layers[l].GetOutputSize()
		switch l {
		case len(wh.Layers) - 1:
			nodeProperties = "node [shape=circle, color=coral1, style=filled, fillcolor=coral1]"
			for x := 0; x < size.X; x++ {
				vertex := fmt.Sprintf("O%[1]d [label=<o<sub>%[1]d</sub><sup>(%[2]d)</sup>>]", x, l)
				verticesLabels = append(verticesLabels, vertex)
				vertices = append(vertices, fmt.Sprintf("O%[1]d", x))
			}
			layerType = "output"
			break
		default:
			nodeProperties = "node [shape=circle, color=dodgerblue, style=filled, fillcolor=dodgerblue]"
			for x := 0; x < size.X; x++ {
				vertex := fmt.Sprintf("h%[1]d%[2]d [label=<h<sub>%[1]d</sub><sup>(%[2]d)</sup>>]", x, l)
				verticesLabels = append(verticesLabels, vertex)
				vertices = append(vertices, fmt.Sprintf("h%[1]d%[2]d", x, l))
			}
			layerType = "hidden"
			break
		}

		nodeProperties = fmt.Sprintf("{%s;%s;}", nodeProperties, strings.Join(verticesLabels, ";"))
		rankProperties := fmt.Sprintf("{rank=same;%s;}", strings.Join(vertices, "->"))
		layerProperties := fmt.Sprintf("l%[1]d [shape=plaintext, label=\"layer %[1]d (%[2]s layer)\"];", l, layerType)
		layerRankProperties := fmt.Sprintf("{rank=same; l%[1]d;%[2]s};", l, vertices[0])

		layerVertices := fmt.Sprintf("{%s}", strings.Join(vertices, ";"))
		layersVertices = append(layersVertices, layerVertices)

		graph += nodeProperties
		graph += rankProperties
		graph += layerProperties
		graph += layerRankProperties
	}

	edgesStyle := "edge[style=solid, tailport=e, headport=w];"
	graph += edgesStyle
	inputEdges := fmt.Sprintf("%[1]s -> %[2]s;", inputLayerVertices, layersVertices[0])
	graph += inputEdges
	for l := 1; l < len(layersVertices); l++ {
		edges := fmt.Sprintf("%[1]s -> %[2]s;", layersVertices[l-1], layersVertices[l])
		graph += edges
	}

	graph += "}"

	return graph
}
