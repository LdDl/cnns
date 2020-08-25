package cnns

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"strings"

	t "github.com/LdDl/cnns/tensor"
)

// WholeNet - net itself (array of layers)
type WholeNet struct {
	Layers []Layer
	LP     LearningParams
}

// FeedForward - forward pass through the net
func (wh *WholeNet) FeedForward(t *t.Tensor) {
	wh.Layers[0].FeedForward(t)
	for l := 1; l < len(wh.Layers); l++ {
		out := wh.Layers[l-1].GetOutput()
		wh.Layers[l].FeedForward(out)
	}
}

// Backpropagate - backward pass through the net (training)
func (wh *WholeNet) Backpropagate(target *t.Tensor) {
	lastLayer := wh.Layers[len(wh.Layers)-1].GetOutput()

	difference := lastLayer.Sub(target)
	wh.Layers[len(wh.Layers)-1].CalculateGradients(difference)

	for i := len(wh.Layers) - 2; i >= 0; i-- {
		grad := wh.Layers[i+1].GetGradients()
		wh.Layers[i].CalculateGradients(grad)
	}
	for i := range wh.Layers {
		wh.Layers[i].UpdateWeights()
	}
}

// PrintOutput - prints net's output (last layer output)
func (wh *WholeNet) PrintOutput() {
	wh.Layers[len(wh.Layers)-1].PrintOutput()
}

// GetOutput - returns net's output (last layer output)
func (wh *WholeNet) GetOutput() *t.Tensor {
	return wh.Layers[len(wh.Layers)-1].GetOutput()
}

// ImportFromFile load network to file
/*
	fname - filename,
	randomWeights:
		true: random weights for new network
		false: weights from files for using network (or continue training))
*/
func (wh *WholeNet) ImportFromFile(fname string, randomWeights bool) error {
	var err error
	fileBytes, err := ioutil.ReadFile(fname)
	if err != nil {
		return err
	}
	var data NetJSON
	err = json.Unmarshal(fileBytes, &data)
	if err != nil {
		return err
	}
	for i := range data.Network.Layers {
		switch data.Network.Layers[i].LayerType {
		case "conv":
			stride := data.Network.Layers[i].Parameters.Stride
			kernelSize := data.Network.Layers[i].Parameters.KernelSize
			numOfFilters := len(data.Network.Layers[i].Weights)
			x := data.Network.Layers[i].InputSize.X
			y := data.Network.Layers[i].InputSize.Y
			z := data.Network.Layers[i].InputSize.Z
			conv := NewConvLayer(stride, kernelSize, numOfFilters, t.TDsize{X: x, Y: y, Z: z})
			if randomWeights == false {
				var weights = make([]*t.Tensor, numOfFilters)
				for w := 0; w < numOfFilters; w++ {
					weights[w] = t.NewTensor(kernelSize, kernelSize, 1)
					weights[w].SetData3D(data.Network.Layers[i].Weights[w].Data)
				}
				conv.SetCustomWeights(weights)
			}
			wh.Layers = append(wh.Layers, conv)
			break
		case "relu":
			x := data.Network.Layers[i].InputSize.X
			y := data.Network.Layers[i].InputSize.Y
			z := data.Network.Layers[i].InputSize.Z
			relu := NewReLULayer(&t.TDsize{X: x, Y: y, Z: z})
			wh.Layers = append(wh.Layers, relu)
			break
		case "pool":
			stride := data.Network.Layers[i].Parameters.Stride
			kernelSize := data.Network.Layers[i].Parameters.KernelSize
			x := data.Network.Layers[i].InputSize.X
			y := data.Network.Layers[i].InputSize.Y
			z := data.Network.Layers[i].InputSize.Z
			pool := NewMaxPoolingLayer(stride, kernelSize, &t.TDsize{X: x, Y: y, Z: z})
			wh.Layers = append(wh.Layers, pool)
			break
		case "fc":
			x := data.Network.Layers[i].InputSize.X
			y := data.Network.Layers[i].InputSize.Y
			z := data.Network.Layers[i].InputSize.Z
			outSize := data.Network.Layers[i].OutputSize.X
			fullyconnected := NewFullyConnectedLayer(&t.TDsize{X: x, Y: y, Z: z}, outSize)
			if randomWeights == false {
				var weights *t.Tensor
				weights = t.NewTensor(x*y*z, outSize, 1)
				weights.SetData3D(data.Network.Layers[i].Weights[0].Data)
				fullyconnected.SetCustomWeights([]*t.Tensor{weights})
			}
			wh.Layers = append(wh.Layers, fullyconnected)
			break
		default:
			err = errors.New("Unrecognized layer type: " + data.Network.Layers[i].LayerType)
			return err
		}
	}

	wh.LP.LearningRate = data.Parameters.LearningRate
	wh.LP.Momentum = data.Parameters.Momentum
	wh.LP.WeightDecay = data.Parameters.WeightDecay
	return err
}

// ExportToFile saves network to file
func (wh *WholeNet) ExportToFile(fname string) error {
	var err error
	var save NetJSON

	for i := 0; i < len(wh.Layers); i++ {
		switch wh.Layers[i].GetType() {
		case "conv":
			var newLayer NetLayerJSON
			newLayer.LayerType = "conv"
			newLayer.InputSize = wh.Layers[i].GetInputSize()
			newLayer.Parameters.Stride = wh.Layers[i].GetStride()
			newLayer.Parameters.KernelSize = wh.Layers[i].GetKernelSize()
			kernels := wh.Layers[i].GetWeights()
			newLayer.Weights = make([]TensorJSON, len(kernels))
			for k := range kernels {
				newLayer.Weights[k].TDSize = kernels[k].Size
				newLayer.Weights[k].Data = kernels[k].GetData3D()
			}
			save.Network.Layers = append(save.Network.Layers, newLayer)
			break
		case "relu":
			var newLayer NetLayerJSON
			newLayer.LayerType = "relu"
			newLayer.InputSize = wh.Layers[i].GetInputSize()
			save.Network.Layers = append(save.Network.Layers, newLayer)
			break
		case "pool":
			var newLayer NetLayerJSON
			newLayer.LayerType = "pool"
			newLayer.InputSize = wh.Layers[i].GetInputSize()
			newLayer.Parameters.Stride = wh.Layers[i].GetStride()
			newLayer.Parameters.KernelSize = wh.Layers[i].GetKernelSize()
			save.Network.Layers = append(save.Network.Layers, newLayer)
			break
		case "fc":
			var newLayer NetLayerJSON
			newLayer.LayerType = "fc"
			newLayer.InputSize = wh.Layers[i].GetInputSize()
			newLayer.OutputSize = wh.Layers[i].GetOutput().Size
			newLayer.Weights = make([]TensorJSON, 1)
			kernels := wh.Layers[i].GetWeights()
			if len(kernels) != 1 {
				err = fmt.Errorf("Fully connected layer can have only 1 'kernel'")
				return err
			}
			newLayer.Weights[0].TDSize = kernels[0].Size
			newLayer.Weights[0].Data = kernels[0].GetData3D()

			save.Network.Layers = append(save.Network.Layers, newLayer)
			break
		default:
			err = fmt.Errorf("Unrecognized layer type: %v", wh.Layers[i].GetType())
			return err
		}
	}

	// Hardcoded training parameters
	save.Parameters.LearningRate = 0.01
	save.Parameters.Momentum = 0.6
	save.Parameters.WeightDecay = 0.001

	saveJSON, err := json.Marshal(save)
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(fname, saveJSON, 0644)
	if err != nil {
		return err
	}

	return err
}

// NetJSON - json representation of network structure (for import and export)
type NetJSON struct {
	Network    NetworkJSON    `json:"Network"`
	Parameters LearningParams `json:"Parameters"`
}

// TensorJSON ...
type TensorJSON struct {
	TDSize *t.TDsize     `json:"TDSize"`
	Data   [][][]float64 `json:"Data"`
}

// LayerParamsJSON ...
type LayerParamsJSON struct {
	Stride     int `json:"Stride"`
	KernelSize int `json:"KernelSize"`
}

// NetLayerJSON ...
type NetLayerJSON struct {
	LayerType  string          `json:"LayerType"`
	InputSize  *t.TDsize       `json:"InputSize"`
	Parameters LayerParamsJSON `json:"Parameters,omitempty"`
	Weights    []TensorJSON    `json:"Weights,omitempty"`
	// Actually "OutputSize" parameter is useful for fully connected layer only
	// There are automatic calculation of output size for other layers' types
	OutputSize *t.TDsize `json:"OutputSize,omitempty"`
}

// NetworkJSON ...
type NetworkJSON struct {
	Layers []NetLayerJSON `json:"Layers"`
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

		size := wh.Layers[l].GetOutput().Size
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
