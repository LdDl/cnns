package nns

import (
	"encoding/json"
	"errors"
	"io/ioutil"
)

// WholeNet - net itself (array of layers)
type WholeNet struct {
	Layers []Layer
}

// Layer - interface for all layer types
type Layer interface {
	// OutSize - returns output size (dimensions)
	OutSize() Point

	// GetOutput - returns layer's output
	GetOutput() Tensor

	// GetWeights - returns layer's weights
	GetWeights() []Tensor

	// GetGradients - returns layer's gradients
	GetGradients() Tensor

	// FeedForward - feed data to layer
	FeedForward(t *Tensor)

	// CalculateGradients - calculate layers' gradients
	CalculateGradients(nextLayerGradients *Tensor)

	// UpdateWeights - update layer's weights
	UpdateWeights()

	// PrintOutput - print layer's output
	PrintOutput()

	// PrintWeights - print layer's weights
	PrintWeights()

	// PrintGradients - print layer's gradients
	PrintGradients()

	// GetType - get type of layer
	GetType() string

	SetActivationFunc(f func(v float64) float64)
	SetActivationDerivativeFunc(f func(v float64) float64)
	SetCustomWeights(t *[]Tensor)
}

// LayerStruct - struct wraps layer interface
type LayerStruct struct {
	Layer
}

// FeedForward - forward pass throught the net
func (wh *WholeNet) FeedForward(t *Tensor) {
	(*wh).Layers[0].FeedForward(t)
	for l := 1; l < len((*wh).Layers); l++ {
		out := (*wh).Layers[l-1].GetOutput()
		(*wh).Layers[l].FeedForward(&out)
	}
}

// Backpropagate - backward pass throught the net (training)
func (wh *WholeNet) Backpropagate(target *Tensor) {
	difference := (*wh).Layers[len((*wh).Layers)-1].GetOutput()
	difference.Sub(target)
	(*wh).Layers[len((*wh).Layers)-1].CalculateGradients(&difference)
	for i := len((*wh).Layers) - 2; i >= 0; i-- {
		grad := (*wh).Layers[i+1].GetGradients()
		(*wh).Layers[i].CalculateGradients(&grad)
	}
	for i := range (*wh).Layers {
		(*wh).Layers[i].UpdateWeights()
	}
}

// PrintOutput - prints net's output (last layer output)
func (wh *WholeNet) PrintOutput() {
	(*wh).Layers[len((*wh).Layers)-1].PrintOutput()
}

// GetOutput - returns net's output (last layer output)
func (wh *WholeNet) GetOutput() Tensor {
	return (*wh).Layers[len((*wh).Layers)-1].GetOutput()
}

// ExportToFile saves network to file
func (wh *WholeNet) ExportToFile(fname string) error {
	var err error
	var save NetJSON

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

// ImportFromFile load network to file
func (wh *WholeNet) ImportFromFile(fname string) error {
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
			var weights = make([]Tensor, numOfFilters)
			for w := 0; w < numOfFilters; w++ {
				weights[w] = NewTensor(kernelSize, kernelSize, 1)
				weights[w].SetData(data.Network.Layers[i].Weights[w].Data)
			}
			conv := NewConvLayer(stride, kernelSize, numOfFilters, TDsize{X: x, Y: y, Z: z})
			conv.SetCustomWeights(&weights)
			(*wh).Layers = append((*wh).Layers, conv)
			break
		case "relu":
			x := data.Network.Layers[i].InputSize.X
			y := data.Network.Layers[i].InputSize.Y
			z := data.Network.Layers[i].InputSize.Z
			relu := NewReLULayer(TDsize{X: x, Y: y, Z: z})
			(*wh).Layers = append((*wh).Layers, relu)
			break
		case "pool":
			stride := data.Network.Layers[i].Parameters.Stride
			kernelSize := data.Network.Layers[i].Parameters.KernelSize
			x := data.Network.Layers[i].InputSize.X
			y := data.Network.Layers[i].InputSize.Y
			z := data.Network.Layers[i].InputSize.Z
			pool := NewMaxPoolingLayer(stride, kernelSize, TDsize{X: x, Y: y, Z: z})
			(*wh).Layers = append((*wh).Layers, pool)
			break
		case "fc":
			break
		default:
			err = errors.New("Unrecognized layer type: " + data.Network.Layers[i].LayerType)
			return err
		}
	}
	return err
}

// NetJSON - json representation of network structure (for import and export)
type NetJSON struct {
	Network struct {
		Layers []struct {
			LayerType string `json:"LayerType"`
			InputSize struct {
				X int `json:"X"`
				Y int `json:"Y"`
				Z int `json:"Z"`
			} `json:"InputSize"`
			Parameters struct {
				Stride     int `json:"Stride"`
				KernelSize int `json:"KernelSize"`
			} `json:"Parameters,omitempty"`
			Weights []struct {
				TDSize struct {
					X int `json:"X"`
					Y int `json:"Y"`
					Z int `json:"Z"`
				} `json:"TDSize"`
				Data [][][]float64 `json:"Data"`
			} `json:"Weights,omitempty"`
		} `json:"Layers"`
	} `json:"Network"`
	Parameters struct {
		LearningRate float64 `json:"LearningRate"`
		Momentum     float64 `json:"Momentum"`
		WeightDecay  float64 `json:"WeightDecay"`
	} `json:"Parameters"`
}
