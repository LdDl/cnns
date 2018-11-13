package nns

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
)

// WholeNet - net itself (array of layers)
type WholeNet struct {
	Layers []Layer
	LP     LearningParams
}

// Layer - interface for all layer types
type Layer interface {
	// OutSize - returns output size (dimensions)
	OutSize() Point

	// GetInputSize - returns input size (dimensions)
	GetInputSize() Point

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

	// GetStride - get stride of layer
	GetStride() int

	// GetKernelSize - get kernel size of layer
	GetKernelSize() int

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

	// target.Sub(&difference)
	// (*wh).Layers[len((*wh).Layers)-1].CalculateGradients(target)
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
			conv := NewConvLayer(stride, kernelSize, numOfFilters, TDsize{X: x, Y: y, Z: z})
			if randomWeights == false {
				var weights = make([]Tensor, numOfFilters)
				for w := 0; w < numOfFilters; w++ {
					weights[w] = NewTensor(kernelSize, kernelSize, 1)
					weights[w].SetData3D(data.Network.Layers[i].Weights[w].Data)
				}
				conv.SetCustomWeights(&weights)
			}
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
			x := data.Network.Layers[i].InputSize.X
			y := data.Network.Layers[i].InputSize.Y
			z := data.Network.Layers[i].InputSize.Z
			outSize := data.Network.Layers[i].OutputSize.X
			fullyconnected := NewFullConnectedLayer(TDsize{X: x, Y: y, Z: z}, outSize)
			if randomWeights == false {
				var weights Tensor
				weights = NewTensor(x*y*z, outSize, 1)
				weights.SetData3D(data.Network.Layers[i].Weights[0].Data)
				fullyconnected.SetCustomWeights(&[]Tensor{weights})
			}
			(*wh).Layers = append((*wh).Layers, fullyconnected)
			break
		default:
			err = errors.New("Unrecognized layer type: " + data.Network.Layers[i].LayerType)
			return err
		}
	}

	(*wh).LP.LearningRate = data.Parameters.LearningRate
	(*wh).LP.Momentum = data.Parameters.Momentum
	(*wh).LP.WeightDecay = data.Parameters.WeightDecay
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
	TDSize TDsize        `json:"TDSize"`
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
	InputSize  TDsize          `json:"InputSize"`
	Parameters LayerParamsJSON `json:"Parameters,omitempty"`
	Weights    []TensorJSON    `json:"Weights,omitempty"`
	// Actually "OutputSize" parameter is usefull for fully connected layer only
	// There are automatic calculation of output size for other layers' types
	OutputSize TDsize `json:"OutputSize,omitempty"`
}

// NetworkJSON ...
type NetworkJSON struct {
	Layers []NetLayerJSON `json:"Layers"`
}
