package cnns

import (
	"encoding/json"
	"errors"
	"io/ioutil"

	"github.com/LdDl/cnns/tensor"
	"gonum.org/v1/gonum/mat"
)

// NetJSON JSON representation of network structure (for import and export)
type NetJSON struct {
	Network    *NetworkJSON    `json:"network"`
	Parameters *LearningParams `json:"parameters"`
}

// NetworkJSON JSON representation of networks' layers
type NetworkJSON struct {
	Layers []*NetLayerJSON `json:"layers"`
}

// NetLayerJSON JSON representation of layer
type NetLayerJSON struct {
	LayerType  string           `json:"layer_type"`
	InputSize  *tensor.TDsize   `json:"input_size"`
	Parameters *LayerParamsJSON `json:"parameters"`
	Weights    []*TensorJSON    `json:"weights"`
	// Actually "OutputSize" parameter is useful for fully-connected layer only
	// There are automatic calculation of output size for other layers' types
	OutputSize *tensor.TDsize `json:"output_size"`
}

// LayerParamsJSON JSON representation of layers attributes
type LayerParamsJSON struct {
	Stride     int `json:"stride"`
	KernelSize int `json:"kernel_size"`
}

// TensorJSON JSON representation of tensor
type TensorJSON struct {
	TDSize *tensor.TDsize `json:"data_size"`
	Data   []float64      `json:"data"`
}

// ImportFromFile Load network to file
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
			break
		case "relu":
			x := data.Network.Layers[i].InputSize.X
			y := data.Network.Layers[i].InputSize.Y
			z := data.Network.Layers[i].InputSize.Z
			relu := NewReLULayer(&tensor.TDsize{X: x, Y: y, Z: z})
			wh.Layers = append(wh.Layers, relu)
			break
		case "pool":
			break
		case "fc":
			x := data.Network.Layers[i].InputSize.X
			y := data.Network.Layers[i].InputSize.Y
			z := data.Network.Layers[i].InputSize.Z
			outSize := data.Network.Layers[i].OutputSize.X
			fullyconnected := NewFullyConnectedLayer(&tensor.TDsize{X: x, Y: y, Z: z}, outSize)
			if randomWeights == false {
				weights := mat.NewDense(outSize, x*y, data.Network.Layers[i].Weights[0].Data)
				fullyconnected.SetCustomWeights([]*mat.Dense{weights})
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
