package cnns

import (
	"encoding/json"
	"fmt"
	"io/ioutil"

	"github.com/pkg/errors"
)

// ExportToFile Save network structure and its weights to JSON file
func (wh *WholeNet) ExportToFile(fname string, saveWeights bool) error {
	save := NetJSON{}
	for i := 0; i < len(wh.Layers); i++ {
		switch wh.Layers[i].GetType() {
		case "conv":
			layer := wh.Layers[i].(*ConvLayer)
			kernels := wh.Layers[i].GetWeights()
			newLayer := &NetLayerJSON{
				LayerType: "conv",
				InputSize: wh.Layers[i].GetInputSize(),
				Parameters: &LayerParamsJSON{
					Stride:     wh.Layers[i].GetStride(),
					KernelSize: layer.KernelSize,
				},
				Weights: make([]*TensorJSON, len(kernels)),
			}
			if saveWeights {
				for k := range kernels {
					newLayer.Weights[k].Data = kernels[k].RawMatrix().Data
				}
			}
			save.Network.Layers = append(save.Network.Layers, newLayer)
			break
		case "relu":
			newLayer := &NetLayerJSON{
				LayerType: "relu",
				InputSize: wh.Layers[i].GetInputSize(),
			}
			save.Network.Layers = append(save.Network.Layers, newLayer)
			break
		case "pool":
			layer := wh.Layers[i].(*PoolingLayer)
			newLayer := &NetLayerJSON{
				LayerType: "pool",
				InputSize: wh.Layers[i].GetInputSize(),
				Parameters: &LayerParamsJSON{
					Stride:          wh.Layers[i].GetStride(),
					KernelSize:      layer.ExtendFilter,
					PoolingType:     layer.PoolingType.String(),
					ZeroPaddingType: layer.ZeroPadding.String(),
				},
			}
			save.Network.Layers = append(save.Network.Layers, newLayer)
			break
		case "fc":
			newLayer := &NetLayerJSON{
				LayerType:  "fc",
				InputSize:  wh.Layers[i].GetInputSize(),
				OutputSize: wh.Layers[i].GetOutputSize(),
				Weights:    make([]*TensorJSON, 1),
			}
			if saveWeights {
				weights := wh.Layers[i].GetWeights()
				if len(weights) != 1 {
					return fmt.Errorf("Fully connected layer can have only 1 array for weights")
				}
				newLayer.Weights[0].Data = weights[0].RawMatrix().Data
			}
			save.Network.Layers = append(save.Network.Layers, newLayer)
			break
		default:
			return fmt.Errorf("Unrecognized layer type: %v", wh.Layers[i].GetType())
		}
	}

	save.Parameters.LearningRate = lp.LearningRate
	save.Parameters.Momentum = lp.Momentum
	save.Parameters.WeightDecay = lp.WeightDecay

	saveJSON, err := json.Marshal(save)
	if err != nil {
		return errors.Wrap(err, "Can't marshal network to JSON")
	}

	err = ioutil.WriteFile(fname, saveJSON, 0644)
	if err != nil {
		return errors.Wrap(err, fmt.Sprintf("Can't write data to file '%s'", fname))
	}

	return err
}
