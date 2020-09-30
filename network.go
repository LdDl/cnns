package cnns

import (
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// WholeNet Neural net itself (slice of layers)
type WholeNet struct {
	Layers []Layer
	LP     LearningParams
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
		wh.Layers[i].UpdateWeights()
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
