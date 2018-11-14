package nns

import "errors"

// LearningParams - Parameters for training neural network.
/*
	LearningRate
	Momentum
	WeightDecay
*/
type LearningParams struct {
	LearningRate float64 `json:"LearningRate"`
	Momentum     float64 `json:"Momentum"`
	WeightDecay  float64 `json:"WeightDecay"`
}

var (
	lp = LearningParams{
		LearningRate: 0.01,
		Momentum:     0.6,
		WeightDecay:  0.001,
	}
)

// SetEta Set learning rate
func SetEta(v float64) error {
	if v <= 0 {
		return errors.New("η (learning rate) can not be less or equal zero. Setting default value which is 0.01")
	}
	lp.LearningRate = v
	return nil
}

// SetMomentum Set momentum
func SetMomentum(v float64) error {
	if v <= 0 {
		return errors.New("α (momentum) can not be less or equal zero. Setting default value which is 0.6")
	}
	lp.Momentum = v
	return nil
}

// UpdateWeight - Update weights with new value.
func UpdateWeight(w float64, grad *Gradient, multp float64) float64 {
	m := (*grad).Grad + (*grad).OldGrad*lp.Momentum
	w -= lp.LearningRate*m*multp + lp.LearningRate*lp.WeightDecay*w
	return w
}

// UpdateGradient - Update gradient with new values (based on gradient from previous training iteration).
func UpdateGradient(grad *Gradient) {
	(*grad).OldGrad = (*grad).Grad + (*grad).OldGrad*lp.Momentum
}
