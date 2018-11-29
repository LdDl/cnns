package nns

import "errors"

// LearningParams - Parameters for training neural network.
/*
	LearningRate - η
	Momentum - α
	WeightDecay - λ (L2 regularization)
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
		WeightDecay:  0.005,
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

// SetL2Decay Set weight's decay
func SetL2Decay(v float64) error {
	if v <= 0 {
		return errors.New("λ (momentum) can not be less or equal zero. Setting default value which is 0.005")
	}
	lp.Momentum = v
	return nil
}
