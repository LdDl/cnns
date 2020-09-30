package cnns

import (
	"fmt"
)

// LearningParams - Parameters for training neural network.
/*
	LearningRate - η
	Momentum - α
*/
type LearningParams struct {
	LearningRate float64 `json:"learning_rate"`
	Momentum     float64 `json:"momentum"`
}

// NewLearningParametersDefault Constructor for LearningParams
func NewLearningParametersDefault() *LearningParams {
	return &LearningParams{
		LearningRate: 0.01,
		Momentum:     0.6,
	}
}

// SetEta Set learning rate
func (lp *LearningParams) SetEta(v float64) error {
	if v <= 0 {
		return fmt.Errorf("η (learning rate) can not be less or equal zero. Setting default value which is 0.01")
	}
	lp.LearningRate = v
	return nil
}

// SetMomentum Set momentum
func (lp *LearningParams) SetMomentum(v float64) error {
	if v <= 0 {
		return fmt.Errorf("α (momentum) can not be less or equal zero. Setting default value which is 0.6")
	}
	lp.Momentum = v
	return nil
}

// SetL2Decay Set weight's decay
func (lp *LearningParams) SetL2Decay(v float64) error {
	if v <= 0 {
		return fmt.Errorf("λ (momentum) can not be less or equal zero. Setting default value which is 0.005")
	}
	lp.Momentum = v
	return nil
}
