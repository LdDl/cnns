package nns

// LearningParams ...
type LearningParams struct {
	LearningRate float64 `json:"LearningRate"` // LearningRate - learning rate
	Momentum     float64 `json:"Momentum"`     // Momentum - momentum
	WeightDecay  float64 `json:"WeightDecay"`  // WeightDecay - decay for weights
}

var (
	lp = LearningParams{
		LearningRate: 0.01,
		Momentum:     0.6,
		WeightDecay:  0.001,
	}
)

// UpdateWeight - updates weights with new value
func UpdateWeight(w float64, grad *Gradient, multp float64) float64 {
	m := (*grad).Grad + (*grad).OldGrad*lp.Momentum
	w -= lp.LearningRate*m*multp + lp.LearningRate*lp.WeightDecay*w
	return w
}

// UpdateGradient - updates gradient with new values (based on gradient on previous training iteration)
func UpdateGradient(grad *Gradient) {
	(*grad).OldGrad = (*grad).Grad + (*grad).OldGrad*lp.Momentum
}



