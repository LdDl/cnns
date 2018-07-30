package nns

const (
	// LearningRate - learning rate
	LearningRate = 0.01
	// Momentum - momentum
	Momentum = 0.6
	// WeightDecay - decay for weights
	WeightDecay = 0.001
)

// UpdateWeight - updates weights with new value
func UpdateWeight(w float64, grad *Gradient, multp float64) float64 {
	m := (*grad).Grad + (*grad).OldGrad*Momentum
	w -= LearningRate*m*multp + LearningRate*WeightDecay*w
	return w
}

// UpdateGradient - updates gradient with new values (based on gradient on previous training iteration)
func UpdateGradient(grad *Gradient) {
	(*grad).OldGrad = (*grad).Grad + (*grad).OldGrad*Momentum
}
