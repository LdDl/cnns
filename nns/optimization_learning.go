package nns

const (
	LearningRate = 0.01
	Momentum     = 0.6
	WeightDecay  = 0.001
)

func UpdateWeight(w float64, grad *Gradient, multp float64) float64 {
	m := (*grad).Grad + (*grad).OldGrad*Momentum
	w -= LearningRate*m*multp + LearningRate*WeightDecay*w
	return w
}

func UpdateGradient(grad *Gradient) {
	(*grad).OldGrad = (*grad).Grad + (*grad).OldGrad*Momentum
}
