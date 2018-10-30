package nns

import "math"

// ActivationTanh - hyperbolic tangent
func ActivationTanh(v float64) float64 {
	return math.Tanh(v)
}

// ActivationTanhDerivative - derivative of hyperbolic tangent
func ActivationTanhDerivative(v float64) float64 {
	return 1 - ActivationTanh(v)*ActivationTanh(v)
}
