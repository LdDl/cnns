package nns

import "math"

// ActivationTanh is hyperbolic tangent
/*
	See the reference: http://www.wolframalpha.com/input/?i=tanh(x)
*/
func ActivationTanh(v float64) float64 {
	return math.Tanh(v)
}

// ActivationTanhDerivative is derivative of hyperbolic tangent
/*
	See the reference: http://www.wolframalpha.com/input/?i=(tanh(x))%27
*/
func ActivationTanhDerivative(v float64) float64 {
	return 1 - ActivationTanh(v)*ActivationTanh(v)
}

// ActivationSygmoid is sigmoid function
/*
	See the reference: http://www.wolframalpha.com/input/?i=1%2F(1%2B+exp(-x))
*/
func ActivationSygmoid(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1.0*v))
}

// ActivationSygmoidDerivative is derivative of sigmoid tangent
/*
	See the reference: http://www.wolframalpha.com/input/?i=(1%2F(1%2B+exp(-x)))%27
*/
func ActivationSygmoidDerivative(v float64) float64 {
	return ActivationSygmoid(v) * (1 - ActivationSygmoid(v))
}
