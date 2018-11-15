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

// ActivationSygmoidDerivative is derivative of sigmoid
/*
	See the reference: http://www.wolframalpha.com/input/?i=(1%2F(1%2B+exp(-x)))%27
*/
func ActivationSygmoidDerivative(v float64) float64 {
	return ActivationSygmoid(v) * (1 - ActivationSygmoid(v))
}

// ActivationArcTan is arctan function
/*
	See the reference: http://www.wolframalpha.com/input/?i=atan(x)
*/
func ActivationArcTan(v float64) float64 {
	return math.Atan(v)
}

// ActivationArcTanDerivative is derivative of arctan
/*
	See the reference: http://www.wolframalpha.com/input/?i=(atan(x))%27
*/
func ActivationArcTanDerivative(v float64) float64 {
	return 1.0 / (v*v + 1)
}

// ActivationSoftPlus is logarithmic function (ln(1+exp(x))). This is integrated sigmoid actually.
/*
	See the reference: http://www.wolframalpha.com/input/?i=ln(1%2Bexp(x))
					or http://www.wolframalpha.com/input/?i=integrate(1%2F(1%2B+exp(-x)))
*/
func ActivationSoftPlus(v float64) float64 {
	return math.Log(1 + math.Exp(v))
}

// ActivationSoftPlusDerivative is derivative of logarithm ln(1+exp(x)). This is sigmoid actually.
/*
	See the reference: http://www.wolframalpha.com/input/?i=(ln(1%2Bexp(x)))%27
*/
func ActivationSoftPlusDerivative(v float64) float64 {
	return ActivationSygmoid(v)
}

// ActivationGaussian is gaussian function
/*
	See the reference: http://www.wolframalpha.com/input/?i=exp(-(x%5E2))
*/
func ActivationGaussian(v float64) float64 {
	return math.Exp(-1.0 * v * v)
}

// ActivationGaussianDerivative is derivative of gaussian
/*
	See the reference: http://www.wolframalpha.com/input/?i=(exp(-(x%5E2)))%27
*/
func ActivationGaussianDerivative(v float64) float64 {
	return -2.0 * v * math.Exp(-1.0*v*v)
}
