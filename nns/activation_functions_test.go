package nns

import (
	"fmt"
	"testing"
)

const (
	testFloat64 = 0.314
)

func TestActivationTanh(t *testing.T) {
	correct := 0.30407166013575465
	got := ActivationTanh(testFloat64)
	if correct != got {
		t.Errorf("Should be %f, but got %f", correct, got)
	}

}

func TestActivationTanhDerivative(t *testing.T) {
	correct := 0.9075404255022861
	got := ActivationTanhDerivative(testFloat64)
	if correct != got {
		t.Errorf("Should be %f, but got %f", correct, got)
	}

}

func TestActivationSygmoid(t *testing.T) {
	correct := 0.5778613142807203
	got := ActivationSygmoid(testFloat64)
	if correct != got {
		t.Errorf("Should be %f, but got %f", correct, got)
	}

}

func TestActivationSygmoidDerivative(t *testing.T) {
	correct := 0.2439376157384789
	got := ActivationSygmoidDerivative(testFloat64)
	if correct != got {
		t.Errorf("Should be %f, but got %f", correct, got)
	}

}

func TestActivationArcTan(t *testing.T) {
	correct := 0.3042508322379845
	got := ActivationArcTan(testFloat64)
	if correct != got {
		t.Errorf("Should be %f, but got %f", correct, got)
	}

}

func TestActivationArcTanDerivative(t *testing.T) {
	correct := 0.9102527225658933
	got := ActivationArcTanDerivative(testFloat64)
	if correct != got {
		t.Errorf("Should be %f, but got %f", correct, got)
	}

}

func TestActivationSoftPlus(t *testing.T) {
	correct := 0.862421379790928
	got := ActivationSoftPlus(testFloat64)
	if correct != got {
		t.Errorf("Should be %f, but got %f", correct, got)
	}

}

func TestActivationSoftPlusDerivative(t *testing.T) {
	correct := 0.5778613142807203
	got := ActivationSoftPlusDerivative(testFloat64)
	if correct != got {
		t.Errorf("Should be %f, but got %f", correct, got)
	}

}

func TestActivationGaussian(t *testing.T) {
	correct := 0.9061087020033959
	got := ActivationGaussian(testFloat64)
	fmt.Println(got)
	if correct != got {
		t.Errorf("Should be %f, but got %f", correct, got)
	}

}

func TestActivationGaussianDerivative(t *testing.T) {
	correct := -0.5690362648581326
	got := ActivationGaussianDerivative(testFloat64)
	if correct != got {
		t.Errorf("Should be %f, but got %f", correct, got)
	}

}
