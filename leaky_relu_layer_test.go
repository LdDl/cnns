package cnns

import (
	"testing"

	"github.com/LdDl/cnns/tensor"
)

func TestLeakyReLUSize(t *testing.T) {
	lrelu := NewLeakyReLULayer(&tensor.TDsize{X: 6, Y: 7, Z: 1}, 0.001)
	correct := tensor.TDsize{X: 6, Y: 7, Z: 1}
	outSize := lrelu.GetOutputSize()
	if outSize.X != correct.X {
		t.Errorf("X dimension should be of value %d, but got %d", correct.X, outSize.X)
	}
	if outSize.Y != correct.Y {
		t.Errorf("Y dimension should be of value %d, but got %d", correct.Y, outSize.Y)
	}
	if outSize.Z != correct.Z {
		t.Errorf("Z dimension should be of value %d, but got %d", correct.Z, outSize.Z)
	}
}
