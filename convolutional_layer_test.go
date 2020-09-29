package cnns

import (
	"testing"

	"github.com/LdDl/cnns/tensor"
)

func TestConvSize(t *testing.T) {
	conv := NewConvLayer(tensor.TDsize{X: 9, Y: 8, Z: 1}, 1, 3, 1)
	correct := tensor.TDsize{X: 7, Y: 6, Z: 1}
	outSize := conv.GetOutputSize()
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
