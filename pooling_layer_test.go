package cnns

import (
	"testing"

	"github.com/LdDl/cnns/tensor"
)

func TestPoolSize(t *testing.T) {
	pool := NewMaxPoolingLayer(2, 2, &tensor.TDsize{X: 8, Y: 9, Z: 1})
	correct := tensor.TDsize{X: 4, Y: 4, Z: 1}
	outSize := pool.GetOutputSize()
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
