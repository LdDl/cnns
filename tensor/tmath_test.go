package tensor

import (
	"testing"
)

/*
	Test coverage for math stuff
*/

func TestTranspose(t *testing.T) {
	tensor := NewTensor(2, 3, 1)
	tensor.SetData(2, 3, 1, []float64{1, 2, 3, 4, 5, 6})

	tensorCorrect := NewTensor(3, 2, 1)
	tensorCorrect.SetData(3, 2, 1, []float64{1, 3, 5, 2, 4, 6})

	transponsed := tensor.Transpose()

	for i := range transponsed.Data {
		if transponsed.Data[i] != tensorCorrect.Data[i] {
			t.Error("Tensors are not equal")
		}
	}
}
