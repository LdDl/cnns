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
			t.Errorf("Tensors are not equal at pos #%d. Expected value: %f. Got: %f", i, tensorCorrect.Data[i], transponsed.Data[i])
		}
	}
}

func TestProduct(t *testing.T) {
	tensor1 := NewTensor(2, 2, 1)
	tensor1.SetData(2, 2, 1, []float64{1, 2, 3, 4})
	tensor2 := NewTensor(2, 2, 1)
	tensor2.SetData(2, 2, 1, []float64{5, 6, 7, 8})

	tensorCorrect := NewTensor(2, 2, 1)
	tensorCorrect.SetData(2, 2, 1, []float64{19, 22, 43, 50})

	tensor3, err := tensor1.Product(tensor2)
	if err != nil {
		t.Error(err)
		return
	}

	for i := range tensor3.Data {
		if tensor3.Data[i] != tensorCorrect.Data[i] {
			t.Errorf("Tensors are not equal at pos #%d. Expected value: %f. Got: %f", i, tensorCorrect.Data[i], tensor3.Data[i])
		}
	}

	tensor1 = NewTensor(3, 2, 1)
	tensor1.SetData(3, 2, 1, []float64{1, 2, 3, 4, 11, 12})
	tensor2 = NewTensor(2, 2, 1)
	tensor2.SetData(2, 2, 1, []float64{5, 6, 7, 8})

	tensor3, err = tensor1.Product(tensor2)
	if err == nil || err != ErrDimensionsNotFit {
		t.Error(err)
	}
}

func TestHadamardProduct(t *testing.T) {
	tensor1 := NewTensor(2, 3, 1)
	tensor1.SetData(2, 3, 1, []float64{1, 2, 3, 4, 5, 6})
	tensor2 := NewTensor(2, 3, 1)
	tensor2.SetData(2, 3, 1, []float64{7, 8, 9, 10, 11, 12})

	tensorCorrect := NewTensor(2, 3, 1)
	tensorCorrect.SetData(2, 3, 1, []float64{7, 16, 27, 40, 55, 72})

	tensor3, err := HadamardProduct(tensor1, tensor2)
	if err != nil {
		t.Error(err)
		return
	}

	for i := range tensor3.Data {
		if tensor3.Data[i] != tensorCorrect.Data[i] {
			t.Errorf("Tensors are not equal at pos #%d. Expected value: %f. Got: %f", i, tensorCorrect.Data[i], tensor3.Data[i])
		}
	}

	tensor1 = NewTensor(3, 3, 2)
	tensor1.SetData(3, 3, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, -4, -5, -6, -7, -8, -9})
	tensor2 = NewTensor(3, 3, 1)
	tensor1.SetData(3, 3, 1, []float64{10, 11, 12, 13, 14, 15, 16, 18, 19})

	tensor3, err = HadamardProduct(tensor1, tensor2)
	if err == nil || err != ErrDimensionsAreNotEqual {
		t.Error("Error must appear because of tensor1 has shape (3,3,2) and tensor2 has shape (3,3,1)")
	}
}
