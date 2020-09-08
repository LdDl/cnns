package tensor

import (
	"testing"

	"github.com/LdDl/cnns/utils/u"
)

/* Test coverage for math stuff */

func TestAddWise(t *testing.T) {
	tensor1 := NewTensor(2, 2, 1)
	tensor1.SetData(2, 2, 1, []float64{1, 2, 3, 4})
	tensor2 := NewTensor(2, 2, 1)
	tensor2.SetData(2, 2, 1, []float64{5, 6, 7, 8})

	tensorCorrect := NewTensor(2, 2, 1)
	tensorCorrect.SetData(2, 2, 1, []float64{6, 8, 10, 12})

	added, err := tensor1.Add(tensor2)
	if err != nil {
		t.Error(err)
		return
	}
	for i := range added.Data {
		if added.Data[i] != tensorCorrect.Data[i] {
			t.Errorf("Tensors are not equal at pos #%d. Expected value: %f. Got: %f", i, tensorCorrect.Data[i], added.Data[i])
		}
	}
}

func TestSubWise(t *testing.T) {
	tensor1 := NewTensor(2, 2, 1)
	tensor1.SetData(2, 2, 1, []float64{1, 2, 3, 4})
	tensor2 := NewTensor(2, 2, 1)
	tensor2.SetData(2, 2, 1, []float64{5, 6, 7, 8})

	tensorCorrect := NewTensor(2, 2, 1)
	tensorCorrect.SetData(2, 2, 1, []float64{-4, -4, -4, -4})

	added, err := tensor1.Sub(tensor2)
	if err != nil {
		t.Error(err)
		return
	}
	for i := range added.Data {
		if added.Data[i] != tensorCorrect.Data[i] {
			t.Errorf("Tensors are not equal at pos #%d. Expected value: %f. Got: %f", i, tensorCorrect.Data[i], added.Data[i])
		}
	}
}

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

func TestMultiply(t *testing.T) {
	tensor1 := NewTensor(2, 2, 1)
	tensor1.SetData(2, 2, 1, []float64{1, 2, 3, 4})
	tensor2 := NewTensor(2, 2, 1)
	tensor2.SetData(2, 2, 1, []float64{5, 6, 7, 8})

	tensorCorrect := NewTensor(2, 2, 1)
	tensorCorrect.SetData(2, 2, 1, []float64{19, 22, 43, 50})

	tensor3, err := tensor1.Multiply(tensor2)
	if err != nil {
		t.Error(err)
		return
	}

	for i := range tensor3.Data {
		if tensor3.Data[i] != tensorCorrect.Data[i] {
			t.Errorf("Tensors are not equal at pos #%d. Expected value: %f. Got: %f", i, tensorCorrect.Data[i], tensor3.Data[i])
		}
	}

	tensor1 = NewTensor(2, 2, 2)
	tensor1.SetData(2, 2, 2, []float64{5, 6, 7, 8, 1, 2, 3, 4})
	tensor2 = NewTensor(2, 2, 2)
	tensor2.SetData(2, 2, 2, []float64{1, 2, 3, 4, 9, 8, 7, 6})
	tensorCorrect = NewTensor(2, 2, 2)
	tensorCorrect.SetData(2, 2, 2, []float64{23, 34, 31, 46, 23, 20, 55, 48})
	tensor3, err = tensor1.Multiply(tensor2)
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

	tensor3, err = tensor1.Multiply(tensor2)
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

func TestConvolve2D(t *testing.T) {
	tensor1 := NewTensor(8, 9, 1)
	tensor1.SetData(8, 9, 1, []float64{
		-0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
		-0.9, -0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
		-0.17, 0.18, -0.19, 0.20, 0.21, 0.22, 0.23, 0.24,
		-0.25, 0.26, 0.27, -0.28, 0.29, 0.30, 0.31, 0.32,
		-0.33, 0.34, 0.35, 0.36, -0.37, 0.38, 0.39, 0.40,
		-0.41, 0.42, 0.43, 0.44, 0.45, -0.46, 0.47, 0.48,
		-0.49, 0.50, 0.51, 0.52, 0.53, 0.54, -0.55, 0.56,
		-0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, -0.64,
		-0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72,
	})

	kernel := NewTensor(3, 3, 1)
	kernel.SetData(3, 3, 1, []float64{
		0.10466029, -0.06228581, -0.43436298,
		0.44050909, -0.07536250, -0.34348075,
		0.16456005, 0.18682307, -0.40303048,
	})
	stride := 1

	tensorCorrect := NewTensor(6, 7, 1)
	tensorCorrect.SetData(6, 7, 1, []float64{
		-0.4977081632, -0.3515390213, -0.2944759209, -0.2714417671, -0.3109404323, -0.3504390975,
		-0.2603203045, 0.1615497118, -0.3480871226, -0.1724460774, -0.0845121496, -0.0887318498,
		-0.3007816946, 0.1104071772, 0.1908442942, -0.4989842132, -0.2400441882, -0.1224894514,
		-0.6131539374, 0.1038750166, 0.1454678562, 0.2201388766, -0.6498813038, -0.3076422990,
		-0.7604682478, -0.1731258538, 0.1440830512, 0.1805285352, 0.2494334590, -0.8007783944,
		-0.9077825582, -0.2068834554, -0.2111031556, 0.1842910858, 0.2155892142, 0.2787280414,
		-1.0550968686, -0.2406410570, -0.2448607572, -0.2490804574, 0.2244991204, 0.2506498932,
	})

	tensor3, err := tensor1.Convolve2D(kernel, stride)
	if err != nil {
		t.Error(err)
		return
	}

	for i := range tensor3.Data {
		if u.RoundPlaces(tensor3.Data[i], 12) != u.RoundPlaces(tensorCorrect.Data[i], 12) {
			t.Errorf("Tensors are not equal at pos #%d. Expected value: %f. Got: %f", i, tensorCorrect.Data[i], tensor3.Data[i])
		}
	}

}
