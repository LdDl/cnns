package tensor

import "math"

/*
	Simple math operations for Tensor type
*/

// Add - Element-wise summation.
func (t1 *Tensor) Add(t2 *Tensor) Tensor {
	var ret = NewTensor(t1.Size.X, t1.Size.Y, t1.Size.Z)
	for i := 0; i < (*t2).Size.X*(*t2).Size.Y*(*t2).Size.Z; i++ {
		ret.Data[i] = t1.Data[i] + t2.Data[i]
	}
	return ret
}

// Sub - Element-wise substraction.
func (t1 *Tensor) Sub(t2 *Tensor) Tensor {
	var ret = NewTensor(t1.Size.X, t1.Size.Y, t1.Size.Z)
	for i := 0; i < (*t2).Size.X*(*t2).Size.Y*(*t2).Size.Z; i++ {
		ret.Data[i] = t1.Data[i] - t2.Data[i]
	}
	return ret
}

// MSE - Mean square error
func (t1 *Tensor) MSE(t2 *Tensor) float64 {
	sum := 0.0
	var ret = NewTensorCopy(t1)
	num := (*t2).Size.X * (*t2).Size.Y * (*t2).Size.Z
	for i := 0; i < num; i++ {
		ret.Data[i] = math.Pow((ret.Data[i] - t2.Data[i]), 2.0)
		sum += ret.Data[i]
	}
	return sum / float64(num)
}
