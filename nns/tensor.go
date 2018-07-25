package nns

import (
	"fmt"
)

// Tensor is structure storing Data: one-dimensional array; and TDsize: tensor data size (see "Size" structure)
type Tensor struct {
	Data []float64
	Size TDsize
}

func NewTensor(x, y, z int) Tensor {
	return Tensor{
		Data: make([]float64, x*y*z),
		Size: TDsize{
			X: x,
			Y: y,
			Z: z,
		},
	}
}

func NewTensorCopy(t *Tensor) Tensor {
	return Tensor{
		Data: (*t).Data,
		Size: TDsize{
			X: (*t).Size.X,
			Y: (*t).Size.Y,
			Z: (*t).Size.Z,
		},
	}
}

func (t1 *Tensor) Add(t2 *Tensor) Tensor {
	var ret = NewTensorCopy(t1)
	for i := 0; i < (*t2).Size.X*(*t2).Size.Y*(*t2).Size.Z; i++ {
		ret.Data[i] += t2.Data[i]
	}
	return ret
}

func (t1 *Tensor) Sub(t2 *Tensor) Tensor {
	var ret = NewTensorCopy(t1)
	for i := 0; i < (*t2).Size.X*(*t2).Size.Y*(*t2).Size.Z; i++ {
		ret.Data[i] -= t2.Data[i]
	}
	return ret
}

func (t1 *Tensor) Get(x, y, z int) float64 {
	return (*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x]
}

func (t1 *Tensor) Set(x, y, z int, val float64) {
	(*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x] = val
}

func (t1 *Tensor) SetAdd(x, y, z int, val float64) {
	(*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x] += val
}

func (t1 *Tensor) CopyFrom(data [][][]float64) {
	z := len(data)
	y := len(data[0])
	x := len(data[0][0])
	for i := 0; i < x; i++ {
		for j := 0; j < y; j++ {
			for k := 0; k < z; k++ {
				(*t1).Set(i, j, k, data[k][j][i])
			}
		}
	}
}

func (t1 *Tensor) Print() {
	mx := (*t1).Size.X
	my := (*t1).Size.Y
	mz := (*t1).Size.Z
	for z := 0; z < mz; z++ {
		fmt.Printf("Dim: %v\n", z)
		for y := 0; y < my; y++ {
			for x := 0; x < mx; x++ {
				fmt.Printf("%.8f\t", (*t1).Get(x, y, z))
			}
			fmt.Println()
		}
	}
}
