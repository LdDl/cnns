package nns

import (
	"fmt"
)

// Tensor is structure storing Data: one-dimensional array; and TDsize: tensor data size (see "Size" structure)
type Tensor struct {
	Data []float64
	Size TDsize
}

// NewTensor - constructor for Tensor type. You need to provide dimensions: x, y and z.
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

// NewTensorCopy - constructor for Tensor type. You need to provide another tensor for cloning it into new one.
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

// Add - element-wise summation
func (t1 *Tensor) Add(t2 *Tensor) Tensor {
	var ret = NewTensorCopy(t1)
	for i := 0; i < (*t2).Size.X*(*t2).Size.Y*(*t2).Size.Z; i++ {
		ret.Data[i] += t2.Data[i]
	}
	return ret
}

// Sub - element-wise substraction
func (t1 *Tensor) Sub(t2 *Tensor) Tensor {
	var ret = NewTensorCopy(t1)
	for i := 0; i < (*t2).Size.X*(*t2).Size.Y*(*t2).Size.Z; i++ {
		ret.Data[i] -= t2.Data[i]
	}
	return ret
}

// Get - gets [i][j][k]-th element
func (t1 *Tensor) Get(x, y, z int) float64 {
	// return (*t1).Data[x+(y+z*(*t1).Size.Z)*(*t1).Size.X]
	return (*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x]
}

// Set - sets [i][j][k]-th element with value
func (t1 *Tensor) Set(x, y, z int, val float64) {
	// (*t1).Data[x+(y+z*(*t1).Size.Z)*(*t1).Size.X] = val
	(*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x] = val
}

// SetAdd - adds value to [i][j][k]-th element
func (t1 *Tensor) SetAdd(x, y, z int, val float64) {
	// (*t1).Data[x+(y+z*(*t1).Size.Z)*(*t1).Size.X] += val
	(*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x] += val
}

// SetData - sets data for Tensor (as 3-d array)
func (t1 *Tensor) SetData(data [][][]float64) {
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

// Print - prints Tensor (cube as set of matrices)
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

// GetData3D returns Tensor as 3-dimensional array
func (t1 *Tensor) GetData3D() [][][]float64 {
	mx := (*t1).Size.X
	my := (*t1).Size.Y
	mz := (*t1).Size.Z
	ret := make([][][]float64, mz)
	for z := 0; z < mz; z++ {
		ret[z] = make([][]float64, my)
		for y := 0; y < my; y++ {
			ret[z][y] = make([]float64, mx)
			for x := 0; x < mx; x++ {
				ret[z][y][x] = (*t1).Get(x, y, z)
			}
		}
	}
	return ret
}
