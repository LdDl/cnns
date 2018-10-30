package nns

import (
	"fmt"
)

// Tensor is structure storing date
/*
	Data - one-dimensional array
	TDsize: tensor data size (see "Size" structure)
*/
type Tensor struct {
	Data []float64
	Size TDsize
}

// NewTensor is constructor for Tensor type.
/*
	x - number of columns (width)
	y - number of rows (height)
	z - depth
*/
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

// NewTensorCopy is constructor for Tensor type.
/*
	t - *Tensor which you want to copy
*/
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

// Add - Element-wise summation
func (t1 *Tensor) Add(t2 *Tensor) Tensor {
	var ret = NewTensorCopy(t1)
	for i := 0; i < (*t2).Size.X*(*t2).Size.Y*(*t2).Size.Z; i++ {
		ret.Data[i] += t2.Data[i]
	}
	return ret
}

// Sub - Element-wise substraction
func (t1 *Tensor) Sub(t2 *Tensor) Tensor {
	var ret = NewTensorCopy(t1)
	for i := 0; i < (*t2).Size.X*(*t2).Size.Y*(*t2).Size.Z; i++ {
		ret.Data[i] -= t2.Data[i]
	}
	return ret
}

// Get - Return [i][j][k]-th element
/*
	x - row
	y - col
	z - depth
*/
func (t1 *Tensor) Get(x, y, z int) float64 {
	// return (*t1).Data[x+(y+z*(*t1).Size.Z)*(*t1).Size.X]
	return (*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x]
}

// Set - Set [i][j][k]-th element with value
/*
	x - row
	y - col
	z - depth
	value - value of float64
*/
func (t1 *Tensor) Set(x, y, z int, val float64) {
	// (*t1).Data[x+(y+z*(*t1).Size.Z)*(*t1).Size.X] = val
	(*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x] = val
}

// SetAdd - Add value to [i][j][k]-th element
/*
	x - row
	y - col
	z - depth
	value - value of float64
*/
func (t1 *Tensor) SetAdd(x, y, z int, val float64) {
	// (*t1).Data[x+(y+z*(*t1).Size.Z)*(*t1).Size.X] += val
	(*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x] += val
}

// SetData3D - Set data for Tensor (as 3-d array)
/*
	data - 3-D array of float64
*/
func (t1 *Tensor) SetData3D(data [][][]float64) {
	z := len(data)       // depth
	y := len(data[0])    // height (number of rows)
	x := len(data[0][0]) // width (number of columns)
	for i := 0; i < x; i++ {
		for j := 0; j < y; j++ {
			for k := 0; k < z; k++ {
				(*t1).Set(i, j, k, data[k][j][i])
			}
		}
	}
}

// SetData - Set data for Tensor
/*
	r - number of rows
	c - number of columns
	d - depth
	data - 1-D array of float64
*/
func (t1 *Tensor) SetData(r, c, d int, data []float64) {
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			for k := 0; k < d; k++ {
				(*t1).Set(i, j, k, data[k*r*d+j*r+i])
			}
		}
	}
}

// Print - Pretty print *Tensor
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
