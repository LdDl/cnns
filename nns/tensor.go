package nns

import (
	"errors"
	"fmt"
	"math"
)

// Tensor - Structure for storing data of float64.
/*
	Data - one-dimensional array of float64;
	Size - tensor's data size (see "TDsize" structure).
*/
type Tensor struct {
	Data []float64
	Size TDsize
}

// NewTensor - Constructor for Tensor type.
/*
	x - number of columns (width);
	y - number of rows (height);
	z - depth.
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

// NewTensorCopy - Constructor for Tensor type.
/*
	t - *Tensor which you want to copy.
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

// Get - Return [i][j][k]-th element.
/*
	x - row;
	y - col;
	z - depth.
*/
func (t1 *Tensor) Get(x, y, z int) float64 {
	// return (*t1).Data[x+(y+z*(*t1).Size.Z)*(*t1).Size.X]
	return (*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x]
}

// GetPtr - Return [i][j][k]-th element (as pointer)
/*
	x - row;
	y - col;
	z - depth.
*/
func (t1 *Tensor) GetPtr(x, y, z int) *float64 {
	return &t1.Data[z*t1.Size.X*t1.Size.Y+y*t1.Size.X+x]
}

// Set - Set [i][j][k]-th element with value.
/*
	x - row;
	y - col;
	z - depth;
	value - value of float64.
*/
func (t1 *Tensor) Set(x, y, z int, val float64) {
	// (*t1).Data[x+(y+z*(*t1).Size.Z)*(*t1).Size.X] = val
	(*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x] = val
}

// SetAdd - Add value to [i][j][k]-th element
/*
	x - row;
	y - col;
	z - depth;
	value - value of float64.
*/
func (t1 *Tensor) SetAdd(x, y, z int, val float64) {
	// (*t1).Data[x+(y+z*(*t1).Size.Z)*(*t1).Size.X] += val
	(*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x] += val
}

// SetData3D - Set data for *Tensor (as 3-d array)
/*
	data - 3-D array of float64.
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

// SetData - Set data for *Tensor
/*
	r - number of rows;
	c - number of columns (width);
	d - depth;
	data - 1-D array of float64.
*/
func (t1 *Tensor) SetData(c, r, d int, data []float64) {
	for i := 0; i < c; i++ {
		for j := 0; j < r; j++ {
			for k := 0; k < d; k++ {
				(*t1).Set(i, j, k, data[k*c*d+j*c+i])
			}
		}
	}
}

// Print - Pretty print for *Tensor
func (t1 *Tensor) Print() {
	mx := (*t1).Size.X
	my := (*t1).Size.Y
	mz := (*t1).Size.Z
	for z := 0; z < mz; z++ {
		fmt.Printf("Dim: %v\n", z)
		for y := 0; y < my; y++ {
			for x := 0; x < mx; x++ {
				fmt.Printf("%.10f\t", (*t1).Get(x, y, z))
			}
			fmt.Println()
		}
	}
}

// GetData3D - Return *Tensor as 3-D array
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

// Rot2D90 Rotate tensor (2d component) by 90 degrees
func (t1 *Tensor) Rot2D90(times ...int) Tensor {
	ret := NewTensor(t1.Size.Y, t1.Size.X, t1.Size.Z)
	for z := 0; z < ret.Size.Z; z++ {
		for y := 0; y < ret.Size.Y; y++ {
			for x := 0; x < ret.Size.X; x++ {
				ret.Set(x, y, z, t1.Get(t1.Size.X-y-1, x, z))
			}
		}
	}
	if len(times) != 0 && times[0] > 1 {
		for t := 0; t < times[0]-1; t++ {
			ret = ret.Rot2D90()
		}
	}
	return ret
}

// Rot2D180 Rotate tensor (2d component) by 180 degrees
func (t1 *Tensor) Rot2D180() Tensor {
	return t1.Rot2D90(2)
}

// Rot2D270 Rotate tensor (2d component) by 270 degrees
func (t1 *Tensor) Rot2D270() Tensor {
	return t1.Rot2D90(3)
}

func (t1 *Tensor) Conv2D(kernel *Tensor) (*Tensor, error) {
	if kernel.Size.X%2 != 1 || kernel.Size.Y%2 != 1 {
		return nil, errors.New("Kernel size has to be odd")
	}
	if kernel.Size.X != kernel.Size.Y {
		return nil, errors.New("W and H have to be same")
	}

	pad := math.Floor(float64(kernel.Size.X) / 2)
	out := NewTensor(t1.Size.X+int(pad), t1.Size.Y, t1.Size.Z)

	t1.Print()
	out.Print()

	for i := 0; i < t1.Size.Y; i++ {
		for j := 0; j < t1.Size.X; j++ {
			for ki := 0; ki < kernel.Size.Y; ki++ {
				for kj := 0; kj < kernel.Size.X; kj++ {
					//  @TODO
					// out.SetAdd(i, j, 0, out.Get(i+ki, j+kj, 0)*kernel.Get(ki, kj, 0))
				}
			}
		}
	}
	return &out, nil
}
