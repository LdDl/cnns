package nns

import (
	"fmt"
	"math"
)

// Tensor is structure storing Data: one-dimensional array; and TDsize: tensor data size (see "Size" structure)
type Tensor struct {
	Data []float64
	X    int //	width (columns)
	Y    int //	height (rows)
	Z    int //	depth
}

// NewTensorEmpty - constructor for "Tensor" struct, and you could provide x: width, y: height, z: depth
func NewTensorEmpty(x, y, z int) *Tensor {
	return &Tensor{Data: make([]float64, x*y*z), X: x, Y: y, Z: z}
}

// NewTensorEmptyWithBias - constructor for "Tensor" struct, and you could provide x: width, y: height, z: depth. Additionally sets +1 for bias
func NewTensorEmptyWithBias(x, y, z int) *Tensor {
	return &Tensor{Data: make([]float64, x*y*z+1), X: x, Y: y, Z: z}
}

// NewTensorCopy - constructor for "Tensor" struct, and you could provide t: "Tensor"
func NewTensorCopy(t *Tensor) *Tensor {
	var copyT = Tensor{Data: make([]float64, len((*t).Data))}
	copyT.X = (*t).X
	copyT.Y = (*t).Y
	copyT.Z = (*t).Z
	for i := range (*t).Data {
		copyT.Data[i] = (*t).Data[i]
	}
	return &copyT
}

// Set - fill empty "Tensor" with values (should be three-dimensional array [width][height][depth])
func (t1 *Tensor) Set(v *[][][]float64) {
	var x = len((*v)[0][0])
	var y = len((*v)[0])
	var z = len((*v))
	for k := 0; k < z; k++ {
		for j := 0; j < y; j++ {
			for i := 0; i < x; i++ {
				(*t1).SetValue(i, j, k, (*v)[k][j][i])
			}
		}
	}
}

// SetTensor - fill empty "Tensor" with values of another Tensor
func (t1 *Tensor) SetTensor(t2 *Tensor) {
	x := (*t2).X
	y := (*t2).Y
	z := (*t2).Z
	for k := 0; k < z; k++ {
		for j := 0; j < y; j++ {
			for i := 0; i < x; i++ {
				(*t1).SetValue(i, j, k, (*t2).GetValue(i, j, k))
			}
		}
	}
}

// AddValue - add value to [i][j][k]-th element
func (t1 *Tensor) AddValue(i, j, k int, val float64) {
	(*t1).Data[i+(j+k*(*t1).Z)*(*t1).X] += val
}

// SetValue - sets values for [i][j][k]-th element
func (t1 *Tensor) SetValue(i, j, k int, val float64) {
	(*t1).Data[i+(j+k*(*t1).Z)*(*t1).X] = val
}

// GetValue - returns value for [i][j][k]-th element
func (t1 *Tensor) GetValue(i, j, k int) (val float64) {
	// ii+(jj+k*outDepth)*(*t1).X
	return (*t1).Data[i+(j+k*(*t1).Z)*(*t1).X]
}

// GetIndex - returns one-dim index for [i][j][k]-th element
func (t1 *Tensor) GetIndex(i, j, k int) (val int) {
	return (i + (j+k*(*t1).Z)*(*t1).X)
}

// Print - print "Tensor" as set (depth) of matrices (width x height)
func (t1 *Tensor) Print() {
	x := (*t1).X
	y := (*t1).Y
	z := (*t1).Z
	//i + (j+k*depth)*width = [i][j][k]
	for k := 0; k < z; k++ {
		fmt.Printf("  Depth level: %v\n", k)
		for j := 0; j < y; j++ {
			for i := 0; i < x; i++ {
				kernelVal := (*t1).GetValue(i, j, k)
				fmt.Printf("%v ", kernelVal)
			}
			fmt.Println()
		}
	}
}

// Convolve2D - make convolution for t1 via t2
func Convolve2D(t1 *Tensor, kernel *Tensor, strideWidth, strideHeight, paddingWidth, paddingHeight int) (outMatrix *Tensor) {
	inWidth := (*t1).X
	inHeight := (*t1).Y
	inDepth := (*t1).Z
	filterWidth := (*kernel).X
	filterHeight := (*kernel).Y
	outWidth := (inWidth-filterWidth+2*paddingWidth)/strideWidth + 1
	outHeight := (inHeight-filterHeight+2*paddingHeight)/strideHeight + 1
	outDepth := inDepth
	outMatrix = NewTensorEmpty(outWidth, outHeight, outDepth)
	for y := 0; y < outHeight; y++ {
		for x := 0; x < outWidth; x++ {
			mapY := x * strideHeight
			mapX := y * strideWidth
			mapZ := 0 // Чтоб было
			_ = mapZ
			sum := 0.0
			for j := 0; j < filterHeight; j++ {
				for i := 0; i < filterWidth; i++ {
					for k := 0; k < outDepth; k++ {
						ii := i + mapY
						jj := j + mapX
						matrixVal := (*t1).GetValue(ii, jj, k)
						kernelVal := (*kernel).GetValue(i, j, 0)
						// fmt.Printf("%v * %v |", matrixVal, kernelVal)
						sum += matrixVal * kernelVal
					}
				}
				// fmt.Println()
			}
			// fmt.Printf("i,j,k : %v, %v, %v  = %v\n", x, y, 0, sum)
			outMatrix.SetValue(x, y, 0, sum)
		}
	}
	// outMatrix.Print()
	return outMatrix
}

// Rectify - max(0, x) (rectifier as activation func)
func Rectify(t *Tensor, out *Tensor) {
	for k := 0; k < (*t).Z; k++ {
		for j := 0; j < (*t).Y; j++ {
			for i := 0; i < (*t).X; i++ {
				valueFloat := (*t).GetValue(i, j, k)
				if valueFloat < 0 {
					valueFloat = 0
				}
				(*out).SetValue(i, j, k, valueFloat)
			}
		}
	}
}

// Pool - doing pooling for Tensor. "poolType" can be "max"/"min"/"average"
func Pool(t *Tensor, out *Tensor, strideWidth, strideHeight int, poolType string) {
	// outMatrix = NewTensorEmpty((*t).X/strideWidth, (*t).Y/strideHeight, 1)
	switch poolType {
	case "max":
		for y := 0; y < out.Y; y++ {
			for x := 0; x < out.X; x++ {
				mapY := x * strideHeight
				mapX := y * strideWidth
				mapZ := 0 // Чтоб было
				_ = mapZ
				maxValue := -1.0 * math.MaxFloat64
				for j := 0; j < strideHeight; j++ {
					for i := 0; i < strideWidth; i++ {
						ii := i + mapY
						jj := j + mapX
						tmp := (*t).GetValue(ii, jj, 0)
						if tmp > maxValue {
							maxValue = tmp
						}
					}
				}
				(*out).SetValue(x, y, 0, maxValue)
			}
		}
		break
	case "min":
		for y := 0; y < (*out).X; y++ {
			for x := 0; x < (*out).Y; x++ {
				mapX := x * strideHeight
				mapY := y * strideWidth
				minValue := math.MaxFloat64
				for i := 0; i < strideHeight; i++ {
					for j := 0; j < strideWidth; j++ {
						tmp := (*t).GetValue(i+mapX, j+mapY, 0)
						if tmp < minValue {
							minValue = tmp
						}
					}
				}
				(*out).SetValue(x, y, 0, minValue)
			}
		}
		break
	case "average":
		for y := 0; y < (*out).X; y++ {
			for x := 0; x < (*out).Y; x++ {
				mapX := x * strideHeight
				mapY := y * strideWidth
				summ := 0.0
				for i := 0; i < strideHeight; i++ {
					for j := 0; j < strideWidth; j++ {
						tmp := (*t).GetValue(i+mapX, j+mapY, 0)
						summ += tmp
					}
				}
				(*out).SetValue(x, y, 0, summ/(float64(strideWidth)*float64(strideHeight)))
			}
		}
		break
	default:
		break
	}
	// return outMatrix
}

// Sub - Sub t2's values from t1's values
func (t1 *Tensor) Sub(t2 *Tensor) *Tensor {
	newT := NewTensorCopy(t1)
	for i := range (*t2).Data {
		newT.Data[i] -= (*t2).Data[i]
	}
	return newT
}
