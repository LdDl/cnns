package cnns

import (
	"fmt"

	t "github.com/LdDl/cnns/tensor"
)

// Gradient - Gradient.
/*
	Grad - current value of gradient;
	OldGradient - previous value of gradient.
*/
type Gradient struct {
	Grad    float64
	OldGrad float64
}

// NewGradient - Constructor for Gradient.
func NewGradient() Gradient {
	return Gradient{
		Grad:    0.0,
		OldGrad: 0.0,
	}
}

// Update - Update gradient with new values (based on gradient from previous training iteration).
func (grad *Gradient) Update() {
	grad.OldGrad = grad.Grad //+ (*grad).OldGrad*lp.Momentum
}

// TensorGradient - Tensor for gradient.
type TensorGradient struct {
	Data []Gradient
	Size *t.TDsize
}

// NewTensorGradient - Constructor for tensor of gradients.
func NewTensorGradient(x, y, z int) TensorGradient {
	return TensorGradient{
		Data: make([]Gradient, x*y*z),
		Size: &t.TDsize{
			X: x,
			Y: y,
			Z: z,
		},
	}
}

// Get - Return (i,j,k)-th gradient from tensor.
func (t1 *TensorGradient) Get(x, y, z int) Gradient {
	return t1.Data[z*t1.Size.X*t1.Size.Y+y*t1.Size.X+x]
}

// Set - Set (i,j,k)-th gradient with new gradient value
func (t1 *TensorGradient) Set(x, y, z int, val Gradient) {
	t1.Data[z*t1.Size.X*t1.Size.Y+y*t1.Size.X+x] = val
}

// SetGrad - Set (i,j,k)-th gradient's current value with new one.
func (t1 *TensorGradient) SetGrad(x, y, z int, val float64) {
	t1.Data[z*t1.Size.X*t1.Size.Y+y*t1.Size.X+x].Grad = val
}

// AddToGrad - Add float64 to (i,j,k)-th gradient's current value.
func (t1 *TensorGradient) AddToGrad(x, y, z int, val float64) {
	t1.Data[z*t1.Size.X*t1.Size.Y+y*t1.Size.X+x].Grad += val
}

// Print - Pretty print tensor of gradients.
func (t1 *TensorGradient) Print() {
	mx := t1.Size.X
	my := t1.Size.Y
	mz := t1.Size.Z
	for z := 0; z < mz; z++ {
		fmt.Printf("Dim: %v\n", z)
		for y := 0; y < my; y++ {
			for x := 0; x < mx; x++ {
				fmt.Printf("%.8f\t", t1.Get(x, y, z))
			}
			fmt.Println()
		}
	}
}
