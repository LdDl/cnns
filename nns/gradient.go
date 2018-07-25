package nns

type Gradient struct {
	Grad    float64
	OldGrad float64
}

func NewGradient() Gradient {
	return Gradient{
		Grad:    0.0,
		OldGrad: 0.0,
	}
}

type TensorGradient struct {
	Data []Gradient
	Size TDsize
}

func NewTensorGradient(x, y, z int) TensorGradient {
	return TensorGradient{
		Data: make([]Gradient, x*y*z),
		Size: TDsize{
			X: x,
			Y: y,
			Z: z,
		},
	}
}

func (t1 *TensorGradient) Get(x, y, z int) Gradient {
	return (*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x]
}

func (t1 *TensorGradient) Set(x, y, z int, val Gradient) {
	(*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x] = val
}

func (t1 *TensorGradient) SetGrad(x, y, z int, val float64) {
	(*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x].Grad = val
}

func (t1 *TensorGradient) AddToGrad(x, y, z int, val float64) {
	(*t1).Data[z*(*t1).Size.X*(*t1).Size.Y+y*(*t1).Size.X+x].Grad += val
}
