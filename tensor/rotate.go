package tensor

// Rot2D90 Rotate tensor (2d component) by 90 degrees. Returns new instance of Tensor
func (t1 *Tensor) Rot2D90(times ...int) *Tensor {
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

// Rot2D180 Rotate tensor (2d component) by 180 degrees. Returns new instance of Tensor
func (t1 *Tensor) Rot2D180() *Tensor {
	return t1.Rot2D90(2)
}

// Rot2D270 Rotate tensor (2d component) by 270 degrees. Returns new instance of Tensor
func (t1 *Tensor) Rot2D270() *Tensor {
	return t1.Rot2D90(3)
}
