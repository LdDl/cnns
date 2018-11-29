package tensor

// Conv2D - EXPEREMENTAL(need to be refactored)! Convolve tensor with respect only to X and Y axis (2D)
func (t1 *Tensor) Conv2D(t2 Tensor, stride [2]int, padding [2]int) Tensor {
	outputX := ((*t1).Size.X-t2.Size.X+2*padding[0])/stride[0] + 1
	outputY := ((*t1).Size.Y-t2.Size.Y+2*padding[1])/stride[1] + 1
	outputData := NewTensor(outputX, outputY, t2.Size.Z)
	el := 0
	for i := 0; i < outputY; i = (i + stride[0]) {
		for j := 0; j < outputX; j = (j + stride[1]) {
			if i == 0 && j == 0 {
				el = 0
			} else {
				el++
			}
			for m := 0; m < t2.Size.X; m++ {
				for n := 0; n < t2.Size.Y; n++ {
					ii := i + m
					jj := j + n
					if ii >= 0 && ii < (*t1).Size.Y && jj >= 0 && jj < (*t1).Size.X {
						outputElement := 0.0
						kernelElement := t2.Data[(m*t2.Size.X + n)]
						if i == 0 {
							inputElement := (*t1).Data[((m)*t1.Size.X+n)+j]
							outputElement = inputElement * kernelElement
						} else {
							inputElement := (*t1).Data[ii*(*t1).Size.X+jj]
							outputElement = inputElement * kernelElement
						}
						outputData.Data[el] += outputElement
					}
				}
			}
		}
	}
	return outputData
}
