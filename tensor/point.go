package tensor

// Point - dimensions
type Point struct {
	X int `json:"X"`
	Y int `json:"Y"`
	Z int `json:"Z"`
}

// TDsize - alias to Point
type TDsize = Point

// Total Returns total number of elements
func (td *TDsize) Total() int {
	return td.X * td.Y * td.Z
}
