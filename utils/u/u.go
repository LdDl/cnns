package u

import (
	"errors"
	"math"
	"math/rand"
)

// SuffleSlice - randomly shuffles a slice
func SuffleSlice(data []interface{}) []interface{} {
	for i := range data {
		j := rand.Intn(i + 1)
		data[i], data[j] = data[j], data[i]
	}
	return data
}

// RandomInt - random integer
func RandomInt(min, max int) int {
	return rand.Intn(max-min) + min
}

// NormalizeRange - normalizing range
func NormalizeRange(f float64, max int, limitMin bool) int {
	if f <= 0 {
		return 0
	}
	max--
	if f >= float64(max) {
		return max
	}
	if limitMin {
		return int(math.Ceil(f))
	}
	return int(math.Floor(f))
}

type Matrix2D [][]float64

func flatten(f Matrix2D) (r, c int, d []float64, err error) {
	r = len(f)
	if r == 0 {
		return 0, 0, nil, errors.New("No row")
	}
	c = len(f[0])
	d = make([]float64, 0, r*c)
	for _, row := range f {
		if len(row) != c {
			return 0, 0, nil, errors.New("Ragge input")
		}
		d = append(d, row...)
	}
	return r, c, d, nil
}

func unflatten(r, c int, d []float64) Matrix2D {
	m := make(Matrix2D, r)
	for i := 0; i < r; i++ {
		m[i] = d[i*c : (i+1)*c]
	}
	return m
}
