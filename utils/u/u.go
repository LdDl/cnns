package u

import (
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
