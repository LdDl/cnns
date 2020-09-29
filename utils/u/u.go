package u

import (
	"errors"
	"image"
	"image/jpeg"
	"image/png"
	"math"
	"math/rand"
	"os"

	"golang.org/x/image/bmp"
)

// AndINT - Logical AND for two inputs of type int.
func AndINT(x, y int) int {
	firstBool := false
	secondBool := false
	if x == 1 {
		firstBool = true
	}
	if y == 1 {
		secondBool = true
	}
	outputBool := (firstBool && secondBool)
	outputInt := 0
	if outputBool == true {
		outputInt = 1
	}
	return outputInt
}

// OrINT - Logical OR for two inputs of type int.
func OrINT(x, y int) int {
	firstBool := false
	secondBool := false
	if x == 1 {
		firstBool = true
	}
	if y == 1 {
		secondBool = true
	}
	outputBool := (firstBool || secondBool)
	outputInt := 0
	if outputBool == true {
		outputInt = 1
	}
	return outputInt
}

// XorINT - Logical XOR for two inputs of type int.
func XorINT(x, y int) int {
	firstBool := false
	secondBool := false
	if x == 1 {
		firstBool = true
	}
	if y == 1 {
		secondBool = true
	}
	outputBool := (firstBool != secondBool)
	outputInt := 0
	if outputBool == true {
		outputInt = 1
	}
	return outputInt
}

// SuffleSlice - Shuffle a slice in random way.
func SuffleSlice(data []interface{}) []interface{} {
	for i := range data {
		j := rand.Intn(i + 1)
		data[i], data[j] = data[j], data[i]
	}
	return data
}

// RandomInt - Get random integer.
func RandomInt(min, max int) int {
	return rand.Intn(max-min) + min
}

// NormalizeRange - Normalizing range.
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

//ReadImage - wrapper function for reading png/jpeg/jpg or bmp images
/*
	It can determine type of image and use proper decoder for it
*/
func ReadImage(fname string) (image.Image, error) {
	var err error

	reader, err := os.Open(fname)
	if err != nil {
		return nil, err
	}

	var im image.Image
	var imType string

	// Try to decode
	_, imType, err = image.Decode(reader)
	if err != nil {
		// NOTHING
		// NEED TO DO SEEK(0,0) and use proper decoder
	}

	reader.Seek(0, 0)
	switch imType {
	case "png":
		im, err = png.Decode(reader)
		if err != nil {
			return nil, err
		}
	case "jpeg", "jpg":
		im, err = jpeg.Decode(reader)
		if err != nil {
			return nil, err
		}
	case "bmp":
		im, err = bmp.Decode(reader)
		if err != nil {
			return nil, err
		}
	default:
		return nil, errors.New(err.Error() + ": Please use only png/jpeg/jpg or bmp")
	}

	reader.Close()

	return im, nil
}

// Round Round float64 to 0 decimal places
func Round(v float64) float64 {
	if v >= 0 {
		return math.Floor(v + 0.5)
	}
	return math.Ceil(v - 0.5)
}

// RoundPlaces Round float64 to N decimal places
func RoundPlaces(v float64, places int) float64 {
	shift := math.Pow(10, float64(places))
	return Round(v*shift) / shift
}
