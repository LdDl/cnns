package im

import (
	"image"
	"io"
)

// Pixel Representation of RGBA
type Pixel struct {
	R int
	G int
	B int
	A int
}

// GetPixels Returns array of pixels for image
func GetPixels(file io.Reader) ([][]Pixel, error) {
	img, _, err := image.Decode(file)

	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	var pixels [][]Pixel
	for y := 0; y < height; y++ {
		var row []Pixel
		for x := 0; x < width; x++ {
			row = append(row, RGBAToPixel(img.At(x, y).RGBA()))
		}
		pixels = append(pixels, row)
	}

	return pixels, nil
}

// RGBAToPixel Returns pixel value based on img.At(x, y).RGBA() result
func RGBAToPixel(r uint32, g uint32, b uint32, a uint32) Pixel {
	return Pixel{int(r / 257), int(g / 257), int(b / 257), int(a / 257)}
}
