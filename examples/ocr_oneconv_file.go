package examples

import (
	"log"

	"github.com/LdDl/cnns/nns"
)

// CheckOCRNetFromFile - solve OCR problem
// It uses GoCV library https://gocv.io (https://github.com/hybridgroup/gocv)
// gocv.IMRead - reading image from filesystem
// gocv.CvtColor - do grayscale
// gocv.Threshold - threshold
// gocv.Resize - resize image
func CheckOCRNetFromFile() {
	// rand.Seed(time.Now().UnixNano())
	var err error
	var net nns.WholeNet
	err = net.ImportFromFile("datasets/ocr_one_conv.json", false)
	if err != nil {
		log.Println(err)
		return
	}
	// Paste your path for test data
	testFiles, err := readFileNames("/home/keep/work/src/github.com/LdDl/cnns/datasets/symbols_test_3/")
	if err != nil {
		log.Println(err)
		return
	}
	testMats, err := readMatsTests(&testFiles)
	_ = testMats
	if err != nil {
		log.Println(err)
		return
	}
	testTrained(&net, &testMats)
}
