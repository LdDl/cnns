package examples

import (
	"errors"
	"log"
	"math/rand"
	"time"

	"github.com/LdDl/cnns/nns"
)

// CheckOCRTwoConv - проверка свёрточного слоя для решения задачи OCR
// It uses GoCV library https://gocv.io (https://github.com/hybridgroup/gocv)
func CheckOCRTwoConv() {
	rand.Seed(time.Now().UnixNano())
	conv := nns.NewConvLayer(1, 5, 8, nns.TDsize{X: trainWidth, Y: trainHeight, Z: 1}) //
	relu := nns.NewReLULayer(conv.OutSize())
	maxpool := nns.NewMaxPoolingLayer(2, 2, relu.OutSize())

	conv2 := nns.NewConvLayer(1, 3, 10, maxpool.OutSize()) //
	relu2 := nns.NewReLULayer(conv2.OutSize())
	maxpool2 := nns.NewMaxPoolingLayer(2, 2, relu2.OutSize())

	fullyconnected := nns.NewFullConnectedLayer(maxpool2.OutSize(), 22)

	var net nns.WholeNet
	net.Layers = append(net.Layers, conv)
	net.Layers = append(net.Layers, relu)
	net.Layers = append(net.Layers, maxpool)
	net.Layers = append(net.Layers, conv2)
	net.Layers = append(net.Layers, relu2)
	net.Layers = append(net.Layers, maxpool2)
	net.Layers = append(net.Layers, fullyconnected)

	trainFiles, err := readFileNames("/home/keep/work/src/github.com/LdDl/cnnsdatasets/symbols_2/")
	if err != nil {
		log.Println(err)
		return
	}
	labelsNumber := len(trainFiles)
	testFiles, err := readFileNames("/home/keep/work/src/github.com/LdDl/cnnsdatasets/symbols_test_3/")
	if err != nil {
		log.Println(err)
		return
	}
	if labelsNumber != len(testFiles) {
		err = errors.New("number of labels in train data and test data should be the same (for proper testing, actually)")
		log.Println(err)
		//return
	}

	trainMats, err := readMatsTrain(&trainFiles, adjustAmountOfFiles)
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

	train(&net, &trainMats)

	testTrained(&net, &testMats)
}
