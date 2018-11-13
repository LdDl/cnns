package examples

import (
	"errors"
	"fmt"
	"image"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"strconv"
	"time"

	"github.com/LdDl/cnns/nns"

	"github.com/LdDl/cnns/utils/u"

	"gocv.io/x/gocv"
)

var (
	// Labels
	chars = map[int]string{
		0:  "0",
		1:  "1",
		2:  "2",
		3:  "3",
		4:  "4",
		5:  "5",
		6:  "6",
		7:  "7",
		8:  "8",
		9:  "9",
		10: "A",
		11: "B",
		12: "C",
		13: "E",
		14: "H",
		15: "K",
		16: "M",
		17: "O",
		18: "P",
		19: "T",
		20: "X",
		21: "Y",
	}
	trainWidth          = 10
	trainHeight         = 15
	trainDepth          = 1
	adjustAmountOfFiles = 10000 // see 246-th line of this code
)

// CheckOCR - solve OCR problem
// It uses GoCV library https://gocv.io (https://github.com/hybridgroup/gocv)
// gocv.IMRead - reading image from filesystem
// gocv.CvtColor - do grayscale
// gocv.Threshold - threshold
// gocv.Resize - resize image
func CheckOCR() {
	rand.Seed(time.Now().UnixNano())
	conv := nns.NewConvLayer(1, 5, 4, nns.TDsize{X: trainWidth, Y: trainHeight, Z: 1}) //
	relu := nns.NewReLULayer(conv.OutSize())
	maxpool := nns.NewMaxPoolingLayer(2, 2, relu.OutSize())
	fullyconnected := nns.NewFullConnectedLayer(maxpool.OutSize(), 22)
	fullyconnected.SetActivationFunc(nns.ActivationSygmoid)
	fullyconnected.SetActivationDerivativeFunc(nns.ActivationSygmoidDerivative)

	var net nns.WholeNet
	net.Layers = append(net.Layers, conv)
	net.Layers = append(net.Layers, relu)
	net.Layers = append(net.Layers, maxpool)
	net.Layers = append(net.Layers, fullyconnected)

	// Paste your path for training data
	trainFiles, err := readFileNames("/home/keep/work/src/github.com/LdDl/cnns/datasets/symbols_2/")
	if err != nil {
		log.Println(err)
		return
	}
	labelsNumber := len(trainFiles)
	// Paste your path for test data
	testFiles, err := readFileNames("/home/keep/work/src/github.com/LdDl/cnns/datasets/symbols_test_3/")
	if err != nil {
		log.Println(err)
		return
	}
	if labelsNumber != len(testFiles) {
		err = errors.New("Warning: number of labels in train data and test data should be the same (for proper testing, actually)")
		log.Println(err)
		// Just warning. For testing every label you should have equal amount of labels both in train data and test data.
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
	// _ = trainMats
	// err = net.ImportFromFile("datasets/ocr_one_conv.txt", false)
	// if err != nil {
	// 	log.Println(err)
	// 	return
	// }
	// log.Println(net.Layers[0].GetWeights())
	testTrained(&net, &testMats)

	// log.Println(net.Layers[len(net.Layers)-1].GetWeights())

	err = net.ExportToFile("datasets/ocr_one_conv_2k.json")
	if err != nil {
		log.Println(err)
		return
	}
}

// train - train network
func train(net *nns.WholeNet, data *map[string][]gocv.Mat) error {
	st := time.Now()
	var err error
	rand.Seed(time.Now().UnixNano())
	var trainers []Trainer
	for k, v := range *data {
		intK, err := strconv.Atoi(k)
		if err != nil {
			return err
		}
		for mat := range v {
			var temp Trainer
			temp.Desired = nns.NewTensor(22, 1, 1)
			var target [][][]float64
			target = [][][]float64{[][]float64{[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}}
			target[0][0][intK] = 1.0
			temp.Desired.SetData3D(target)
			temp.Image = nns.NewTensor(trainWidth, trainHeight, trainDepth)
			imageArray := make([][][]float64, trainDepth)
			for k := range imageArray {
				imageArray[k] = make([][]float64, trainHeight)
				for j := range imageArray[k] {
					imageArray[k][j] = make([]float64, trainWidth)
					for i := range imageArray[k][j] {
						imageArray[k][j][i] = float64(v[mat].GetUCharAt3(j, i, k))
						imageArray[k][j][i] /= 255.0
						// if imageArray[k][j][i]/255 == 0 {
						// 	imageArray[k][j][i] = 1
						// } else if imageArray[k][j][i]/255 == 1 {
						// 	imageArray[k][j][i] = 0
						// }
						// fmt.Printf("%v ", imageArray[k][j][i])
					}
					// fmt.Println()
				}
				// fmt.Println()
			}
			temp.Image.SetData3D(imageArray)
			temp.LabelStr = chars[intK]
			temp.LabelInt = intK
			trainers = append(trainers, temp)
			// break
		}
		// break
	}

	trainers = SuffleTrainers(trainers)
	lentrainer := len(trainers)
	log.Println("Number of train data", lentrainer)

	for _, t := range trainers {
		// Feedforward
		net.FeedForward(&t.Image)
		// Backward
		net.Backpropagate(&t.Desired)
	}
	log.Println("Elapsed to train", time.Since(st))
	return err
}

// testTrained - test network
func testTrained(net *nns.WholeNet, data *map[string][]gocv.Mat) error {
	var err error
	var testers []Trainer
	for k, v := range *data {
		intK, err := strconv.Atoi(k)
		if err != nil {
			return err
		}
		for mat := range v {
			var target [][][]float64
			target = [][][]float64{[][]float64{[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}}
			target[0][0][intK] = 1.0
			var temp Trainer
			temp.Desired = nns.NewTensor(22, 1, 1)
			temp.Desired.SetData3D(target)
			temp.Image = nns.NewTensor(trainWidth, trainHeight, trainDepth)
			temp.LabelStr = chars[intK]
			temp.LabelInt = intK
			imageArray := make([][][]float64, trainDepth)
			for k := range imageArray {
				imageArray[k] = make([][]float64, trainHeight)
				for j := range imageArray[k] {
					imageArray[k][j] = make([]float64, trainWidth)
					for i := range imageArray[k][j] {
						imageArray[k][j][i] = float64(v[mat].GetUCharAt3(j, i, k))
						imageArray[k][j][i] /= 255.0
						// if imageArray[k][j][i]/255 == 0 {
						// 	imageArray[k][j][i] = 1
						// } else if imageArray[k][j][i]/255 == 1 {
						// 	imageArray[k][j][i] = 0
						// }
						// fmt.Printf("%v ", imageArray[k][j][i])
					}
					// fmt.Println()
				}
				// fmt.Println()
			}
			temp.Image.SetData3D(imageArray)
			testers = append(testers, temp)
		}
	}
	testers = SuffleTrainers(testers)
	for _, t := range testers {
		// Feedforward
		st := time.Now()
		net.FeedForward(&t.Image)
		log.Println("Elapsed to detect", time.Since(st))
		max := -math.MaxFloat64 // net.Layers[len(net.Layers)-1].GetOutput().Data[0]
		maxidx := -math.MaxInt64
		for i, value := range net.Layers[len(net.Layers)-1].GetOutput().Data {
			if value > max {
				max = value
				maxidx = i
			}
		}
		log.Println(t.LabelStr, max, maxidx, chars[maxidx])
		// t.Image.Print()
		net.PrintOutput()
		fmt.Println("Should be:")
		t.Desired.Print()
	}
	return err
}

// Trainer - struct for training. Contains Image, Desired output
type Trainer struct {
	Image    nns.Tensor
	Desired  nns.Tensor
	LabelStr string
	LabelInt int
}

// SuffleTrainers - randomly shuffles a slice
func SuffleTrainers(data []Trainer) []Trainer {
	for i := range data {
		j := rand.Intn(i + 1)
		data[i], data[j] = data[j], data[i]
	}
	return data
}

// readMatsTrain - fill map[string][]gocv.Mat with data (for training)
// adjust - parameter to fill data with same amount of images for each label (needed if you have a few amount of images for some label)
// but, for a good training you have to provide a lot of unique data (not randomly repeated)
func readMatsTrain(data *map[string][]string, adjust int) (map[string][]gocv.Mat, error) {
	var err error
	var ret map[string][]gocv.Mat
	ret = make(map[string][]gocv.Mat)
	for k, v := range *data {
		if _, ok := ret[k]; !ok {
			ret[k] = []gocv.Mat{}
		}
		for _, j := range v {
			var temp gocv.Mat
			temp = gocv.NewMat()
			defer temp.Close()
			temp = gocv.IMRead(j, gocv.IMReadColor)
			// Binarization
			gocv.CvtColor(temp, &temp, gocv.ColorRGBAToGray)
			// gocv.Threshold(temp, &temp, 127.0, 255.0, gocv.ThresholdBinary)
			// Resize
			gocv.Resize(temp, &temp, image.Pt(trainWidth, trainHeight), 0.0, 0.0, gocv.InterpolationNearestNeighbor)
			ret[k] = append(ret[k], temp.Clone())
		}
	}
	for k, v := range ret {
		length := len(ret[k])
		i := 0
		for i < adjust && len(ret[k]) < adjust {
			ret[k] = append(ret[k], v[u.RandomInt(0, length)])
			i++
		}
	}
	return ret, err
}

// readMatsTests - fill map[string][]gocv.Mat with data (for testing)
func readMatsTests(data *map[string][]string) (map[string][]gocv.Mat, error) {
	var err error
	var ret map[string][]gocv.Mat
	ret = make(map[string][]gocv.Mat)
	for k, v := range *data {
		if _, ok := ret[k]; !ok {
			ret[k] = []gocv.Mat{}
		}
		for _, j := range v {
			var temp gocv.Mat
			temp = gocv.NewMat()
			defer temp.Close()
			temp = gocv.IMRead(j, gocv.ColorRGBAToGray)
			// Binarization
			gocv.CvtColor(temp, &temp, gocv.ColorRGBAToGray)
			// gocv.Threshold(temp, &temp, 127.0, 255.0, gocv.ThresholdBinary)
			// gocv.AdaptiveThreshold(temp, &temp, 255.0, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinary, 7, 5)
			// Resize
			gocv.Resize(temp, &temp, image.Pt(trainWidth, trainHeight), 0.0, 0.0, gocv.InterpolationNearestNeighbor)
			ret[k] = append(ret[k], temp.Clone())
			// break
		}
		// break
	}
	return ret, err
}

// readFileNames - get filenames in directory
func readFileNames(dir string) (map[string][]string, error) {
	var err error
	charMap := make(map[string][]string)
	dirs, err := ioutil.ReadDir(dir)
	if err != nil {
		return charMap, err
	}
	for _, f := range dirs {
		chardir := dir + f.Name()
		charMap[f.Name()] = []string{}
		subDir, err := ioutil.ReadDir(chardir)
		if err != nil {
			return charMap, err
		}
		for _, subf := range subDir {
			charDirImage := chardir + "/" + subf.Name()
			charMap[f.Name()] = append(charMap[f.Name()], charDirImage)
		}
	}
	return charMap, err
}
