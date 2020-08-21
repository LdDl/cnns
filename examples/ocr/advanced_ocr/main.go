package main

import (
	"fmt"
	"image/color"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"strconv"
	"time"

	"github.com/LdDl/cnns"
	t "github.com/LdDl/cnns/tensor"
	"github.com/LdDl/cnns/utils/u"
	"github.com/nfnt/resize"
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
	trainWidth          = 28
	trainHeight         = 28
	trainDepth          = 1
	adjustAmountOfFiles = 2000 // see readMatsTrain func
	doAdjust            = true // see readMatsTrain func
	trainImagesPath     = "../../datasets/ocr_symbols/"
	testImagesPath      = "../../datasets/ocr_symbols_test/"
)

func main() {
	CheckOCR()
}

// CheckOCR - solve OCR problem
func CheckOCR() {
	rand.Seed(time.Now().UnixNano())
	conv := cnns.NewConvLayer(1, 5, 4, t.TDsize{X: trainWidth, Y: trainHeight, Z: 1}) //
	relu := cnns.NewReLULayer(conv.OutSize())
	maxpool := cnns.NewMaxPoolingLayer(2, 2, relu.OutSize())
	fullyconnected := cnns.NewFullConnectedLayer(maxpool.OutSize(), 22)
	fullyconnected.SetActivationFunc(cnns.ActivationSygmoid)
	fullyconnected.SetActivationDerivativeFunc(cnns.ActivationSygmoidDerivative)

	fullyconnected2 := cnns.NewFullConnectedLayer(fullyconnected.OutSize(), 44)
	fullyconnected2.SetActivationFunc(cnns.ActivationSygmoid)
	fullyconnected2.SetActivationDerivativeFunc(cnns.ActivationSygmoidDerivative)

	fullyconnected3 := cnns.NewFullConnectedLayer(fullyconnected2.OutSize(), 22)
	fullyconnected3.SetActivationFunc(cnns.ActivationSygmoid)
	fullyconnected3.SetActivationDerivativeFunc(cnns.ActivationSygmoidDerivative)

	var net cnns.WholeNet
	net.Layers = append(net.Layers, conv)
	net.Layers = append(net.Layers, relu)
	net.Layers = append(net.Layers, maxpool)
	net.Layers = append(net.Layers, fullyconnected)
	net.Layers = append(net.Layers, fullyconnected2)
	net.Layers = append(net.Layers, fullyconnected3)

	// inputs, desired := formTrainDataOCR()

	// inputsTests, desiredTests := formTestDataOCR()

	// numOfEpochs := 15
	// _, _, err := net.Train(&inputs, &desired, &inputsTests, &desiredTests, numOfEpochs)
	// if err != nil {
	// 	log.Fatalln(err)
	// }

	// Paste your path for training data
	trainFiles, err := readFileNames(trainImagesPath)
	if err != nil {
		log.Panicln(err)
	}

	// Paste your path for test data
	testFiles, err := readFileNames(testImagesPath)
	if err != nil {
		log.Panicln(err)
	}

	trainMats, err := readMatsTrain(&trainFiles, adjustAmountOfFiles, doAdjust)
	if err != nil {
		log.Panicln(err)
	}

	testMats, err := readMatsTests(&testFiles)
	if err != nil {
		log.Panicln(err)
	}

	train(&net, &trainMats)
	testTrained(&net, &testMats)

}

func formTrainDataOCR() ([]t.Tensor, []t.Tensor) {
	fileNames, err := readFileNames(trainImagesPath)
	if err != nil {
		log.Panicln(err)
	}
	numExamples := 0
	for _, v := range fileNames {
		numExamples += len(v)
	}
	var inputs []t.Tensor
	var desired []t.Tensor
	tensorFiles := make(map[string][]t.Tensor)
	for k, v := range fileNames {
		if _, ok := tensorFiles[k]; !ok {
			tensorFiles[k] = []t.Tensor{}
		}
		for _, j := range v {
			img, err := u.ReadImage(j)
			if err != nil {
				log.Panicln(err)
			}
			img = resize.Resize(uint(trainWidth), uint(trainHeight), img, resize.Bicubic)
			tmpTensor := t.NewTensor(trainWidth, trainHeight, 1)
			for i := 0; i < trainHeight; i++ {
				for j := 0; j < trainWidth; j++ {
					r, g, b, _ := img.At(j, i).RGBA()
					lum := 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
					pixel := color.Gray{uint8(lum / 256)}
					tmpTensor.Set(j, i, 0, float64(pixel.Y)/255.0)
				}
			}
			tensorFiles[k] = append(tensorFiles[k], tmpTensor)
		}
	}
	if doAdjust {
		for k, v := range tensorFiles {
			length := len(tensorFiles[k])
			i := 0
			for i < adjustAmountOfFiles && len(tensorFiles[k]) < adjustAmountOfFiles {
				tensorFiles[k] = append(tensorFiles[k], v[u.RandomInt(0, length)])
				i++
			}
		}
	}

	for k, v := range tensorFiles {
		intK, err := strconv.Atoi(k)
		if err != nil {
			log.Panicln(err)
		}
		for mat := range v {
			tmpDesired := t.NewTensor(22, 1, 1)
			var target [][][]float64
			target = [][][]float64{[][]float64{[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}}
			target[0][0][intK] = 1.0
			tmpDesired.SetData3D(target)
			tmpInput := t.NewTensor(trainWidth, trainHeight, trainDepth)
			tmpInput.SetData3D(v[mat].GetData3D())
			inputs = append(inputs, tmpInput)
			desired = append(desired, tmpDesired)
		}
	}

	log.Println("Train data total:", len(inputs))
	return inputs, desired
}

func formTestDataOCR() ([]t.Tensor, []t.Tensor) {

	fileNames, err := readFileNames(testImagesPath)
	if err != nil {
		log.Panicln(err)
	}
	numExamples := 0
	for _, v := range fileNames {
		numExamples += len(v)
	}

	var inputs []t.Tensor
	var desired []t.Tensor

	tensorFiles := make(map[string][]t.Tensor)
	for k, v := range fileNames {
		if _, ok := tensorFiles[k]; !ok {
			tensorFiles[k] = []t.Tensor{}
		}
		for _, j := range v {
			img, err := u.ReadImage(j)
			if err != nil {
				log.Panicln(err)
			}
			img = resize.Resize(uint(trainWidth), uint(trainHeight), img, resize.Bicubic)
			tmpTensor := t.NewTensor(trainWidth, trainHeight, 1)
			for i := 0; i < trainHeight; i++ {
				for j := 0; j < trainWidth; j++ {
					r, g, b, _ := img.At(j, i).RGBA()
					lum := 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
					pixel := color.Gray{uint8(lum / 256)}
					tmpTensor.Set(j, i, 0, float64(pixel.Y)/255.0)
				}
			}
			tensorFiles[k] = append(tensorFiles[k], tmpTensor)
		}
	}

	for k, v := range tensorFiles {
		intK, err := strconv.Atoi(k)
		if err != nil {
			log.Panicln(err)
		}
		for mat := range v {
			tmpDesired := t.NewTensor(22, 1, 1)
			var target [][][]float64
			target = [][][]float64{[][]float64{[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}}
			target[0][0][intK] = 1.0
			tmpDesired.SetData3D(target)
			tmpInput := t.NewTensor(trainWidth, trainHeight, trainDepth)
			tmpInput.SetData3D(v[mat].GetData3D())
			inputs = append(inputs, tmpInput)
			desired = append(desired, tmpDesired)
		}
	}

	log.Println("Test data total:", len(inputs))
	return inputs, desired
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

// train - train network
func train(net *cnns.WholeNet, data *map[string][]t.Tensor) error {
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
			temp.Desired = t.NewTensor(22, 1, 1)
			var target [][][]float64
			target = [][][]float64{[][]float64{[]float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}}
			target[0][0][intK] = 1.0
			temp.Desired.SetData3D(target)
			temp.Image = t.NewTensor(trainWidth, trainHeight, trainDepth)
			temp.Image.SetData3D(v[mat].GetData3D())
			temp.LabelStr = chars[intK]
			temp.LabelInt = intK
			trainers = append(trainers, temp)
		}
	}

	// rand.Seed(time.Now().UTC().UnixNano())
	trainers = SuffleTrainers(trainers)
	lentrainer := len(trainers)
	log.Println("Number of train data", lentrainer)

	numEpochs := 15
	for e := 0; e < numEpochs; e++ {
		st := time.Now()
		trainers = SuffleTrainers(trainers)
		for _, t := range trainers {
			// Feedforward
			net.FeedForward(&t.Image)
			// Backward
			net.Backpropagate(&t.Desired)
		}
		fmt.Printf("Epoch #%v in %v\n", e, time.Since(st))
	}
	log.Println("Elapsed to train:", time.Since(st), "Num of epochs:", numEpochs)
	return err
}

// testTrained - test network
func testTrained(net *cnns.WholeNet, data *map[string][]t.Tensor) error {
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
			temp.Desired = t.NewTensor(22, 1, 1)
			temp.Desired.SetData3D(target)
			temp.Image = t.NewTensor(trainWidth, trainHeight, trainDepth)
			temp.LabelStr = chars[intK]
			temp.LabelInt = intK
			temp.Image.SetData3D(v[mat].GetData3D())
			testers = append(testers, temp)
		}
	}
	testers = SuffleTrainers(testers)
	for _, t := range testers {
		// Feedforward
		net.FeedForward(&t.Image)
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
		// net.PrintOutput()
		// fmt.Println("Should be:")
		// t.Desired.Print()
	}
	return err
}

// Trainer - struct for training. Contains Image, Desired output
type Trainer struct {
	Image    t.Tensor
	Desired  t.Tensor
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

// readMatsTrain - fill map[string][]t.Tensor with data (for training)
// adjust - parameter to fill data with same amount of images for each label (needed if you have a few amount of images for some label)
// but, for a good training you have to provide a lot of unique data (not randomly repeated)
func readMatsTrain(data *map[string][]string, adjust int, doAdjust bool) (map[string][]t.Tensor, error) {
	var err error
	var ret map[string][]t.Tensor
	ret = make(map[string][]t.Tensor)
	for k, v := range *data {
		if _, ok := ret[k]; !ok {
			ret[k] = []t.Tensor{}
		}
		for _, j := range v {
			img, err := u.ReadImage(j)
			if err != nil {
				return nil, err
			}
			img = resize.Resize(uint(trainWidth), uint(trainHeight), img, resize.Bicubic)
			tmpTensor := t.NewTensor(trainWidth, trainHeight, 1)
			for i := 0; i < trainHeight; i++ {
				for j := 0; j < trainWidth; j++ {
					r, g, b, _ := img.At(j, i).RGBA()
					lum := 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
					pixel := color.Gray{uint8(lum / 256)}
					tmpTensor.Set(j, i, 0, float64(pixel.Y)/255.0)
				}
			}
			ret[k] = append(ret[k], tmpTensor)
		}
	}

	if doAdjust {
		for k, v := range ret {
			length := len(ret[k])
			i := 0
			for i < adjust && len(ret[k]) < adjust {
				ret[k] = append(ret[k], v[u.RandomInt(0, length)])
				i++
			}
		}
	}

	return ret, err
}

// readMatsTests - fill map[string][]t.Tensor with data (for testing)
func readMatsTests(data *map[string][]string) (map[string][]t.Tensor, error) {
	var err error
	var ret map[string][]t.Tensor
	ret = make(map[string][]t.Tensor)
	for k, v := range *data {
		if _, ok := ret[k]; !ok {
			ret[k] = []t.Tensor{}
		}
		for _, j := range v {
			img, err := u.ReadImage(j)
			if err != nil {
				return nil, err
			}
			img = resize.Resize(uint(trainWidth), uint(trainHeight), img, resize.Bicubic)
			tmpTensor := t.NewTensor(trainWidth, trainHeight, 1)
			for i := 0; i < trainHeight; i++ {
				for j := 0; j < trainWidth; j++ {
					r, g, b, _ := img.At(j, i).RGBA()
					lum := 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
					pixel := color.Gray{uint8(lum / 256)}
					tmpTensor.Set(j, i, 0, float64(pixel.Y)/255.0)
				}
			}
			ret[k] = append(ret[k], tmpTensor)
		}
	}
	return ret, err
}
