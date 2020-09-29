package main

import (
	"fmt"
	"image/color"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/LdDl/cnns"
	"github.com/LdDl/cnns/tensor"
	"github.com/LdDl/cnns/utils/u"
	"github.com/fogleman/gg"
	"github.com/nfnt/resize"
	"gonum.org/v1/gonum/mat"
)

var (
	trainWidth          = 30
	trainHeight         = 28
	trainDepth          = 1
	trainImagesPath     = "./out/"
	testImagesPath      = "./out/"
	adjustAmountOfFiles = 200  // see readMatsTrain func
	doAdjust            = true // see readMatsTrain func
	numEpochs           = 15
	dict                = []byte("abcdefghijklmnopqrstwxyz0123456789")
	fontsFile           = "../../datasets/fonts/lucidagrande.ttf"
)

// Example by David Cuadrado https://github.com/dcu
func main() {
	fmt.Printf("Generating fake data...")
	st := time.Now()
	chars, err := generatePNGs(trainImagesPath, fontsFile)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Done in %v\n", time.Since(st))

	fmt.Printf("Preparing network architecture...")
	st = time.Now()
	net := prepareNetwork(len(chars))
	fmt.Printf("Done in %v\n", time.Since(st))

	fmt.Printf("Preparing training data...")
	st = time.Now()
	trainFiles, err := readFileNames(trainImagesPath)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Done in %v\n", time.Since(st))

	fmt.Printf("Preparing test data...")
	st = time.Now()
	testFiles, err := readFileNames(testImagesPath)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Done in %v\n", time.Since(st))

	fmt.Printf("Preparing training data as tensors...")
	st = time.Now()
	trainMats, err := readMatsTrain(&trainFiles, adjustAmountOfFiles, doAdjust)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Done in %v\n", time.Since(st))

	fmt.Printf("Preparing test data as tensors...")
	st = time.Now()
	testMats, err := readMatsTests(&testFiles)
	if err != nil {
		fmt.Println(err)
		return
	}
	_ = testMats
	fmt.Printf("Done in %v\n", time.Since(st))

	fmt.Printf("Start training...\n")
	st = time.Now()
	err = train(net, trainMats, chars)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Done in %v\n", time.Since(st))

	fmt.Printf("Start testing...\n")
	st = time.Now()
	err = testTrained(net, testMats, chars)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Done in %v\n", time.Since(st))
}

func generatePNGs(mainFolder, fontsFile string) (map[int]string, error) {
	chars := make(map[int]string)

	for i := range dict {
		charName := string(dict[i])
		chars[i] = charName
		folderName := fmt.Sprintf("%s/%d", mainFolder, i)
		err := ensureDir(folderName)
		if err != nil {
			return nil, err
		}

		fname := fmt.Sprintf("%s/0.png", folderName)
		dc := gg.NewContext(trainWidth, trainHeight)
		dc.SetRGB(1, 1, 1)
		dc.Clear()
		dc.SetRGB(0, 0, 0)
		dc.Scale(1.0, 1.0)
		err = dc.LoadFontFace(fontsFile, 38)
		if err != nil {

			return nil, err
		}
		dc.DrawStringAnchored(charName, float64(trainWidth/2), float64(trainHeight/2), 0.5, 0.5)
		err = dc.SavePNG(fname)

		if err != nil {
			return nil, err
		}
	}
	return chars, nil
}

func prepareNetwork(classesNum int) *cnns.WholeNet {
	conv := cnns.NewConvLayer(tensor.TDsize{X: trainHeight, Y: trainWidth, Z: 1}, 1, 5, 4) //
	relu := cnns.NewReLULayer(conv.GetOutputSize())
	maxpool := cnns.NewPoolingLayer(relu.GetOutputSize(), 2, 2, "max", "valid")
	fullyconnected := cnns.NewFullyConnectedLayer(maxpool.GetOutputSize(), classesNum)
	fullyconnected.SetActivationFunc(cnns.ActivationSygmoid)
	fullyconnected.SetActivationDerivativeFunc(cnns.ActivationSygmoidDerivative)

	fullyconnected2 := cnns.NewFullyConnectedLayer(fullyconnected.GetOutputSize(), classesNum*2)
	fullyconnected2.SetActivationFunc(cnns.ActivationSygmoid)
	fullyconnected2.SetActivationDerivativeFunc(cnns.ActivationSygmoidDerivative)

	fullyconnected3 := cnns.NewFullyConnectedLayer(fullyconnected2.GetOutputSize(), classesNum)
	fullyconnected3.SetActivationFunc(cnns.ActivationSygmoid)
	fullyconnected3.SetActivationDerivativeFunc(cnns.ActivationSygmoidDerivative)

	var net cnns.WholeNet
	net.Layers = append(net.Layers, conv)
	net.Layers = append(net.Layers, relu)
	net.Layers = append(net.Layers, maxpool)
	net.Layers = append(net.Layers, fullyconnected)
	net.Layers = append(net.Layers, fullyconnected2)
	net.Layers = append(net.Layers, fullyconnected3)
	return &net
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

// readMatsTrain - fill map[string][]*mat.Dense with data (for training)
// adjust - parameter to fill data with same amount of images for each label (needed if you have a few amount of images for some label)
// but, for a good training you have to provide a lot of unique data (not randomly repeated)
func readMatsTrain(data *map[string][]string, adjust int, doAdjust bool) (map[string][]*mat.Dense, error) {
	var err error
	var ret map[string][]*mat.Dense
	ret = make(map[string][]*mat.Dense)
	for k, v := range *data {
		if _, ok := ret[k]; !ok {
			ret[k] = []*mat.Dense{}
		}
		for _, j := range v {
			img, err := u.ReadImage(j)
			if err != nil {
				return nil, err
			}
			img = resize.Resize(uint(trainWidth), uint(trainHeight), img, resize.Bicubic)
			tmpTensor := mat.NewDense(trainHeight, trainWidth, nil)
			for i := 0; i < trainHeight; i++ {
				for j := 0; j < trainWidth; j++ {
					r, g, b, _ := img.At(j, i).RGBA()
					lum := 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
					pixel := color.Gray{uint8(lum / 256)}
					tmpTensor.Set(i, j, float64(pixel.Y)/255.0)
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

// readMatsTests - fill map[string][]*mat.Dense with data (for testing)
func readMatsTests(data *map[string][]string) (map[string][]*mat.Dense, error) {
	var err error
	var ret map[string][]*mat.Dense
	ret = make(map[string][]*mat.Dense)
	for k, v := range *data {
		if _, ok := ret[k]; !ok {
			ret[k] = []*mat.Dense{}
		}
		for _, j := range v {
			img, err := u.ReadImage(j)
			if err != nil {
				return nil, err
			}
			img = resize.Resize(uint(trainWidth), uint(trainHeight), img, resize.Bicubic)
			tmpTensor := mat.NewDense(trainHeight, trainWidth, nil)
			for i := 0; i < trainHeight; i++ {
				for j := 0; j < trainWidth; j++ {
					r, g, b, _ := img.At(j, i).RGBA()
					lum := 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
					pixel := color.Gray{uint8(lum / 256)}
					tmpTensor.Set(i, j, float64(pixel.Y)/255.0)
				}
			}
			ret[k] = append(ret[k], tmpTensor)
		}
	}
	return ret, err
}

// ensureDir Создание папки (и вложенных) если не имеется
func ensureDir(dirName string) error {
	err := os.MkdirAll(dirName, 0777)
	if err == nil || os.IsExist(err) {
		return nil
	}
	return err
}

// Trainer - struct for training. Contains Image, Desired output
type Trainer struct {
	Image    *mat.Dense
	Desired  *mat.Dense
	LabelStr string
	LabelInt int
}

// SuffleTrainers Randomly shuffles a slice
func SuffleTrainers(data []Trainer) []Trainer {
	for i := range data {
		j := rand.Intn(i + 1)
		data[i], data[j] = data[j], data[i]
	}
	return data
}

// train - train network
func train(net *cnns.WholeNet, data map[string][]*mat.Dense, chars map[int]string) error {
	st := time.Now()
	var err error
	rand.Seed(time.Now().UnixNano())
	var trainers []Trainer
	for k, v := range data {
		intK, err := strconv.Atoi(k)
		if err != nil {
			return err
		}
		for m := range v {
			target := make([]float64, len(chars))
			target[intK] = 1.0
			var temp Trainer
			temp.Desired = mat.NewDense(len(chars), 1, target)
			temp.Image = mat.NewDense(trainHeight, trainWidth, nil)
			temp.Image.CloneFrom(v[m])
			temp.LabelStr = chars[intK]
			temp.LabelInt = intK
			trainers = append(trainers, temp)
		}
	}

	trainers = SuffleTrainers(trainers)
	lentrainer := len(trainers)
	log.Println("Number of train data", lentrainer)

	for e := 0; e < numEpochs; e++ {
		st := time.Now()
		trainers = SuffleTrainers(trainers)
		for _, t := range trainers {
			// Feedforward
			err = net.FeedForward(t.Image)
			if err != nil {
				log.Printf("Feedforward caused error: %s", err.Error())
				return err
			}

			// Backward
			err = net.Backpropagate(t.Desired)
			if err != nil {
				log.Printf("Backpropagate caused error: %s", err.Error())
				return err
			}
		}
		fmt.Printf("Epoch #%v in %v\n", e, time.Since(st))
	}
	log.Println("Elapsed to train:", time.Since(st), "Num of epochs:", numEpochs)
	return err
}

// testTrained - test network
func testTrained(net *cnns.WholeNet, data map[string][]*mat.Dense, chars map[int]string) error {
	var err error
	var testers []Trainer
	for k, v := range data {
		intK, err := strconv.Atoi(k)
		if err != nil {
			return err
		}
		for m := range v {
			target := make([]float64, len(chars))
			target[intK] = 1.0
			var temp Trainer
			temp.Desired = mat.NewDense(len(chars), 1, target)
			temp.Image = mat.NewDense(trainHeight, trainWidth, nil)
			temp.Image.CloneFrom(v[m])
			temp.LabelStr = chars[intK]
			temp.LabelInt = intK
			testers = append(testers, temp)
		}
	}
	testers = SuffleTrainers(testers)
	for _, t := range testers {
		// Feedforward
		err = net.FeedForward(t.Image)
		if err != nil {
			log.Printf("Feedforward caused error: %s [test]", err.Error())
			return err
		}

		max := -math.MaxFloat64
		maxidx := -math.MaxInt64
		for i, value := range net.Layers[len(net.Layers)-1].GetActivatedOutput().RawMatrix().Data {
			if value > max {
				max = value
				maxidx = i
			}
		}
		fmt.Printf("Desired symbol: %s. Got: %s\n", t.LabelStr, chars[maxidx])
		fmt.Println("Actual output is:")
		fmt.Println("\t", net.GetOutput().RawMatrix().Data)
		fmt.Println("Should be:")
		fmt.Println("\t", t.Desired.RawMatrix().Data)
	}
	return err
}
