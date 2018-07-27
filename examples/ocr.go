package examples

import (
	"cnns_vika/nns"
	"cnns_vika/utils/u"
	"errors"
	"image"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"strconv"
	"time"

	"gocv.io/x/gocv"
)

var (
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
	trainWidth          = 16
	trainHeight         = 18
	trainDepth          = 1
	adjustAmountOfFiles = 20000
)

// CheckOCR - проверка свёрточного слоя для решения задачи OCR
// It uses GoCV library https://gocv.io (https://github.com/hybridgroup/gocv)
func CheckOCR() {
	rand.Seed(time.Now().UnixNano())
	clayer := nns.NewConvLayer(1, 5, 8, nns.TDsize{X: trainWidth, Y: trainHeight, Z: 1}) //
	conv := &nns.LayerStruct{
		Layer: clayer,
	}
	rlayer := nns.NewReLULayer(clayer.Out.Size)
	relu := &nns.LayerStruct{
		Layer: rlayer,
	}
	mlayer := nns.NewMaxPoolingLayer(2, 2, rlayer.Out.Size)
	maxpool := &nns.LayerStruct{
		Layer: mlayer,
	}

	clayer2 := nns.NewConvLayer(1, 3, 10, mlayer.Out.Size) //
	conv2 := &nns.LayerStruct{
		Layer: clayer2,
	}
	rlayer2 := nns.NewReLULayer(clayer2.Out.Size)
	relu2 := &nns.LayerStruct{
		Layer: rlayer2,
	}
	mlayer2 := nns.NewMaxPoolingLayer(2, 2, rlayer2.Out.Size)
	maxpool2 := &nns.LayerStruct{
		Layer: mlayer2,
	}

	flayer := nns.NewFullConnectedLayer(mlayer2.Out.Size, 22)
	fullyconnected := &nns.LayerStruct{
		Layer: flayer,
	}

	var net nns.WholeNet
	net.Layers = append(net.Layers, conv)
	net.Layers = append(net.Layers, relu)
	net.Layers = append(net.Layers, maxpool)
	net.Layers = append(net.Layers, conv2)
	net.Layers = append(net.Layers, relu2)
	net.Layers = append(net.Layers, maxpool2)
	net.Layers = append(net.Layers, fullyconnected)

	trainFiles, err := readFileNames("/home/keep/work/src/cnns_vika/datasets/symbols_2/")
	if err != nil {
		log.Println(err)
		return
	}
	labelsNumber := len(trainFiles)
	testFiles, err := readFileNames("/home/keep/work/src/cnns_vika/datasets/symbols_test_2/")
	if err != nil {
		log.Println(err)
		return
	}
	if labelsNumber != len(testFiles) {
		err = errors.New("number of labels in train data and test data should be the same (for proper testing, actually)")
		log.Println(err)
		return
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

func train(net *nns.WholeNet, data *map[string][]gocv.Mat) error {
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
			temp.Desired.CopyFrom(target)
			temp.Image = nns.NewTensor(trainWidth, trainHeight, trainDepth)
			imageArray := make([][][]float64, trainDepth)
			for k := range imageArray {
				imageArray[k] = make([][]float64, trainHeight)
				for j := range imageArray[k] {
					imageArray[k][j] = make([]float64, trainWidth)
					for i := range imageArray[k][j] {
						imageArray[k][j][i] = float64(v[mat].GetUCharAt3(j, i, k))
						if imageArray[k][j][i]/255 == 0 {
							imageArray[k][j][i] = 1
						} else if imageArray[k][j][i]/255 == 1 {
							imageArray[k][j][i] = 0
						}
						// fmt.Printf("%v ", imageArray[k][j][i])
					}
					// fmt.Println()
				}
				// fmt.Println()
			}
			temp.Image.CopyFrom(imageArray)
			temp.LabelStr = chars[intK]
			temp.LabelInt = intK
			// if temp.LabelStr == "T" {
			// log.Println(temp.LabelStr)
			// temp.Image.Print()
			// temp.Desired.Print()
			// }
			trainers = append(trainers, temp)
			// break
		}
		// break
	}

	trainers = SuffleTrainers(trainers)
	lentrainer := len(trainers)
	log.Println("Number of train data", lentrainer)
	// for i := 0; i < 500; i++ {
	// 	trainers = append(trainers, trainers[0:lentrainer]...)
	// }
	// trainers = SuffleTrainers(trainers)
	// log.Println("Number of train data", len(trainers))

	for _, t := range trainers {
		// Feedforward
		net.Layers[0].FeedForward(&t.Image)
		for l := 1; l < len(net.Layers); l++ {
			out := net.Layers[l-1].GetOutput()
			net.Layers[l].FeedForward(&out)
		}
		// log.Println("Train output:")
		// net.Layers[len(net.Layers)-1].PrintOutput()
		// log.Println("Desired:")
		// t.Desired.Print()
		// Backpropagate
		difference := net.Layers[len(net.Layers)-1].GetOutput()
		difference.Sub(&t.Desired)
		// difference.Print()
		net.Layers[len(net.Layers)-1].CalculateGradients(&difference)
		for i := len(net.Layers) - 2; i >= 0; i-- {
			grad := net.Layers[i+1].GetGradients()
			net.Layers[i].CalculateGradients(&grad)
		}
		for i := range net.Layers {
			net.Layers[i].UpdateWeights()
		}
	}

	// var xmatrix = [][][]float64{
	// 	[][]float64{
	// 		[]float64{0, 0, 0, 0, 0, 0, 0, 0},
	// 		[]float64{0, 1, 1, 0, 0, 1, 1, 0},
	// 		[]float64{0, 1, 1, 1, 1, 1, 1, 0},
	// 		[]float64{0, 0, 1, 1, 1, 1, 0, 0},
	// 		[]float64{0, 0, 1, 1, 1, 1, 0, 0},
	// 		[]float64{0, 0, 1, 1, 1, 1, 0, 0},
	// 		[]float64{0, 1, 1, 1, 1, 1, 1, 0},
	// 		[]float64{0, 1, 1, 1, 1, 1, 1, 0},
	// 		[]float64{0, 1, 1, 0, 0, 1, 1, 0},
	// 	},
	// }
	// var ximage = nns.NewTensor(8, 9, 1)
	// ximage.CopyFrom(xmatrix)

	// fmt.Println("For X should be: [1, 0, 0], Got:")
	// net.Layers[0].FeedForward(&ximage)
	// for l := 1; l < len(net.Layers); l++ {
	// 	out := net.Layers[l-1].GetOutput()
	// 	net.Layers[l].FeedForward(&out)
	// }
	// net.Layers[len(net.Layers)-1].PrintOutput()

	return err
}

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
			temp.Desired.CopyFrom(target)
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
						if imageArray[k][j][i]/255 == 0 {
							imageArray[k][j][i] = 1
						} else if imageArray[k][j][i]/255 == 1 {
							imageArray[k][j][i] = 0
						}
						// fmt.Printf("%v ", imageArray[k][j][i])
					}
					// fmt.Println()
				}
				// fmt.Println()
			}
			temp.Image.CopyFrom(imageArray)
			testers = append(testers, temp)
		}
	}
	testers = SuffleTrainers(testers)
	for _, t := range testers {
		// Forward
		net.Layers[0].FeedForward(&t.Image)
		for l := 1; l < len(net.Layers); l++ {
			out := net.Layers[l-1].GetOutput()
			net.Layers[l].FeedForward(&out)
		}
		max := -math.MaxFloat64 // net.Layers[len(net.Layers)-1].GetOutput().Data[0]
		for _, value := range net.Layers[len(net.Layers)-1].GetOutput().Data {
			if value > max {
				max = value
			}
		}
		log.Println(t.LabelStr, max)
		// t.Image.Print()
		net.Layers[len(net.Layers)-1].PrintOutput()
		// fmt.Println("Should be:")
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

// readMatsTrain - fill map[string][]gocv.Mat with data.
// adjust - parameter to fill data with same amount of images for each label (needed if you have a few amount of images for some label)
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
			gocv.Threshold(temp, &temp, 127.0, 255.0, gocv.ThresholdBinary)
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
			temp = gocv.IMRead(j, gocv.IMReadColor)
			// Binarization
			gocv.CvtColor(temp, &temp, gocv.ColorRGBAToGray)
			gocv.Threshold(temp, &temp, 127.0, 255.0, gocv.ThresholdBinary)
			// Resize
			gocv.Resize(temp, &temp, image.Pt(trainWidth, trainHeight), 0.0, 0.0, gocv.InterpolationNearestNeighbor)
			ret[k] = append(ret[k], temp.Clone())
			break
		}
		// break
	}
	return ret, err
}

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
