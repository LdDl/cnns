package cnns

import (
	"errors"
	"fmt"
	"log"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Train Train neural network
/*
	inputs - input data for training
	desired - target outputs for input

	testData - input data for doing tests
	testDesired - target outputs for testing

	epochsNum - number of epochs
*/
func (n *WholeNet) Train(inputs []*mat.Dense, desired []*mat.Dense, testData []*mat.Dense, testDesired []*mat.Dense, epochsNum int) (float64, float64, error) {
	var err error
	trainError := 0.0
	testError := 0.0

	if len(inputs) != len(desired) {
		return trainError, testError, errors.New("number of inputs not equal to number of desired")
	}

	if len(testData) != len(testDesired) {
		return trainError, testError, errors.New("number of inputs for test not equal to number of desired for test")
	}

	// Initial shuffling of input data
	// rand.Seed(time.Now().UTC().UnixNano())
	for i := range inputs {
		j := rand.Intn(i + 1)
		inputs[i], inputs[j] = inputs[j], inputs[i]
		desired[i], desired[j] = desired[j], desired[i]
	}

	start := time.Now()
	for e := 0; e < epochsNum; e++ {
		// Shuffle training data every epoch
		for i := range inputs {
			j := rand.Intn(i + 1)
			inputs[i], inputs[j] = inputs[j], inputs[i]
			desired[i], desired[j] = desired[j], desired[i]
		}

		st := time.Now()
		for i := range inputs {
			in := inputs[i]
			target := desired[i]
			n.FeedForward(in)
			err := n.Backpropagate(target)
			if err != nil {
				log.Printf("Backpropagate caused error: %s", err.Error())
				return 0.0, 0.0, err
			}
		}
		log.Printf("Epoch #%v done in %v", e, time.Since(st))
	}
	log.Printf("Training %v epochs done in %v", epochsNum, time.Since(start))

	fmt.Println("Evaluating errors...")

	for i := range inputs {
		in := inputs[i]
		target := desired[i]
		n.FeedForward(in)
		out := n.GetOutput()
		loss := mse(target, out)
		trainError += loss
	}

	for i := range testData {
		in := testData[i]
		target := testDesired[i]
		n.FeedForward(in)
		fmt.Println("\n>>>Out:")
		fmt.Println(n.GetOutput())
		fmt.Println(">>>Desired:")
		fmt.Println(target)
		out := n.GetOutput()
		loss := mse(target, out)
		testError += loss
	}

	return trainError, testError, err
}

func mse(t1, t2 *mat.Dense) float64 {
	r, c := t1.Dims()
	ret := mat.NewDense(r, c, nil)
	ret.Sub(t1, t2)
	ret.Pow(ret, 2)

	sum := 0.0
	raw := ret.RawMatrix().Data
	for i := range raw {
		sum += raw[i]
	}
	return sum
}
