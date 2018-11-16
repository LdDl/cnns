package nns

import (
	"errors"
	"fmt"
	"log"
	"time"
)

// Train Train neural network
/*
	inputs - input data for training
	desired - target outputs for input

	testData - input data for doing tests
	testDesired - target outputs for testing

	epochsNum - number of epochs
*/
func (n *WholeNet) Train(inputs *[]Tensor, desired *[]Tensor, testData *[]Tensor, testDesired *[]Tensor, epochsNum int) (float64, float64, error) {
	var err error
	trainError := 0.0
	testError := 0.0

	if len(*inputs) != len(*desired) {
		return trainError, testError, errors.New("number of inputs not equal to number of desired")
	}

	if len(*testData) != len(*testDesired) {
		return trainError, testError, errors.New("number of inputs for test not equal to number of desired for test")
	}

	start := time.Now()
	for e := 0; e < epochsNum; e++ {
		st := time.Now()
		for i := range *inputs {
			in := (*inputs)[i]
			target := (*desired)[i]
			n.FeedForward(&in)
			n.Backpropagate(&target)
		}
		log.Printf("Epoch #%v done in %v", e, time.Since(st))
	}
	log.Printf("Training %v epochs done in %v", epochsNum, time.Since(start))

	fmt.Println("Evaluating errors...")
	for i := range *inputs {
		in := (*inputs)[i]
		target := (*desired)[i]
		n.FeedForward(&in)
		out := n.GetOutput()
		loss := target.MSE(&out)
		trainError += loss
	}
	for i := range *testData {
		in := (*testData)[i]
		target := (*testDesired)[i]
		n.FeedForward(&in)
		out := n.GetOutput()
		fmt.Println("\n>>>Out:")
		out.Print()
		fmt.Println(">>>Desired:")
		target.Print()
		loss := target.MSE(&out)
		testError += loss
	}

	return trainError, testError, err
}
