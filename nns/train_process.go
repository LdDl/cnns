package nns

import (
	"errors"
	"fmt"
	"log"
	"time"
)

// Train Train neural network
func (n *WholeNet) Train(inputs *[]Tensor, desired *[]Tensor, testData *[]Tensor, testDesired *[]Tensor) (float64, float64, error) {
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
	for i := range *inputs {
		in := (*inputs)[i]
		target := (*desired)[i]

		n.FeedForward(&in)
		n.Backpropagate(&target)
	}
	log.Println("Training done in:", time.Since(start))

	fmt.Println("Evaluating errors...")
	for i := range *testData {
		in := (*testData)[i]
		target := (*testDesired)[i]
		n.FeedForward(&in)
		out := n.GetOutput()
		out.Print()
		loss := target.MSE(&out)
		testError += loss
	}

	for i := range *inputs {
		in := (*inputs)[i]
		target := (*desired)[i]
		n.FeedForward(&in)
		out := n.GetOutput()
		loss := target.MSE(&out)
		trainError += loss
	}

	return trainError / float64(len(*inputs)), testError / float64(len(*testData)), err
}
