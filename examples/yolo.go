package examples

import (
	"log"
	"math/rand"
	"time"

	"github.com/LdDl/cnns/nns"
)

// YoloTry - solve "AND" problem
func YoloTry() {
	rand.Seed(time.Now().UnixNano())

	var input = nns.NewTensor(608, 608, 3)
	for i := 0; i < 416; i++ {
		for j := 0; j < 416; j++ {
			for z := 0; z < 3; z++ {
				input.Set(i, j, z, rand.Float64()-0.5)
			}
		}
	}

	conv1 := nns.NewConvLayer(1, 3, 16, nns.TDsize{X: 608, Y: 608, Z: 3})
	maxpool1 := nns.NewMaxPoolingLayer(2, 2, conv1.OutSize())
	conv2 := nns.NewConvLayer(1, 3, 32, maxpool1.OutSize())
	maxpool2 := nns.NewMaxPoolingLayer(2, 2, conv2.OutSize())
	conv3 := nns.NewConvLayer(1, 3, 64, maxpool2.OutSize())
	maxpool3 := nns.NewMaxPoolingLayer(2, 2, conv3.OutSize())
	conv4 := nns.NewConvLayer(1, 3, 128, maxpool3.OutSize())
	maxpool4 := nns.NewMaxPoolingLayer(2, 2, conv4.OutSize())
	conv5 := nns.NewConvLayer(1, 3, 256, maxpool4.OutSize())
	maxpool5 := nns.NewMaxPoolingLayer(2, 2, conv5.OutSize())
	conv6 := nns.NewConvLayer(1, 3, 512, maxpool5.OutSize())
	conv7 := nns.NewConvLayer(1, 3, 1024, conv6.OutSize())
	conv8 := nns.NewConvLayer(1, 3, 1024, conv7.OutSize())
	conv9 := nns.NewConvLayer(1, 1, 125, conv8.OutSize())

	var net nns.WholeNet
	net.Layers = append(net.Layers, conv1)
	net.Layers = append(net.Layers, maxpool1)
	net.Layers = append(net.Layers, conv2)
	net.Layers = append(net.Layers, maxpool2)
	net.Layers = append(net.Layers, conv3)
	net.Layers = append(net.Layers, maxpool3)
	net.Layers = append(net.Layers, conv4)
	net.Layers = append(net.Layers, maxpool4)
	net.Layers = append(net.Layers, conv5)
	net.Layers = append(net.Layers, maxpool5)
	net.Layers = append(net.Layers, conv6)
	net.Layers = append(net.Layers, conv7)
	net.Layers = append(net.Layers, conv8)
	net.Layers = append(net.Layers, conv9)

	log.Println("Conv 0", conv1.OutSize())
	log.Println("Maxpool 1", maxpool1.OutSize())
	log.Println("Conv 2", conv2.OutSize())
	log.Println("Maxpool 3", maxpool2.OutSize())
	log.Println("Conv 4", conv3.OutSize())
	log.Println("Maxpool 5", maxpool3.OutSize())
	log.Println("Conv 6", conv4.OutSize())
	log.Println("Maxpool 7", maxpool4.OutSize())
	log.Println("Conv 8", conv5.OutSize())
	log.Println("Maxpool 9", maxpool5.OutSize())
	log.Println("Conv 10", conv6.OutSize())
	log.Println("Conv 11", conv7.OutSize())
	log.Println("Conv 12", conv8.OutSize())
	log.Println("Conv 13", conv9.OutSize())

	startTime := time.Now()
	net.FeedForward(&input)
	log.Println("Elapsed to forward", time.Since(startTime))
}
