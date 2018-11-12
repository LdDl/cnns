[![GoDoc](https://godoc.org/github.com/golang/gddo?status.svg)](https://godoc.org/github.com/LdDl/cnns/nns)

![alt text](https://raw.githubusercontent.com/LdDl/cnns/master/cnns_png.png)

# CNNs #
CNNS (Convolutional Neural Networks) is a little package for developing simple neural networks, such as CNN (you don't say?) and MLP.

# WE ARE WORKING ON DOCS AND SOME MAJOR FIXES CURRENTLY

## Table of Contents

- [Features](#features)
- [Theoretical](#theoretical)
    - [MLP](#mlp)
    - [CNN](#cnn)
- [Installation](#installation)
- [Usage](#usage)
- [ToDo](#todo)
- [Support](#support)

## Features

- CNN (convolutional neural network)
- MLP (multilayer perceptron)

## Theoretical

### MLP
@todo
### CNN
@todo

* Step-by-step feedforward and backpropagation calculations in CNN are made in file of Excel format (xlsx): [step_by_step_cnn.xlsx](https://github.com/LdDl/cnns/blob/master/step_by_step_cnn.xlsx). You can check associated example [examples/conv.go](https://github.com/LdDl/cnns/blob/master/examples/conv.go) also.

* Step-by-step feedforward and backpropagation calculations in MLP:
    1) Without inertia extension (not used in main code of repository):

        [step_by_step_mlp.xlsx](https://github.com/LdDl/cnns/blob/master/step_by_step_mlp.xlsx)
    2) With inirtia extension (this one is used currently):

        [step_by_step_mlp(inertia).xlsx](https://github.com/LdDl/cnns/blob/master/step_by_step_mlp(inertia).xlsx)
        and associated example [examples/mlp_1.go](https://github.com/LdDl/cnns/blob/master/examples/mlp_1.go)

    There are 3 epochs provided (see tabs: Step_1, Step_2, Step_3)) for MLP also.

## Installation

Installation is pretty simple:
```go
go get github.com/LdDl/cnns
```

## Usage

@todo

## ToDo

- Softmax layer;
- Optimization for learning;
- More Excel examples for understanding how feedforward and backpropagate works;
- Define learning parametrs not in library (they are predefined now). Need to change the way network uses learning parameters for backpropagation;
- Padding for Conv layer;
- Dropout layer;
- ~~Integrate inertia momentum into backpropagation functions for FC layer;~~
- Integrate inertia momentum into backpropagation functions for Conv layer.

## Support

If you have troubles or questions please [open an issue](https://github.com/LdDl/cnns/issues/new).