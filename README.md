# MNIST Classification MLP Network

This repository is a two-layer perceptron neural network implementation in Python. This code has includes several experiments on below topics.

- Number of epochs required for convergence
- Effect of changing number of neurons on accuracy and convergence
- Effect of regularization on accuracy and convergence
- Effect of input normalization on accuracy and convergence
- Effect of Initialization on accuracy and convergence
- Effect of learning rate on accuracy and convergence

You can specify the type of experiment using a command-line argument, which will be explained in the following sections.



## Table of Contents

[TOC]



## Getting Started

### Installation

Clone the program.

`git clone https://github.com/raminnakhli/MNIST-Classification-Network.git`



### Prerequisites

First, install requirements.

`pip install -r requirement.txt`

Second, you need to download MNIST or TinyMNIST dataset, and then, put them in the folder in the project repository.

**Notice:**  The dataset should contain 4 csv files including `testData.csv`, `testLabels.csv`, `trainData.csv`, and `trainLabels.csv`. Below is an image of the correct structure of dataset.



![](C:\Users\rnakhli\Desktop\Untitled.png)





## Execution

Now, you can run the experiments with default configuration using the below command.

`python main.py`



## Controlling Network Structure

You can change either the network structure or type of experiment using command-line arguments. Below is a list of such flags.

|         Short Format          |                   Long Format                   |                       Valid Values                       |                         Explanation                          |
| :---------------------------: | :---------------------------------------------: | :------------------------------------------------------: | :----------------------------------------------------------: |
|         -dataset Path         |               --dataset-path Path               |                        Any String                        |             specifies the path of dataset folder             |
|       -tr Training_Type       |           --train-type Training_Type            |                         bgd/sgd                          | specifies training type which can be batch gradient decent (bgd) or stochastic gradient decent (sgd) |
|      -ex Experiment_Type      |           --test-type Experiment_Type           |              con/nc/reg/norm/init/lr/custom              |                 specifies type of experiment                 |
|   -rf Regularization_Factor   |  --regularization-factor Regularization_Factor  |                     Any Float Value                      |         specifies regularization factor for training         |
|       -lr Learning_Rate       |          --learning-rate Learning_Rate          |                     Any Float Value                      |             specifies learning rate for training             |
|        -hs Hidden_Size        |            --hidden-size Hidden_Size            |                    Any Integer Value                     |     specifies hidden size of second layer of the network     |
|              -ne              |             --normalization-enable              |                         No Value                         |                 enabled input normalization                  |
|    -it Initialization_Type    |         --init-type Initialization_Type         |                      random/xavier                       |               specifies type of initialization               |
|       -lf Loss_Function       |          --loss-function Loss_Function          |                       softmax/svd                        |               specifies type of loss function                |
| -af [Activation_Function ...] | --activation-function [Activation_Function ...] | A list of activation functions of relu/tanh/lrelu/linear |            specifies type of activation function             |



## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## Contact

Ramin Ebrahim Nakhli - raminnakhli@gmail.com

Project Link: https://github.com/raminnakhli/MNIST-Classification-Network

