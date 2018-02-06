# Project Title

Implementing a simple feedforward neural network using Numpy and train it
on MNIST dataset

## Getting Started

Install Numpy using pip

`pip install numpy`

Clone the repository

`git clone ... `

Download the data

`./data/download.sh`

### Prerequisites

Python 2.7

Numpy 1.13.3

## Running the nueral network

The parameters for the neural network is defined in train.py

To start the training, you can run
`python train.py`

## Batch normalization

To add mean-only batch normalization, the mean is subtracted: 
`x_hat = x - mu`, where `mu` represents the mean

A bias term is added: `y = x_hat + beta`

Batch normalization would be added to the ouptut of the ReLU units 
(there is a lot of debate as to whether it must be added before or after the activation)

The update for the delta changes as: `delta = delta_prev . W_prev^T * g' * ( 1 - 1/N )`
where `g'` represents the derivative of the activataion function, `N` represents the number of samples in the mini-batch, `.` represents dot product
and `*` represents haddamard product.

The betas are update as: `1 . delta_prev . W_prev^T` where the `1` represents the one vector. Instead of using the one vector, a summation over the first axis can be done to achieve the same results.

## Author

* **Utkarsh Simha** - [Website](http://cseweb.ucsd.edu/~usimha)

## Acknowledgments

* TensorFlow and PyTorch code repositories for reference
* StackOverflow
* Michael Neilson's Neural Network and Deep Learning book for learning
more about neural network and their working



Note: this code is intended for a coding challenge. Please don't plagiarize it
if you are participating in the challenge.
