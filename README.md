# Neural Network from Scratch

This project implements a simple feedforward neural network from scratch using NumPy. The network is trained on the MNIST dataset to classify handwritten digits. This README provides a detailed explanation of the mathematics, scientific principles, and code logic used in the project.

## Overview

A neural network is a series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. In this project, we use a multi-layer perceptron (MLP) model, a type of neural network, to classify the digits in the MNIST dataset.

### Key Components

1. **Activation Functions**: Functions that determine whether a neuron should be activated or not. We use the Sigmoid and Tanh functions.
2. **Forward Propagation**: The process of calculating the output of the neural network.
3. **Backward Propagation**: The method used to update the weights and biases to minimize the error of the neural network.
4. **Gradient Descent**: An optimization algorithm used to minimize the cost function by iteratively moving towards the minimum of the function.

## Mathematical Foundations

### Activation Functions

1. **Sigmoid Function**: The sigmoid function is defined as:

   \[
   \sigma(z) = \frac{1}{1 + e^{-z}}
   \]

   It maps any real-valued number into the range (0, 1), making it useful for models where we need to predict the probability as an output.

2. **Tanh Function**: The hyperbolic tangent function (tanh) is defined as:

   \[
   \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
   \]

   It maps input values to the range (-1, 1), which often helps with training by centering the data.

### Forward Propagation

In forward propagation, the input data passes through each layer of the network. Each layer consists of weights, biases, and an activation function. The output of one layer is the input to the next layer.

For a given layer `i`, the computation is:

\[
Z^{[i]} = W^{[i]}A^{[i-1]} + b^{[i]}

\]

\[
A^{[i]} = \text{activation}(Z^{[i]})
\]

where:
- \( Z^{[i]} \) is the weighted input to the neuron.
- \( W^{[i]} \) and \( b^{[i]} \) are the weights and biases of the layer.
- \( A^{[i]} \) is the activated output of the layer.

### Backward Propagation

Backward propagation is used to calculate the gradient of the loss function with respect to each weight by the chain rule. The gradients are then used to update the weights to minimize the loss.

The gradient of the loss with respect to the weights is computed using:

\[
dW^{[i]} = \frac{\partial L}{\partial W^{[i]}}
\]

\[
db^{[i]} = \frac{\partial L}{\partial b^{[i]}}
\]

These gradients are used in the gradient descent algorithm to update the weights:

\[
W^{[i]} = W^{[i]} - \alpha dW^{[i]}
\]

\[
b^{[i]} = b^{[i]} - \alpha db^{[i]}
\]

where \( \alpha \) is the learning rate.

## Code Explanation

### NeuralNetwork Class

- **`__init__`**: Initializes the neural network with empty lists for storing weights, biases, activations, and gradients.

- **`addLayer`**: Adds layers to the neural network by initializing weights and biases randomly for each layer. This method also initializes placeholders for activations and gradients.

- **`forward_propagation`**: Performs the forward pass through the network, computing the output of each layer using the specified activation functions (Tanh for hidden layers and Sigmoid for the output layer).

- **`backward_propagation`**: Computes the gradients of the loss with respect to each parameter using the chain rule. These gradients are then used to update the parameters.

- **`update_parameters`**: Updates the weights and biases using gradient descent.

- **`train`**: Trains the neural network by repeatedly performing forward and backward propagation and updating the parameters over a specified number of epochs.

- **`predict`**: Uses the trained neural network to make predictions on new data.

### Example Usage

```python
import numpy as np
from tensorflow.keras.datasets.mnist import load_data

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train.reshape(x_train.shape[0], -1).T / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).T / 255.0
y_train = np.expand_dims(y_train, axis=0)

# Initialize and train the neural network
nn = NeuralNetwork()
nn.train(x_train, y_train, learning_rate=0.001, epoch=100, layers=[16, 1])

# Predict on test data
predictions = nn.predict(x_test)
