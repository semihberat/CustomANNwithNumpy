import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def tanh_derivative(z):
    return 1 - tanh(z) ** 2

class NeuralNetwork:
    def __init__(self):
        self.cost = 0
        self.grads = {"dW": [], "db": [], "dA": [], "dZ": []}
        self.cache = {"b": [], "w": [], "Z": [], "A": []}

    def addLayer(self, nx=0, layers=[4, 8, 2, 1]):
        # Initialize cache and grads with empty lists for each new layer
        for key in ["A", "Z"]:
            self.cache[key].append([])

        for key in ["dW", "db", "dA", "dZ"]:
            self.grads[key].append([])

        # Add weights and biases for each layer
        for i in range(len(layers)):
            if i == 0:
                weight = np.random.rand(layers[i], nx)  # Input layer weights
            else:
                weight = np.random.rand(
                    layers[i], layers[i - 1]
                )  # Hidden and output layer weights

            # Initialize weights and biases
            self.cache["w"].append(weight)
            self.cache["b"].append(np.zeros((layers[i], 1)))

    def forward_propagation(self, A, y):
        length = int(len(self.cache["w"]))
        for i in range(length):
            Z = np.dot(self.cache["w"][i], A) + self.cache["b"][i]
            self.cache["Z"][i] = Z
            if i != length - 1:
                A = tanh(Z)
            else:
                A = sigmoid(Z)
            self.cache["A"][i] = A
        return self.cache

    def backward_propagation(self, y, m=1):
        self.grads["dA"][-1] = -(y / self.cache["A"][-1]) + (
            (1 - y) / (1 - self.cache["A"][-1])
        )
        self.grads["dZ"][-1] = self.grads["dA"][-1] * tanh_derivative(
            self.cache["Z"][-1]
        )
        self.grads["dW"][-1] = np.dot(self.grads["dZ"][-1], self.cache["A"][-2].T) / m
        self.grads["db"][-1] = np.sum(self.grads["dZ"][-1], axis=1, keepdims=True) / m
        for i in range(len(self.cache["w"]) - 2, -1, -1):
            self.grads["dA"][i] = np.dot(
                self.cache["w"][i + 1].T, self.grads["dZ"][i + 1]
            )
            self.grads["dZ"][i] = self.grads["dA"][i] * tanh_derivative(
                self.cache["Z"][i]
            )
            self.grads["dW"][i] = (
                np.dot(self.grads["dZ"][i], self.cache["A"][i - 1].T) / m
            )
            self.grads["db"][i] = np.sum(self.grads["dZ"][i], axis=1, keepdims=True) / m

    def update_parameters(self, learning_rate):
        length = int(len(self.cache["w"]))
        for i in range(length - 1, -1, -1):
            self.cache["w"][i] -= self.grads["dW"][i] * learning_rate
            self.cache["b"][i] -= self.grads["db"][i] * learning_rate

    def train(self, X, y, learning_rate, epoch, layers=[16, 1]):
        self.addLayer(nx=X.shape[0], layers=layers)
        for i in range(epoch):
            self.forward_propagation(X, y)
            self.backward_propagation(y, X.shape[1])
            self.update_parameters(learning_rate)

    def predict(self, X):
        return self.forward_propagation(X, self.cache)["A"][-1]


nn = NeuralNetwork()

from tensorflow.keras.datasets.mnist import load_data

(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.reshape(x_train.shape[0], -1).T
x_train = x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).T
x_test = x_test / 255.0
y_train = np.expand_dims(y_train, axis=0)

nn.train(x_train, y_train, 0.001, 100)
nn.predict(x_test)
