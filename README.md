# Improved Neural Network Model

This repository implements an improved neural network with advanced optimization techniques, including **Gradient Descent (GD)**, **Stochastic Gradient Descent with Momentum (SGD)**, **RMSProp**, and **Adam**. The model supports both standard training and mini-batch training.

## Key Features:
- **Optimizers**: Supports GD, SGD with Momentum, RMSProp, and Adam.
- **Momentum and Adaptive Learning Rates**: Optimizers like SGD with momentum, RMSProp, and Adam help improve convergence speed.
- **Mini-Batch Training**: Supports mini-batch gradient descent for more efficient training.
- **Customizable Hyperparameters**: Configure the learning rate (`alpha`), decay rate, and optimizer type.

## Key Functions:
- **`ImprovedNeuralModel`**: A class extending a basic neural network with advanced optimization methods.
    - **`__init__()`**: Initializes the model and the necessary parameters (velocity and squared gradients for optimizers).
    - **`velocity_initialization()`**: Initializes velocity and squared gradients to zero for each layer.
  
- **Optimizers**:
    - **`gradient_descent_with_momentum()`**: Implements SGD with momentum to help speed up convergence.
    - **`rms_prop()`**: Implements RMSProp for adaptive learning rates based on squared gradients.
    - **`adam()`**: Combines momentum and adaptive learning rates with bias correction, using the Adam optimization algorithm.

- **`optimization_selector()`**: Chooses the optimizer (GD, SGD, RMSProp, or Adam) and updates the model's parameters accordingly.

- **Training Methods**:
    - **`fit()`**: Standard training method using the chosen optimizer and gradient descent.
    - **`mini_batch_fit()`**: Mini-batch training, splitting the data into smaller batches for each update.

## How to Use:
1. Initialize the model: `model = ImprovedNeuralModel()`
2. Choose the optimizer: `model.optimization_selector(alpha=0.01, optimizer='Adam')`
3. Train the model using the `fit()` or `mini_batch_fit()` method.

Example:
```python
model = ImprovedNeuralModel()
model.fit(X_train, Y_train, epochs=1000, alpha=0.01, optimizer="Adam")
