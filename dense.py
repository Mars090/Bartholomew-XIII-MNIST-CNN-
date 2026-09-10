import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Initialise weights with random values
        self.weights = np.random.randn(output_size, input_size)
        # Initialise bias with random values
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        # Store the input for use in the backward pass
        self.input = input
        # Compute the output of the dense layer
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        # Compute the gradient of the weights
        weights_gradient = np.dot(output_gradient, self.input.T)
        # Compute the gradient of the input
        input_gradient = np.dot(self.weights.T, output_gradient)
        # Update the weights using the computed gradient and the learning rate
        self.weights -= learning_rate * weights_gradient
        # Update the bias using the computed gradient and the learning rate
        self.bias -= learning_rate * output_gradient
        # Return the gradient of the input for use in the previous layer
        return input_gradient