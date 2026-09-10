import numpy as np
from layer import Layer

class Sigmoid(Layer):
    def __init__(self):
        """
        Initialise the Sigmoid class with the sigmoid function and its derivative.
        """
        pass
    def forward(self, input):
        """
        The sigmoid function.
        
        Arguments:
        x (float): The input to the sigmoid function.

        Returns:
        float: The output of the sigmoid function.
        """
        # Store the input for use in the backward pass
        self.input = input
        # Apply the sigmoid function
        return 1 / (1 + np.exp(-self.input))

    def backward(self, output_gradient, learning_rate):
        """
        The derivative of the sigmoid function.
        
        Arguments:
        x (float): The input to the derivative of the sigmoid function.

        Returns:
        float: The output of the derivative of the sigmoid function.
        """
        # Get the sigmoid of the input
        s = self.forward(self.input)
        # Calculate the derivative of the sigmoid
        s_prime =  s * (1 - s)
        # Multiply by the output gradient for chain rule in backpropagation
        return s_prime * output_gradient