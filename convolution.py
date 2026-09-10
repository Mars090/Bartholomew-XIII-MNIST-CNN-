import numpy as np #i luh numpy
from layer import Layer
import correlate


class Convolution(Layer):
    def __init__(self, in_shape, kernel_size, depth):
        # Unpack input shape tuple: (channels, height, width)
        in_depth, in_height, in_width = in_shape

        self.depth = depth # Number of kernels (output channels)
        self.in_shape = in_shape # Input shape (channels, height, width)
        self.in_depth = in_depth # Number of input channels

        # Calculate output spatial dimensions after convolution
        self.output_shape = (depth,
                             in_height - kernel_size + 1,
                             in_width - kernel_size + 1)

        # Kernel shape: (number of kernels, input channels, kernel height, kernel width)
        self.kernels_shape = (depth, in_depth, kernel_size, kernel_size)

        # Initialise kernels with random values (normal distribution)
        self.kernels = np.random.randn(*self.kernels_shape)

        # Initialise biases randomly, one bias per output feature map spatial position
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        # Save the input for use in backpropagation
        self.input = input

        # Start output with biases copied (shape: output_shape)
        self.output = np.copy(self.biases)

        # Compute convolution output by correlating each input channel with each kernel
        # Iterate over each output depth channel
        for i in range(self.depth):
            # Iterate over each input depth channel
            for j in range(self.in_depth):
                # Add correlation between input channel and kernel to output channel
                self.output[i] += correlate.correlate2d(
                    self.input[j], self.kernels[i, j], "valid")

        # Return the convolution result with biases added
        return self.output

    def backward(self, output_grad, learning_rate):
        # Initialise gradients for kernels and input with zeros (same shapes as kernels and input)
        kernels_grad = np.zeros(self.kernels_shape)
        input_grad = np.zeros(self.in_shape)

        # Compute gradients for kernels and input for each kernel and input channel
        for i in range(self.depth):
            for j in range(self.in_depth):
                # Gradient of kernel i,j is correlation of input channel j and output gradient for channel i
                kernels_grad[i, j] = correlate.correlate2d(
                    self.input[j], output_grad[i], "valid")

                # Gradient of input channel j is convolution of output gradient channel i with kernel i,j
                input_grad[j] += correlate.convolve2d(
                    output_grad[i], self.kernels[i, j], "full")

        # Update kernels by gradient descent: subtract learning rate scaled kernel gradients
        self.kernels -= learning_rate * kernels_grad

        # Update biases by gradient descent: subtract learning rate scaled output gradient
        self.biases -= learning_rate * output_grad

        # Return gradient with respect to input to propagate backward in the network
        return input_grad
