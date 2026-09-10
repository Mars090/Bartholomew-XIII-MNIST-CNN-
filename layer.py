class Layer:
    def __init__(self):
        # This method can be used to initialise layer-specific parameters and structures.
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError("Forward pass not implemented")

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError("Backward pass not implemented")