class NN:
    """
    Represents a neural network.
    
    Attributes:
    network (list): The layers of the network.
    """
    def __init__(self, network):
        """
        Initialises the Network with the given layers.

        Arguments:
        network (list): The layers of the network.
        """
        self.network = network
        
    def predict(self, input):
        """
        Performs a forward pass through the network.

        Arguments:
        input (np.array): The input data.

        Returns:
        np.array: The output of the network.
        """
        output = input
        for layer in self.network:
            output = layer.forward(output)
        return output

    def train(self, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
        """
        Trains the network using the given loss function and training data.

        Arguments:
        loss (function): The loss function.
        loss_prime (function): The derivative of the loss function.
        x_train (np.array): The training input data.
        y_train (np.array): The training output data.
        epochs (int, optional): The number of training epochs. Defaults to 1000.
        learning_rate (float, optional): The learning rate for the backward pass. Defaults to 0.01.
        verbose (bool, optional): Whether to print training progress. Defaults to True.

        Returns:
        None
        """
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # forward
                output = self.predict(x)

                # error
                error += loss(y, output)

                # backward
                grad = loss_prime(y, output)
                for layer in reversed(self.network):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)
            if verbose:
                print(f"{e + 1}/{epochs}, error = {error*100}%")