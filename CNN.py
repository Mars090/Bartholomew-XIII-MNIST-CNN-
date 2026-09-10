import h5py  # stores a bunch of data and lets me manipulate it with numpy - i luh numpy
import numpy as np  # i luh numpy
# transforms class vectors into binary class matrices (e.g., 3 -> [0,0,0,1,0,...])
from keras.utils import to_categorical

# Importing custom classes and functions for the neural network
from network import NN
from dense import Dense
from convolution import Convolution
from sigmoid import Sigmoid
from reshape import Reshape
from losses import mse, mse_prime  # Mean squared error loss and its derivative

# Constants
NUM_CLASSES = 10  # Number of output classes for MNIST (digits 0-9)
# I got 95.02% accuracy for 10 digits (in file TRAINED MODEL.h5) with 40 epochs and 100 samples per class

NUM_NODES = 200   # Number of neurons in the fully connected hidden layer

# Function to preprocess the data before feeding into the network


def preprocess_data(x, y, limit):
    indices = []
    # For each class (digit 0 to 9)
    for i in range(NUM_CLASSES):
        # Find indices of all samples belonging to this class, limit to 'limit' samples per class
        class_indices = np.where(y == i)[0][:limit]
        indices.append(class_indices)
    # Combine indices from all classes into one array
    all_indices = np.hstack(indices)
    # Shuffle the combined indices randomly to mix class samples
    all_indices = np.random.permutation(all_indices)
    # Select and reorder the data and labels according to the shuffled indices
    x, y = x[all_indices], y[all_indices]
    # Reshape input images to (samples, channels=1, height=28, width=28)
    x = x.reshape(len(x), 1, 28, 28)
    # Convert labels to one-hot encoded vectors (e.g., digit '3' -> [0,0,0,1,0,...])
    y = to_categorical(y)
    # Reshape labels to (samples, num_classes, 1) to match expected output shape
    y = y.reshape(len(y), NUM_CLASSES, 1)
    return x, y  # return preprocessed data and labels


# Load MNIST dataset (!!!INSERT YOUR PATH HERE!!! FILE NAME IS 'MNISTdata.hdf5')
MNIST_data = h5py.File(
    '', 'r')

# Extract training and testing data from the file
# Convert to float32 for processing
x_train = np.float32(MNIST_data['x_train'][:])
# Flatten labels array and convert to int32
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))

# Reshape flat images to 28x28 pixel grids (grayscale images)
x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)

# Close the HDF5 file to free resources
MNIST_data.close()

# Preprocess train and test data:
# - Select equal number of samples per class (limit=100)
# - Shuffle and reshape images and labels as required by network
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# Define the network architecture as a list of layersz
network_layers = [
    # Convolution layer: input shape (channels=1, 28x28), 3x3 filters, 8 filters
    Convolution((1, 28, 28), 3, 8),
    Sigmoid(), # Sigmoid activation function
    # Second convolution: input channels=8 (from previous), 3x3 filters, 16 filters
    Convolution((8, 26, 26), 3, 16),
    Sigmoid(), # Sigmoid activation
    # Reshape output into a flat vector for dense layers
    Reshape((16, 24, 24), (16 * 24 * 24, 1)),
    # Fully connected dense layer with NUM_NODES neurons
    Dense(16 * 24 * 24, NUM_NODES),
    Sigmoid(), # Sigmoid activation
    # Output dense layer with NUM_CLASSES neurons (one per digit class)
    Dense(NUM_NODES, NUM_CLASSES),
    # Final activation (sigmoid) to produce output probabilities
    Sigmoid()
]

# Instantiate the neural network with the defined layers
CNN = NN(network_layers)

# Train the network
CNN.train(
    mse,           # Loss function: mean squared error
    mse_prime,     # Derivative of loss function for backpropagation
    x_train,       # Training inputs
    y_train,       # Training targets (one-hot labels)
    epochs=40,     # Number of training iterations over entire dataset
    learning_rate=0.1  # Step size for updating weights during training
)

# Initialise counters for tracking accuracy
correct_predictions = 0
total_predictions = 0

# Test the network on the test data
for x, y in zip(x_test, y_test):
    prediction = CNN.predict(x)  # Predict the output for a single input
    if np.argmax(prediction) == np.argmax(y):  # Compare predicted label with true label
        correct_predictions += 1
    total_predictions += 1

# Print the final accuracy as a percentage
print(f"Accuracy: {correct_predictions / total_predictions * 100}%")
