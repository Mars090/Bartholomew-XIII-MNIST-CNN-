# Bartholomew XIII

The goal of this project was to design and implement a convolutional neural network (CNN) from scratch using Python and NumPy to accurately classify handwritten digits from the MNIST dataset.

## Specifically, my objectives were to:
    Build a modular CNN architecture incorporating convolutional, activation, reshape, and fully connected layers.

    Implement forward and backward propagation algorithms manually to gain a deep understanding of CNN mechanics.

    Achieve at least 90% accuracy on the MNIST test dataset through effective training and hyperparameter tuning.

## What Bartholomew XIII Does

Bartholomew implements a **Convolutional Neural Network (CNN)** from scratch (using NumPy and custom classes) to classify handwritten digits from the **MNIST dataset** (digits 0–9).

- Trains on a balanced subset of MNIST images (100 samples per class).
- Achieves around **95% accuracy** after 40 epochs (Took `~ 10 hours` to train).
- Saves the trained model (`TRAINED MODEL.h5`) for later inference/testing.
- Demonstrates core CNN concepts:
  - Convolutional layers 
  - Sigmoid activations
  - Reshaping
  - Dense (fully connected) layers
  - Forward & backward passes and gradient descent.

---

## Files and Their Roles (<- close explorer for full table)

| Filename            | Purpose                                                                                     |
|---------------------|---------------------------------------------------------------------------------------------|
| `CNN.py`            | Main script: loads data, preprocesses, defines network architecture, trains & tests.        |
| `convolution.py`    | Implements the **Convolution** layer (forward & backward pass using correlation).           |
| `correlate.py`      | Low-level 2D **correlation** and **convolution** functions used by convolution layer.       |
| `dense.py`          | Implements **Dense (fully connected)** layer with forward and backward passes.              |
| `layer.py`          | Abstract base class **Layer** defining interfaces for layers (`forward`, `backward`).       |
| `losses.py`         | Mean Squared Error (`mse`) loss function and its derivative (`mse_prime`).                  |
| `network.py`        | Manages sequence of layers, forward passes (`predict`) and training (`train`).              |
| `reshape.py`        | Implements **Reshape** layer to flatten or reshape data between layers.                     |
| `sigmoid.py`        | Implements **Sigmoid activation** function and its derivative.                              |
| `gui.py`            | Implements Tkinter interface to load the pretrained CNN model, accept a user-provided seed, |
|   ^^^               |     and displays 20 random MNIST test images with their predicted and true labels.          |
| `test.py`           | Loads trained model and test data; evaluates accuracy; displays 20 random sample predictions|
| `MNISTdata.hdf5`    | HDF5 file storing MNIST training and testing data arrays.                                   |
| `TRAINED MODEL.h5`  | Saved trained CNN model for inference/testing.

---

## How to Run the Project

1. ### Setup Python Environment  
   Make sure these packages are installed:  
   `numpy, keras, h5py, matplotlib, tensorflow`

2. ### Prepare Data and Model Files
    Place MNISTdata.hdf5 and TRAINED MODEL.h5 in your working directory 
    and update file paths inside scripts accordingly (Mainly in `CNN.py` and `test.py`).

3. ###  Train the Network (Optional)
    Run the training script to train from scratch and save the model:
    `!python CNN.py`
    This preprocesses data, trains the network, and prints training accuracy.
    
    `NOTE: This takes ~10-12 hours to fully train; If the training completes,`
    `it will overwrite the 95% model`

4. ### Test the Trained Model
    Run the test script to evaluate and visualise predictions:
    `python test.py` or `python gui.py`
    This loads the saved model, evaluates on test data, prints accuracy,
    and displays 20 test images with predicted vs true labels.
    You can change the seed of the images to display different images.

    `Note: running 'gui.py' will open a GUI for the user to input the seed, generating 20 random images`

### Notes
    The CNN architecture, training parameters, and dataset preprocessing are fully customisable inside CNN.py.

    Uses Mean Squared Error (MSE) loss and Sigmoid activations for clarity.

    Demonstrates a CNN implementation without using deep learning frameworks (TensorFlow or PyTorch)
    — only NumPy and custom code, except for model saving/loading, testing, and file manipulation.

    Code is modular and clear: each layer and utility is in its own file for easy extension.

    Also I made the network using Python 3.11.5 so if you have any problems with the imports I recommend switching to that Python version and reinstalling the imports to 3.11.5

    Sidenote: I had to change all of the variable names really last minute so if you see any weird variable names just ignore pls :)

