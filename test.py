import h5py
import numpy as np #i luh numpy
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt

# Load MNIST dataset (!!!INSERT YOUR PATH HERE!!!)
h5_path = 'path/to/your/mnist_dataset.h5'

# IF YOU CHANGE THE SEED, YOU WILL GET DIFFERENT IMAGES, FOR A SET SEED, YOU WILL GET THE SAME IMAGES
np.random.seed(56)


# Open the HDF5 file in read mode
with h5py.File(h5_path, 'r') as f:
    # Read test images dataset into memory
    x_test = f['x_test'][:]
    # Read test labels dataset into memory
    y_test = f['y_test'][:]

# Reshape test images if needed to match the input shape expected by the model
# If images are flattened (784 pixels), reshape to 28x28 with a single channel dimension
if x_test.shape[1] == 784:
    x_test = x_test.reshape(-1, 28, 28, 1)
# Else if images are 28x28 but missing channel dimension, add it
elif x_test.shape[1:] == (28, 28):
    x_test = x_test[..., np.newaxis]

# Normalise pixel values from [0,255] to [0,1] for better neural network performance
x_test = x_test / 255.0

# Ensure labels are flattened into a 1D array
y_test = y_test.reshape(-1,)

# Print shapes of test data and labels for confirmation
print(f"Test data shape: {x_test.shape}, Labels: {y_test.shape}")

# Load the pretrained CNN model saved earlier
model = load_model('TRAINED MODEL.h5')
print("Model loaded successfully!")

# Evaluate the model's performance on the entire test set without verbose output
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
accuracy = test_acc * 100  # Convert accuracy to percentage
print(f"Test Accuracy: {accuracy}%")



# Randomly select 20 unique indices from the test set to visualise predictions
indices = np.random.choice(len(x_test), size=20, replace=False)

# Select the corresponding images and labels using the random indices
x_sample = x_test[indices]
y_sample = y_test[indices]

# Predict the class probabilities for the selected 20 samples
predictions = model.predict(x_sample)

# Create a figure to display 20 images with true and predicted labels
plt.figure(figsize=(15, 8))

# Loop through the selected samples and plot them
for i, idx in enumerate(indices):
    plt.subplot(4, 5, i + 1)  # Arrange images in 4 rows and 5 columns
    plt.imshow(x_test[idx].squeeze(), cmap='gray')  # Show image in grayscale
    # Set the title showing true label and predicted label
    plt.title(f"True: {y_test[idx]}\nPred: {np.argmax(predictions[i])}")
    plt.axis('off')  # Hide axis ticks for cleaner visualisation

# Adjust layout to prevent overlap and display the plot
plt.tight_layout()
plt.show()
