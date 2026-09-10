import h5py
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, messagebox, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load MNIST dataset (!!!INSERT YOUR PATH HERE!!!)
h5_path = ''

# Load the pretrained CNN model saved earlier
model = load_model('TRAINED MODEL.h5')
print("Model loaded successfully!")

# Function to load and preprocess the dataset
def load_data(seed):
    np.random.seed(seed)
    with h5py.File(h5_path, 'r') as f:
        x_test = f['x_test'][:]
        y_test = f['y_test'][:]

    if x_test.shape[1] == 784:
        x_test = x_test.reshape(-1, 28, 28, 1)
    elif x_test.shape[1:] == (28, 28):
        x_test = x_test[..., np.newaxis]

    x_test = x_test / 255.0
    y_test = y_test.reshape(-1,)
    return x_test, y_test

# Global variable to hold the current Matplotlib canvas
canvas = None

# Function to display images based on the seed
def display_images():
    global canvas
    try:
        seed = int(seed_entry.get())
        x_test, y_test = load_data(seed)
        indices = np.random.choice(len(x_test), size=20, replace=False)
        x_sample = x_test[indices]
        y_sample = y_test[indices]
        predictions = model.predict(x_sample)

        # Create a new figure
        fig = plt.Figure(figsize=(8, 6))
        for i, idx in enumerate(indices):
            ax = fig.add_subplot(4, 5, i + 1)
            ax.imshow(x_test[idx].squeeze(), cmap='gray')
            ax.set_title(f"T:{y_test[idx]}\nP:{np.argmax(predictions[i])}", fontsize=8)
            ax.axis('off')
        fig.tight_layout()

        # If a canvas already exists, destroy it
        if canvas is not None:
            canvas.get_tk_widget().destroy()

        # Embed the figure into the Tkinter GUI
        canvas = FigureCanvasTkAgg(fig, master=display_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid integer seed.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main GUI window
root = Tk()
root.title("Image Loader")

# Frame for controls
control_frame = Frame(root)
control_frame.pack(pady=10)

Label(control_frame, text="Enter Seed:").grid(row=0, column=0, padx=10, pady=10)
seed_entry = Entry(control_frame)
seed_entry.grid(row=0, column=1, padx=10, pady=10)

Button(control_frame, text="Load Images", command=display_images).grid(
    row=1, column=0, columnspan=2, pady=10
)

# Frame for the embedded Matplotlib plot
display_frame = Frame(root)
display_frame.pack(fill="both", expand=True, padx=10, pady=10)

root.mainloop()