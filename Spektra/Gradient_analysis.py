# imports

import cv2
import numpy as np
from tkinter import filedialog
from Recalage import *
import matplotlib.pyplot as plt

# helper function

def get_top_intensity_positions(img, top_percent=0.1):
    # Flatten the image into a 1D array
    flat_img = img.flatten()
    
    # Determine the threshold for the top 0.1% intensity
    threshold = np.percentile(flat_img, 100 - top_percent)
    
    # Get the indices of the pixels above the threshold
    top_indices = np.where(img >= threshold)
    
    # Combine the indices into (x, y) tuples
    top_positions = list(zip(top_indices[0], top_indices[1]))
    
    return top_positions

# load image

img_0 = cv2.imread(filedialog.askopenfilename()).astype(int)
img_0 = ( img_0 * 255 / np.max(img_0) ).astype(np.uint8)

img_1 = cv2.imread(filedialog.askopenfilename()).astype(int)
img_1 = ( img_1 * 255 / np.max(img_1) ).astype(np.uint8)

# Gradient analysis

Gy_0 = Gradient_Y(img_0)
Gx_0 = Gradient_X(img_0)

Gy_1 = Gradient_Y(img_1)
Gx_1 = Gradient_X(img_1)

Py_0 = get_top_intensity_positions(Gy_0)
Px_0 = get_top_intensity_positions(Gx_0)

Py_1 = get_top_intensity_positions(Gy_1)
Px_1 = get_top_intensity_positions(Gx_1)

# Initialize average gradient arrays
avg_gradient_y_0 = np.zeros(21)
avg_gradient_x_0 = np.zeros(21)

avg_gradient_y_1 = np.zeros(21)
avg_gradient_x_1 = np.zeros(21)

# Calculate average gradient for Y direction
for pos in Py_0:
    x = pos[0]
    y = pos[1]

    # Ensure the indices stay within bounds
    if x - 10 >= 0 and x + 10 < Gy_0.shape[0]:
        for i in range(21):
            avg_gradient_y_0[i] += Gy_0[x - 10 + i, y,0]

# Calculate average gradient for X direction
for pos in Px_0:
    x = pos[0]
    y = pos[1]

    # Ensure the indices stay within bounds
    if y - 10 >= 0 and y + 10 < Gx_0.shape[1]:
        for i in range(21):
            avg_gradient_x_0[i] += Gx_0[x, y - 10 + i,0]

# Calculate average gradient for Y direction
for pos in Py_1:
    x = pos[0]
    y = pos[1]

    # Ensure the indices stay within bounds
    if x - 10 >= 0 and x + 10 < Gy_1.shape[0]:
        for i in range(21):
            avg_gradient_y_1[i] += Gy_1[x - 10 + i, y,0]

# Calculate average gradient for X direction
for pos in Px_1:
    x = pos[0]
    y = pos[1]

    # Ensure the indices stay within bounds
    if y - 10 >= 0 and y + 10 < Gx_1.shape[1]:
        for i in range(21):
            avg_gradient_x_1[i] += Gx_1[x, y - 10 + i,0]

# Normalize the gradients
avg_gradient_x_0 /= np.max(avg_gradient_x_0)
avg_gradient_y_0 /= np.max(avg_gradient_y_0)

avg_gradient_x_1 /= np.max(avg_gradient_x_1)
avg_gradient_y_1 /= np.max(avg_gradient_y_1)

# Index for plotting
index = (np.arange(21) - 10).astype(np.int8)

# Plot average gradients
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Plot average gradients in X direction on the first subplot
ax1.plot(index, avg_gradient_x_0, color="#ff0000", lw=4, alpha=0.5, label="gradient horizontal Led")
ax1.plot(index, avg_gradient_x_1, color="#0066ff", lw=4, alpha=0.5, label="gradient horizontal Filtre")
ax1.set_title("Average Gradient in X Direction")
ax1.set_xlabel("Offset from center")
ax1.set_ylabel("Normalized Gradient")
ax1.legend()

# Plot average gradients in Y direction on the second subplot
ax2.plot(index, avg_gradient_y_0, color="#ff0000", lw=4, alpha=0.5, label="gradient vertical Led")
ax2.plot(index, avg_gradient_y_1, color="#0066ff", lw=4, alpha=0.5, label="gradient vertical Filtre")
ax2.set_title("Average Gradient in Y Direction")
ax2.set_xlabel("Offset from center")
ax2.set_ylabel("Normalized Gradient")
ax2.legend()

# Adjust layout to make room for the plots
plt.tight_layout()

# Save the figure
plt.savefig(fd.asksaveasfilename())

plt.show()