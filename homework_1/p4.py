import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image (grayscale)
image = cv2.imread('./data/figure1.jpg', cv2.IMREAD_GRAYSCALE)

# Step 1: Perform Fourier Transform
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)

# Step 2: Design a low-pass filter
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2  # Center of the frequency spectrum
radius = 50  # Radius of the low-pass filter

# Create a circular mask
mask = np.zeros((rows, cols), dtype=np.uint8)
cv2.circle(mask, (ccol, crow), radius, 1, -1)  # Circle centered at (ccol, crow)

# Apply the mask to the frequency spectrum
f_shift_filtered = f_shift * mask

# Step 3: Perform Inverse Fourier Transform
f_ishift = np.fft.ifftshift(f_shift_filtered)
image_smoothed = np.fft.ifft2(f_ishift)
image_smoothed = np.abs(image_smoothed)

# Visualize the results
plt.figure(figsize=(16, 8))

# Original Image
plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Frequency Spectrum (Log Scale)
plt.subplot(1, 4, 2)
plt.title("Frequency Spectrum")
plt.imshow(np.log(1 + np.abs(f_shift)), cmap='gray')
plt.axis('off')

# Filtered Spectrum
plt.subplot(1, 4, 3)
plt.title("Filtered Spectrum")
plt.imshow(np.log(1 + np.abs(f_shift_filtered)), cmap='gray')
plt.axis('off')

# Smoothed Image
plt.subplot(1, 4, 4)
plt.title("Smoothed Image")
plt.imshow(image_smoothed, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
