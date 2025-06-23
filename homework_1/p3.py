import cv2
import numpy as np
import matplotlib.pyplot as plt

# 載入圖像
image_path = './data/figure1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 定義 3x3 Gaussian mask
gaussian_mask = (1/16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

# 應用濾波器
filtered_image = cv2.filter2D(image, -1, gaussian_mask)

# 顯示原圖與濾波後的圖像
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Filtered Image (Gaussian)")
plt.imshow(filtered_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
