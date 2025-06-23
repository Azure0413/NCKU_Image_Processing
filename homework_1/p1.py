import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖片
image_path = './data/figure1.jpg'
noisy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 平均濾波器
avg_filtered_3x3 = cv2.blur(noisy_image, (3, 3))
avg_filtered_5x5 = cv2.blur(noisy_image, (5, 5))

# 中值濾波器
median_filtered_3x3 = cv2.medianBlur(noisy_image, 3)
median_filtered_5x5 = cv2.medianBlur(noisy_image, 5)

# 圖片集 1: 原圖、3x3平均濾波器、5x5平均濾波器
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1), plt.imshow(noisy_image, cmap='gray'), plt.title('Original Image with Noise')
plt.subplot(1, 3, 2), plt.imshow(avg_filtered_3x3, cmap='gray'), plt.title('3x3 Average Filter')
plt.subplot(1, 3, 3), plt.imshow(avg_filtered_5x5, cmap='gray'), plt.title('5x5 Average Filter')
plt.tight_layout()
plt.savefig('./average_filter_comparison.jpg')

# 圖片集 2: 原圖、3x3中值濾波器、5x5中值濾波器
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1), plt.imshow(noisy_image, cmap='gray'), plt.title('Original Image with Noise')
plt.subplot(1, 3, 2), plt.imshow(median_filtered_3x3, cmap='gray'), plt.title('3x3 Median Filter')
plt.subplot(1, 3, 3), plt.imshow(median_filtered_5x5, cmap='gray'), plt.title('5x5 Median Filter')
plt.tight_layout()
plt.savefig('./median_filter_comparison.jpg')

print("Images saved: './average_filter_comparison.jpg', './median_filter_comparison.jpg'")
