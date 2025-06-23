import cv2
import numpy as np
import matplotlib.pyplot as plt

# 載入圖像（以灰階方式讀取）
image_path = './data/figure2.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# (a) 使用 Sobel 運算元進行銳化
sobel_x = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向梯度
sobel_y = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向梯度
sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # 結合水平與垂直梯度
sobel_sharpened = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# (b) 使用傅立葉變換進行銳化
dft = cv2.dft(np.float32(original_image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 創建高通濾波器
rows, cols = original_image.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols, 2), np.uint8)
r = 30  # 高通濾波器的半徑
cv2.circle(mask, (ccol, crow), r, (0, 0), -1)

# 應用高通濾波器並進行逆傅立葉變換
filtered_dft = dft_shift * mask
dft_ishift = np.fft.ifftshift(filtered_dft)
img_back = cv2.idft(dft_ishift)
fourier_sharpened = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
fourier_sharpened = cv2.normalize(fourier_sharpened, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

sobel_output_path = './sobel_sharpened.jpg'
fourier_output_path = './fourier_sharpened.jpg'

# 存儲圖像
cv2.imwrite(sobel_output_path, sobel_sharpened)
cv2.imwrite(fourier_output_path, fourier_sharpened)
print("已存檔...")