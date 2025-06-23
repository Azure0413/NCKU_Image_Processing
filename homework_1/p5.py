import numpy as np
import matplotlib.pyplot as plt

# 定義遮罩 (Figure 3)
mask = np.array([
    [1, 0, 7],
    [5, 1, 8],
    [4, 0, 9]
])

# **1. 執行傅立葉變換**
f_transform = np.fft.fft2(mask)  # 2D 傅立葉變換

# **2. 計算頻譜 (幅值)**
spectrum = np.abs(f_transform)  # 幅值 (傅立葉頻譜)

# **3. 計算相位角**
phase_angle = np.angle(f_transform)  # 相位角

# 顯示計算結果
print("Real part of Fourier transform: ")
print(np.real(f_transform))  # 實部

print("\nImaginary part of Fourier transform: ")
print(np.imag(f_transform))  # 虛部

print("\nFourier Spectrum (Magnitude): ")
print(spectrum)  # 幅值

print("\nPhase Angle: ")
print(phase_angle)  # 相位角

