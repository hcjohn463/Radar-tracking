import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 讀取 .h5 檔案
def read_h5_data(file_path):
    with h5py.File(file_path, 'r') as file:
        rdi_data = file['DS1'][0]  # 取得 RDI 數據
        phd_data = file['DS1'][1]  # 取得 PHD 數據
    return rdi_data, phd_data

# 設定 X 軸和 Y 軸範圍
velocity_range = (-0.15, 0.15)  # RDI 圖 X 軸速度範圍（cm/s）
angle_range = (-90, 90)  # PHD 圖 X 軸角度範圍（度）
distance_range = (35, 0)  # Y 軸距離範圍（cm）

# 檔案路徑
file_path = "bin2.h5"  # 確保這個檔案在當前工作目錄下

# 讀取數據
rdi_data, phd_data = read_h5_data(file_path)
num_frames = rdi_data.shape[2]  # 獲取總幀數

# 建立動畫
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 初始化影像顯示
im1 = ax[0].imshow(rdi_data[:, :, 0], cmap='jet', aspect='auto', origin='upper',
                    extent=[velocity_range[0], velocity_range[1], distance_range[0], distance_range[1]])
ax[0].set_title("RDI (Velocity vs. Distance)")
ax[0].set_xlabel("Velocity (cm/s)")
ax[0].set_ylabel("Distance (cm)")
fig.colorbar(im1, ax=ax[0], label="Intensity")

im2 = ax[1].imshow(phd_data[:, :, 0], cmap='jet', aspect='auto', origin='upper',
                    extent=[angle_range[0], angle_range[1], distance_range[0], distance_range[1]])
ax[1].set_title("PHD (Angle vs. Distance)")
ax[1].set_xlabel("Angle (°)")
ax[1].set_ylabel("Distance (cm)")
fig.colorbar(im2, ax=ax[1], label="Intensity")

# 更新函數
def update(frame):
    im1.set_data(rdi_data[:, :, frame])
    im2.set_data(phd_data[:, :, frame])
    ax[0].set_title(f"RDI (Frame {frame})")
    ax[1].set_title(f"PHD (Frame {frame})")
    return im1, im2

# 創建動畫
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval= 10, blit=False, repeat=False)

plt.show()