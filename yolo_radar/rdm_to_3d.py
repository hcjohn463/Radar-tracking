import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 檔案路徑
file_path = "rdi1.h5"
command = 'phd'

# 雷達參數 (根據你提供的資訊)
Tc = 40e-6        # 脈衝寬度 (40 微秒)
Bandwidth = 4e9    # 頻寬 (4 GHz)
range_resolution = 0.0375  # 距離解析度 (3.75 公分)
f0 = 77e9        # 雷達頻率 (Hz)  **重要：確認你的雷達頻率**
c = 3e8          # 光速 (m/s)
azimuth_angle = 0  # 方位角 (弧度) **重要：根據雷達指向更新**
elevation_angle = 0 # 仰角 (弧度) **重要：根據雷達指向更新**

# RDM 尺寸 (假設)
range_bins = 32       # 距離 bins 數量
doppler_bins = 32     # 多普勒 bins 數量
num_frames = 500      # 幀數

# 距離和速度範圍
range_axis = np.arange(range_bins) * range_resolution # 距離軸 (使用給定的距離解析度)

# 計算最大可測速度 (需要知道雷達頻率和脈衝重複頻率)
PRF = 10e3 # 脈衝重複頻率 (10 kHz 假設) **重要：確認你的 PRF**
max_velocity = (c / (4 * f0)) * PRF
doppler_resolution = (2 * max_velocity) / doppler_bins # 計算多普勒解析度
doppler_axis = np.arange(-doppler_bins//2, doppler_bins//2) * doppler_resolution # 假設多普勒軸中心為 0

# 轉換函數
def rdm_to_xyz(range_val, doppler_val, power_val, f0, azimuth, elevation):
    # 計算徑向速度
    v = (doppler_val * c) / (2 * f0)

    # 轉換到直角座標
    x = range_val * np.cos(elevation) * np.cos(azimuth)
    y = range_val * np.cos(elevation) * np.sin(azimuth)
    z = range_val * np.sin(elevation)

    return x, y, z, v, power_val

# 讀取 DS1 資料集
with h5py.File(file_path, "r") as f:
    ds1 = f["//DS1"][:]  # shape: (2, 32, 32, 500)

# 選擇 PhD 數據 (channel 1)
if command == 'rdi':
    frames = ds1[0]
else:
    frames = ds1[1]

# 轉置 frames 為 (500, 32, 32)
frames = np.transpose(frames, (2, 0, 1))

# 影片參數
fps = 30  # 可以根據需要修改
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_name = f"output_{file_path}_{command}_3d.mp4"
video_writer = None  # 初始化 video_writer

# 確定固定的 Power Level Scale 範圍
min_power = np.min(frames)
max_power = np.max(frames)
print(f"Fixed Power Scale: Min = {min_power}, Max = {max_power}")  # 顯示 Scale 範圍

# 處理每一幀
for frame_idx in range(num_frames):
    frame = frames[frame_idx]

    # 創建座標列表
    x_coords = []
    y_coords = []
    z_coords = []
    velocities = []
    powers = []

    # 遍歷每個像素 (RDM 點)
    for i in range(range_bins):
        for j in range(doppler_bins):
            # 提取距離、多普勒和功率
            range_val = range_axis[i]
            doppler_val = doppler_axis[j]
            power_val = frame[i, j] # 使用原始的 Level Scale Color 值

            # 轉換到 3D 座標
            x, y, z, v, power = rdm_to_xyz(range_val, doppler_val, power_val, f0, azimuth_angle, elevation_angle)
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            velocities.append(v)
            powers.append(power)

    # 創建 3D 圖
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 繪製點雲圖 (顏色表示功率)
    scatter = ax.scatter(x_coords, y_coords, z_coords, c=powers, cmap='viridis', marker='o', vmin=min_power, vmax=max_power) # 加入 vmin 和 vmax

    # 添加顏色條
    fig.colorbar(scatter, label='Power (Level Scale Color)')

    # 設置軸標籤
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Point Cloud from Range-Doppler Map')

    # 將 Matplotlib 圖轉換為 OpenCV 圖像
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 在圖像上添加幀數
    print(f"Frame: {frame_idx + 1}/{num_frames}")  # 顯示幀數 (從 1 開始)

    # 初始化 video_writer (只在第一幀執行)
    if video_writer is None:
        height, width, _ = img.shape
        video_writer = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # 寫入幀到影片
    video_writer.write(img)

    plt.close(fig)  # 關閉圖，釋放記憶體

# 釋放影片寫入器
if video_writer is not None:
    video_writer.release()
    print(f"3D 點雲影片已儲存為 {video_name}")
else:
    print("未生成影片。")