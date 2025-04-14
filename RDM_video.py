import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
# 讀取 .h5 檔案
def read_h5_data(file_path):
    with h5py.File(file_path, 'r') as file:
        rdi_data = file['DS1'][0]  # 取得 RDI 數據
    return rdi_data

# 檔案路徑
file_path = "bin1.h5"

# 讀取 RDI 數據
rdi_data = read_h5_data(file_path)
num_frames = rdi_data.shape[2]

# 計算整個資料中的最小與最大值
rdm_min = np.inf
rdm_max = -np.inf

for frame_id in range(num_frames):
    rdm = rdi_data[:, :, frame_id].astype(np.float32)
    rdm_min = min(rdm_min, rdm.min())
    rdm_max = max(rdm_max, rdm.max())

rdm_output_dir = "rdm_raw_frames"
os.makedirs(rdm_output_dir, exist_ok=True)

for frame_id in tqdm(range(num_frames), desc="Rendering RDM images"):
    rdm = rdi_data[:, :, frame_id].astype(np.float32)
    velocity_bins = rdm.shape[1]
    range_bins = rdm.shape[0]

    x = np.linspace(-0.15, 0.15, velocity_bins)  # 速度 (cm/s)
    y = np.linspace(0, 35, range_bins)           # 距離 (cm)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(rdm, extent=[x[0], x[-1], y[-1], y[0]], aspect='auto', cmap='jet', vmin=rdm_min, vmax=rdm_max)
    ax.set_title(f'Range-Doppler Map (Frame {frame_id})')
    ax.set_xlabel('Velocity (cm/s)')
    ax.set_ylabel('Range (cm)')
    plt.colorbar(im, ax=ax, label='Amplitude')

    frame_path = os.path.join(rdm_output_dir, f"rdm_frame_{frame_id:03d}.png")
    plt.savefig(frame_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()

# 將 RDM 圖片轉成影片
rdm_video_path = "bin1_rdm_raw_video.mp4"
rdm_frame_files = sorted([os.path.join(rdm_output_dir, f) for f in os.listdir(rdm_output_dir) if f.endswith(".png")])
frame_example = cv2.imread(rdm_frame_files[0])
height, width, _ = frame_example.shape

rdm_video_writer = cv2.VideoWriter(rdm_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
for file in rdm_frame_files:
    frame = cv2.imread(file)
    rdm_video_writer.write(frame)