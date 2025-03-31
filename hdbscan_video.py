import os
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
import hdbscan

from tqdm import tqdm
from matplotlib import cm

# 載入 RDI 資料
file_path = "hand.h5"
with h5py.File(file_path, 'r') as file:
    rdi_data = file['DS1'][0]
num_frames = rdi_data.shape[2]

# 設定儲存目錄
output_dir = "rdm_frames"
os.makedirs(output_dir, exist_ok=True)

# CFAR 參數
def ca_cfar_2d(rdm, guard_cells=(2, 2), reference_cells=(5, 5), scale_factor=3.0):
    rows, cols = rdm.shape
    g_row, g_col = guard_cells
    r_row, r_col = reference_cells
    binary_map = np.zeros_like(rdm, dtype=np.uint8)
    win_row = g_row + r_row
    win_col = g_col + r_col

    for i in range(win_row, rows - win_row):
        for j in range(win_col, cols - win_col):
            r_start, r_end = i - win_row, i + win_row + 1
            c_start, c_end = j - win_col, j + win_col + 1
            cut_value = rdm[i, j]
            ref_window = rdm[r_start:r_end, c_start:c_end].copy()
            ref_window[r_row:-r_row, r_col:-r_col] = 0
            ref_cells = ref_window[ref_window > 0]
            noise_avg = np.mean(ref_cells)
            threshold = noise_avg * scale_factor
            if cut_value > threshold:
                binary_map[i, j] = 1
    return binary_map

# 建立所有幀的影像
for frame_id in tqdm(range(num_frames), desc="Processing frames"):
    rdm = rdi_data[:, :, frame_id].astype(np.float32)
    binary_map = ca_cfar_2d(rdm)
    coords = np.column_stack(np.where(binary_map == 1))

    if len(coords) > 0:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
        labels = clusterer.fit_predict(coords)
    else:
        labels = np.array([])

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    h, w = binary_map.shape

    fig, ax = plt.subplots(figsize=(6.4, 4.8))  # 約 640x480 畫面
    ax.imshow(binary_map, cmap='hot')
    ax.set_title(f"HDBSCAN clusters (Frame {frame_id}) - {num_clusters} targets")
    ax.axis('off')

    for cluster_id in range(num_clusters):
        cluster_points = coords[labels == cluster_id]
        r_mean, d_mean = cluster_points.mean(axis=0)
        ax.plot(d_mean, r_mean, 'bo')

    frame_path = os.path.join(output_dir, f"frame_{frame_id:03d}.png")
    plt.savefig(frame_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()

# 將圖像轉為影片
video_path = "test_hand_hdbscan_result_video.mp4"
frame_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")])
frame_example = cv2.imread(frame_files[0])
height, width, _ = frame_example.shape

video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

for file in frame_files:
    frame = cv2.imread(file)
    video_writer.write(frame)

video_writer.release()
video_path
