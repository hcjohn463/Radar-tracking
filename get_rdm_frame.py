#第 1 步：取得數值矩陣（不是圖像）
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label
from itertools import product
import pandas as pd
import hdbscan
import networkx as nx
import seaborn as sns

# 讀取 .h5 檔案
def read_h5_data(file_path):
    with h5py.File(file_path, 'r') as file:
        rdi_data = file['DS1'][0]  # 取得 RDI 數據
    return rdi_data

def ca_cfar_2d(rdm, guard_cells=(1, 1), reference_cells=(4, 4), scale_factor=1.5):
    """
    rdm: 2D Range-Doppler Map (float32)
    guard_cells: (g_row, g_col)
    reference_cells: (r_row, r_col)
    scale_factor: threshold multiplier
    """
    rows, cols = rdm.shape
    g_row, g_col = guard_cells
    r_row, r_col = reference_cells

    binary_map = np.zeros_like(rdm, dtype=np.uint8)

    # total window size
    win_row = g_row + r_row
    win_col = g_col + r_col

    for i in range(win_row, rows - win_row):
        for j in range(win_col, cols - win_col):
            # define the full window
            r_start = i - win_row
            r_end   = i + win_row + 1
            c_start = j - win_col
            c_end   = j + win_col + 1

            cut_value = rdm[i, j]

            # exclude guard + CUT
            ref_window = rdm[r_start:r_end, c_start:c_end].copy()
            ref_window[r_row:-r_row, r_col:-r_col] = 0  # zero out guard + CUT

            # compute mean of reference cells (non-zero)
            ref_cells = ref_window[ref_window > 0]
            noise_avg = np.mean(ref_cells)

            threshold = noise_avg * scale_factor

            if cut_value > threshold:
                binary_map[i, j] = 1

    return binary_map   

# 檔案路徑
file_path = "hand.h5"

# 讀取 RDI 數據
rdi_data = read_h5_data(file_path)
num_frames = rdi_data.shape[2]

# 取得第100個frame（Python index從0開始，所以是第99個index）
frame_id = 0

rdm = rdi_data[:, :, frame_id]
rdm_f32 = np.array(rdm, dtype=np.float32)


# 計算全圖平均值
threshold = np.mean(rdm_f32)

# 用全圖平均值當門檻產生 binary map
binary_map_mean = (rdm_f32 > threshold).astype(np.uint8)
# 全圖平均門檻 binary map
plt.imshow(binary_map_mean, cmap='gray', aspect='auto')
plt.title("Global Mean Threshold")
plt.show()

# # 設定橫軸與縱軸的範圍
# velocity_bins = rdm_f32.shape[1]
# range_bins = rdm_f32.shape[0]

# # 橫軸：速度 -0.15 ~ 0.15 cm/s
# x = np.linspace(-0.15, 0.15, velocity_bins)

# # 縱軸：距離 0 ~ 35 cm，由上而下
# y = np.linspace(0, 35, range_bins)

# # 畫圖
# plt.figure(figsize=(6, 4))
# plt.imshow(rdm_f32, extent=[x[0], x[-1], y[-1], y[0]], aspect='auto', cmap='jet')
# plt.colorbar(label='Amplitude')
# plt.xlabel('Velocity (cm/s)')
# plt.ylabel('Range (cm)')
# plt.title(f'Range-Doppler Map (Frame {frame_id})')
# plt.show()

# 原本 rdm_f32 有了（例如取第 100 幀）
binary_map = ca_cfar_2d(rdm_f32,
                        guard_cells=(2,2),
                        reference_cells=(5, 5),
                        scale_factor=3.0)

# # 顯示 CFAR 結果
# plt.imshow(binary_map, cmap='hot')
# plt.title(f"Binary Map CFAR(Frame {frame_id})")
# plt.colorbar()
# plt.show()


#第 3 步：使用 clustering

# 找出所有 "亮點" 的座標 (row = range bin, col = doppler bin)
coords = np.column_stack(np.where(binary_map == 1))

# 建立 HDBSCAN 分群模型
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, gen_min_span_tree=True)
# 執行 fit 才會建立內部結構
clusterer.fit(coords)

# # 繪製 Minimum Spanning Tree
# clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=10)
# plt.title(f"HDBSCAN Minimum Spanning Tree (Frame {frame_id})")
# plt.show()

# # 繪製 Condensed Tree 
# clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
# plt.title(f"HDBSCAN Condensed Tree (Frame {frame_id})")
# plt.show()

labels = clusterer.fit_predict(coords)

# 找有幾群（不包含 label=-1 的雜訊）
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"偵測到 {num_clusters} 個目標")

# 基本資訊
h, w = binary_map.shape
range_resolution = 35 / h             # cm per range bin
doppler_resolution = 0.3 / w          # cm/s per doppler bin
doppler_center = w // 2

# # 把每一群的座標算出中心點
# for cluster_id in range(num_clusters):
#     cluster_points = coords[labels == cluster_id]
#     r_mean, d_mean = cluster_points.mean(axis=0)
#     print(f"目標 {cluster_id+1}: range_bin={r_mean:.2f}, doppler_bin={d_mean:.2f}")

# plt.imshow(binary_map, cmap='hot')
# plt.title(f"HDBSCAN clusters (Frame {frame_id})")
# for cluster_id in range(num_clusters):
#     cluster_points = coords[labels == cluster_id]
#     r_mean, d_mean = cluster_points.mean(axis=0)
#     plt.plot(d_mean, r_mean, 'bo')  # doppler 是橫軸，range 是縱軸

# plt.colorbar()
# plt.show()

# 基本資訊
h, w = binary_map.shape
range_resolution = 35 / h             # cm per range bin
doppler_resolution = 0.3 / w          # cm/s per doppler bin
doppler_center = w // 2

for cluster_id in range(num_clusters):
    cluster_points = coords[labels == cluster_id]
    r_bin, d_bin = cluster_points.mean(axis=0)
    real_range = r_bin * range_resolution
    real_speed = (d_bin - doppler_center) * doppler_resolution
    print(f"目標 {cluster_id+1}: 距離 ≈ {real_range:.2f} cm, 速度 ≈ {real_speed:.4f} cm/s")