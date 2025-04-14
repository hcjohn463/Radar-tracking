import os
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from tqdm import tqdm
from matplotlib import cm
from scipy.ndimage import label
from tracker import KalmanTracker, MultiTargetTracker
import pandas as pd

# 載入 RDI 資料
file_path = "bin1.h5"
with h5py.File(file_path, 'r') as file:
    rdi_data = file['DS1'][0]
num_frames = rdi_data.shape[2]

# 設定儲存目錄
output_dir = "rdm_frames"
os.makedirs(output_dir, exist_ok=True)

# CFAR 參數
def ca_cfar_2d(rdm, guard_rows, guard_cols, ref_rows, ref_cols, scale_factor):
    # ... (之前的 ca_cfar_2d_corrected 函數代碼) ...
    rows, cols = rdm.shape
    binary_map = np.zeros_like(rdm, dtype=np.uint8)
    threshold_map = np.full_like(rdm, np.inf, dtype=float) # 可選：用於除錯

    win_r_half = guard_rows + ref_rows
    win_c_half = guard_cols + ref_cols
    guard_r_half = guard_rows
    guard_c_half = guard_cols

    pad_width = ((win_r_half, win_r_half), (win_c_half, win_c_half))
    padded_rdm = np.pad(rdm, pad_width, mode='edge') # 使用 'edge' 填充通常比 0 好

    for r in range(rows):
        for c in range(cols):
            r_pad = r + win_r_half
            c_pad = c + win_c_half
            window = padded_rdm[r_pad - win_r_half : r_pad + win_r_half + 1,
                                c_pad - win_c_half : c_pad + win_c_half + 1]
            guard_mask = np.zeros_like(window, dtype=bool)
            guard_r_center = win_r_half
            guard_c_center = win_c_half
            guard_mask[guard_r_center - guard_r_half : guard_r_center + guard_r_half + 1,
                       guard_c_center - guard_c_half : guard_c_center + guard_c_half + 1] = True
            ref_values = window[~guard_mask]
            N = len(ref_values)
            if N == 0:
                threshold = np.inf
            else:
                background_estimate = np.mean(ref_values)
                threshold = background_estimate * scale_factor
            threshold_map[r, c] = threshold
            if rdm[r, c] > threshold:
                binary_map[r, c] = 1
    return binary_map
def filter_clusters(binary_map, min_size=4):
    labeled, num = label(binary_map)
    for i in range(1, num + 1):
        if np.sum(labeled == i) < min_size:
            binary_map[labeled == i] = 0
    return binary_map

# 建立所有幀的影像
tracker = MultiTargetTracker()
for frame_id in tqdm(range(num_frames), desc="Processing frames"):
    rdm = rdi_data[:, :, frame_id].astype(np.float32)

    #bin:9 hand:8
    binary_map = ca_cfar_2d(rdm, 6, 6, 5, 5, 9)

    # Step 2: 濾掉雜訊 cluster
    cleaned = filter_clusters(binary_map, min_size= 6)
    coords = np.column_stack(np.where(cleaned == 1))

    if len(coords) > 0:
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
        clusterer = DBSCAN(eps=1.5, min_samples=2)
        labels = clusterer.fit_predict(coords)
    else:
        labels = np.array([])

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    h, w = binary_map.shape

    fig, ax = plt.subplots(figsize=(6.4, 4.8))  # 約 640x480 畫面
    ax.imshow(binary_map, cmap='hot')
    ax.set_title(f"Tracking (Frame {frame_id}) - {len(tracker.trackers)} active targets")
    ax.axis('off')

    # for cluster_id in range(num_clusters):
    #     cluster_points = coords[labels == cluster_id]
    #     r_mean, d_mean = cluster_points.mean(axis=0)
    #     ax.plot(d_mean, r_mean, 'bo')
    detections = []
    for cluster_id in range(num_clusters):
        cluster_points = coords[labels == cluster_id]
        r_vals, d_vals = cluster_points[:, 0], cluster_points[:, 1]
        r_min, r_max = r_vals.min(), r_vals.max()
        d_min, d_max = d_vals.min(), d_vals.max()
        r_center = r_vals.mean()
        d_center = d_vals.mean()
        bbox = (r_min, d_min, r_max - r_min + 1, d_max - d_min + 1)
        detections.append(((r_center, d_center), bbox))  # ⬅️ 傳中心＋大小


    # 更新追蹤器
    tracker.update(detections, frame_id)

    # 畫追蹤器結果
    tracker.draw(ax)

    frame_path = os.path.join(output_dir, f"frame_{frame_id:03d}.png")
    plt.savefig(frame_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()

# 將圖像轉為影片
video_path = "kal_bin_cfar_dbscan.mp4"
frame_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")])
frame_example = cv2.imread(frame_files[0])
height, width, _ = frame_example.shape

video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

for file in frame_files:
    frame = cv2.imread(file)
    video_writer.write(frame)

video_writer.release()

df = pd.DataFrame(tracker.log)
df.to_csv("target_log.csv", index=False)
# #處理尚未死亡的 tracker（補 log）
# for t in tracker.trackers:
#     t.death_frame = num_frames - 1  # 最後一幀視為離場
#     tracker.log.append({
#         'id': t.id,
#         'birth': t.birth_frame,
#         'death': t.death_frame,
#         'total_visible': t.total_visible
#     })

# df = pd.DataFrame(tracker.log)
# df.to_csv("target_log.csv", index=False)
# print("✅ Log saved to target_log.csv")
