import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from tracker import KalmanTracker, MultiTargetTracker
from scipy.ndimage import label
from sklearn.cluster import DBSCAN

# Ns = 64 # number of upsamples
# Fs = 0.625 * 10 ** 6 # ADC sampling rate
# Tc = Ns / Fs  # Chirp time
# Tc = 102.4 micro seconds
# BW = 7 * 10**9 # Bandwidth
# S = BW / Tc = BW * Fs / Ns    # Slope
# dmax = Fs * c / (2 * S) = C * Ns / (2 * BW) 
# dres = c / (2 * BW)

# ---------- 參數設定 ----------
c = 3e8
BW = 7e9
Ns = 64
angle_bins = 32
angle_list = np.linspace(-60, 60, angle_bins)
dres = c / (2 * BW)

# ---------- 載入原始資料 ----------
data_path = r"./Test_pattern/two_hand.h5"
raw = h5py.File(data_path, 'r')
cubeRaw = np.transpose(raw['DS1'][:], (2, 1, 0, 3))
sample_num = cubeRaw.shape[0]
up_sample_num = int(sample_num * 0.5)
used_fft_num = int(up_sample_num * 0.5)
chirp_num = cubeRaw.shape[1]
antenna_num = cubeRaw.shape[2]
frame_num = cubeRaw.shape[3]

RX1_image_compansate = raw['RF_CONFIG'].attrs.get('RX1_image_compansate')
RX1_real_compansate = raw['RF_CONFIG'].attrs.get('RX1_real_compansate')

# ---------- CFAR 函式 ----------
def ca_cfar_2d(rdm, guard_r, guard_c, ref_r, ref_c, scale):
    rows, cols = rdm.shape
    binary_map = np.zeros_like(rdm, dtype=np.uint8)
    pad = ((ref_r + guard_r, ref_r + guard_r), (ref_c + guard_c, ref_c + guard_c))
    padded = np.pad(rdm, pad, mode='edge')
    for r in range(rows):
        for c in range(cols):
            r0, c0 = r + pad[0][0], c + pad[1][0]
            window = padded[r0 - (ref_r+guard_r): r0 + ref_r + guard_r + 1,
                            c0 - (ref_c+guard_c): c0 + ref_c + guard_c + 1]
            guard = np.zeros_like(window, dtype=bool)
            guard[ref_r:ref_r + 2*guard_r + 1, ref_c:ref_c + 2*guard_c + 1] = True
            ref = window[~guard]
            threshold = np.mean(ref) * scale if ref.size > 0 else np.inf
            if rdm[r, c] > threshold:
                binary_map[r, c] = 1
    return binary_map

def filter_clusters(binary_map, min_size=4):
    labeled, num = label(binary_map)
    for i in range(1, num + 1):
        if np.sum(labeled == i) < min_size:
            binary_map[labeled == i] = 0
    return binary_map

# ---------- 輸出設定 ----------
output_video = 'multi_target_tracking_polar.mp4'
fig_dir = 'fig_polar'
os.makedirs(fig_dir, exist_ok=True)
tracker = MultiTargetTracker()

# ---------- 主迴圈 ----------
fig = plt.figure()
writer = animation.FFMpegWriter(fps=10)
with writer.saving(fig, output_video, dpi=100):
    for idxFrame in range(frame_num):
        current_raw = cubeRaw[:up_sample_num, :, :, idxFrame] * 2**15
        fft_fast = np.fft.fft(current_raw, up_sample_num, axis=0)
        fft_fast[:, :, 0] *= (RX1_real_compansate - 1j * RX1_image_compansate) / 1024
        fft_fast = fft_fast[:used_fft_num, :, :]
        rdi_complex = np.fft.fftshift(np.fft.fft(fft_fast, chirp_num, axis=1), axes=1)
        rdi_abs = np.abs(rdi_complex[:, :, 0])

        binary = ca_cfar_2d(rdi_abs, 6, 6, 5, 5, 9)
        cleaned = filter_clusters(binary, min_size=6)
        coords = np.column_stack(np.where(cleaned == 1))

        detections = []
        if len(coords) > 0:
            clusterer = DBSCAN(eps=1.5, min_samples=2)
            labels = clusterer.fit_predict(coords)
            for cluster_id in set(labels):
                if cluster_id == -1: continue
                cluster_points = coords[labels == cluster_id]
                r_idx, d_idx = np.round(cluster_points.mean(axis=0)).astype(int)
                if r_idx >= rdi_complex.shape[0] or d_idx >= rdi_complex.shape[1]:
                    continue
                angle_fft = np.fft.fftshift(np.fft.fft(rdi_complex[r_idx, d_idx, :], n=angle_bins))
                angle_idx = np.argmax(np.abs(angle_fft))
                angle_deg = angle_list[angle_idx]
                angle_rad = np.deg2rad(angle_deg)
                range_m = r_idx * dres
                x = range_m * np.cos(angle_rad)
                y = range_m * np.sin(angle_rad)
                detections.append(((x, y), (0.2, 0.2)))  # 假設每個目標為圓形 bbox

        tracker.update(detections, idxFrame)
        print(len(detections), "目標偵測到")
        # 固定 r_max 範圍
        r_max = 0.7

        # 畫 polar plot
        plt.clf()
        ax = plt.subplot(111, polar=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(-60)
        ax.set_thetamax(60)
        ax.set_ylim(0, r_max)

        for t in tracker.trackers:
            color = 'tab:red' if t.missed > 0 else 'tab:green'
            r_pred = np.linalg.norm(t.predicted_pos)
            theta_pred = np.arctan2(t.predicted_pos[1], t.predicted_pos[0])
            ax.plot(theta_pred, r_pred, 's', color='red', label=f'Pred ID {t.id}')
            if t.latest_observation is not None:
                r_obs = np.linalg.norm(t.latest_observation)
                theta_obs = np.arctan2(t.latest_observation[1], t.latest_observation[0])
                ax.plot(theta_obs, r_obs, 'o', color='lime')
                ax.text(theta_obs, r_obs + 0.05, f'ID {t.id}', fontsize=10, ha='center', color='white', bbox=dict(facecolor=color, alpha=0.6, boxstyle='round'))

            hist = np.array(t.history[-20:])
            r_hist = np.linalg.norm(hist, axis=1)
            theta_hist = np.arctan2(hist[:,1], hist[:,0])
            ax.plot(theta_hist, r_hist, linestyle='--', linewidth=2, alpha=0.7)

        ax.plot(0, 0, 'ro')
        ax.text(0, r_max * 0.05, 'Radar', color='red', ha='center')
        ax.set_title(f"Frame {idxFrame}")
        writer.grab_frame()

print("✅ 多目標追蹤影片已完成：", output_video)