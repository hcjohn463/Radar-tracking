import numpy as np
import h5py
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment
import random

###########################
# Simple 2D Kalman Filter
###########################
class SimpleKalman2D:
    def __init__(self, init_pos):
        self.dt = 1.0
        self.x = np.array([[init_pos[0]], [init_pos[1]], [0.0], [0.0]])
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.P = np.eye(4) * 500
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 0.5

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.get_state()

    def update(self, z):
        z = np.array(z).reshape((2, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_state(self):
        return self.x[0, 0], self.x[1, 0]


###########################
# RadarTracker Class
###########################
class RadarTracker:
    def __init__(self, max_lost=3, max_cost=3.5, w_range=1.0, w_doppler=150.0):
        self.next_id = 0
        self.trackers = []
        self.colors = {}  # Assign color to each ID
        self.max_lost = max_lost
        self.max_cost = max_cost
        self.w_range = w_range
        self.w_doppler = w_doppler

    def update(self, detections):
        detections = np.array(detections)

        if len(self.trackers) == 0:
            for det in detections:
                self._create_tracker(det)
            return

        if len(detections) == 0:
            for t in self.trackers:
                t["lost"] += 1
                t["kf"].predict()
            self._remove_lost()
            return

        N, M = len(self.trackers), len(detections)
        cost_matrix = np.full((N, M), fill_value=1e6)

        predictions = [t["kf"].predict() for t in self.trackers]
        for i in range(N):
            for j in range(M):
                r_diff = predictions[i][0] - detections[j][0]
                d_diff = predictions[i][1] - detections[j][1]
                cost = np.sqrt(self.w_range * r_diff**2 + self.w_doppler * d_diff**2)
                cost_matrix[i, j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_tracker_idx, matched_detection_idx = set(), set()

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.max_cost:
                self.trackers[i]["kf"].update(detections[j])
                self.trackers[i]["lost"] = 0
                matched_tracker_idx.add(i)
                matched_detection_idx.add(j)

        for i in range(N):
            if i not in matched_tracker_idx:
                self.trackers[i]["lost"] += 1

        for j in range(M):
            if j not in matched_detection_idx:
                if self._is_really_new(detections[j], self.trackers, threshold=1.5):
                    self._create_tracker(detections[j])

        self._remove_lost()

    def _create_tracker(self, det):
        self.trackers.append({"id": self.next_id, "kf": SimpleKalman2D(det), "lost": 0})
        self.colors[self.next_id] = np.random.rand(3,)
        self.next_id += 1

    def _remove_lost(self):
        self.trackers = [t for t in self.trackers if t["lost"] <= self.max_lost]

    def _is_really_new(self, det, existing_trackers, threshold=1.5):
        for t in existing_trackers:
            pred = t["kf"].get_state()
            r_diff = det[0] - pred[0]
            d_diff = det[1] - pred[1]
            cost = np.sqrt(self.w_range * r_diff**2 + self.w_doppler * d_diff**2)
            if cost < threshold:
                return False
        return True

    def get_tracks(self):
        return [{"id": t["id"], "range": t["kf"].x[0, 0], "doppler": t["kf"].x[1, 0], "color": self.colors[t["id"]]} for t in self.trackers]


###########################
# CFAR + HDBSCAN Detection
###########################
def ca_cfar_2d(rdm, guard_cells=(2, 2), reference_cells=(5, 5), scale_factor=3.0):
    rows, cols = rdm.shape
    g_row, g_col = guard_cells
    r_row, r_col = reference_cells
    win_row = g_row + r_row
    win_col = g_col + r_col
    binary_map = np.zeros_like(rdm, dtype=np.uint8)

    for i in range(win_row, rows - win_row):
        for j in range(win_col, cols - win_col):
            r_start, r_end = i - win_row, i + win_row + 1
            c_start, c_end = j - win_col, j + win_col + 1
            cut = rdm[i, j]
            ref_window = rdm[r_start:r_end, c_start:c_end].copy()
            ref_window[r_row:-r_row, r_col:-r_col] = 0
            ref_cells = ref_window[ref_window > 0]
            if len(ref_cells) == 0:
                continue
            threshold = np.mean(ref_cells) * scale_factor
            if cut > threshold:
                binary_map[i, j] = 1
    return binary_map

def merge_nearby_detections(detections, merge_thresh=1.0):
    merged = []
    used = set()
    for i, a in enumerate(detections):
        if i in used:
            continue
        group = [a]
        for j, b in enumerate(detections):
            if j <= i or j in used:
                continue
            dist = np.linalg.norm(np.array(a) - np.array(b))
            if dist < merge_thresh:
                group.append(b)
                used.add(j)
        used.add(i)
        if len(group) > 1:
            merged.append(np.mean(group, axis=0))
        else:
            merged.append(a)
    return merged

def get_targets_from_rdm(rdm_f32, range_max=35, doppler_max=0.3):
    binary_map = ca_cfar_2d(rdm_f32)
    coords = np.column_stack(np.where(binary_map == 1))
    if len(coords) < 3:
        return []
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
    labels = clusterer.fit_predict(coords)

    h, w = binary_map.shape
    r_res = range_max / h
    d_res = doppler_max / w
    d_center = w // 2

    targets = []
    for cid in set(labels):
        if cid == -1:
            continue
        pts = coords[labels == cid]
        r_bin, d_bin = pts.mean(axis=0)
        real_range = r_bin * r_res
        real_speed = (d_bin - d_center) * d_res
        targets.append([real_range, real_speed])

    targets = merge_nearby_detections(targets)
    return targets, binary_map


###########################
# Main: Run Tracking + Plot
###########################
def run_tracking(file_path):
    with h5py.File(file_path, 'r') as f:
        rdi_data = f['DS1'][0]

    tracker = RadarTracker()
    range_max = 35
    doppler_max = 0.3

    # 初始化圖形視窗
    fig, ax = plt.subplots(figsize=(6, 4))

    for frame_id in range(rdi_data.shape[2]):
        rdm_f32 = np.array(rdi_data[:, :, frame_id], dtype=np.float32)
        detections, binary_map = get_targets_from_rdm(rdm_f32)
        tracker.update(detections)
        tracks = tracker.get_tracks()

        print(f"\n📦 Frame {frame_id} 偵測結果:")
        for t in tracks:
            print(f"  ➤ ID {t['id']} : Range={t['range']:.2f} cm, Speed={t['doppler']:.4f} cm/s")

        # 清除並重畫內容（不產生新視窗）
        ax.clear()
        h, w = binary_map.shape
        x = np.linspace(-doppler_max/2, doppler_max/2, w)
        y = np.linspace(0, range_max, h)
        ax.imshow(binary_map, extent=[x[0], x[-1], y[-1], y[0]], aspect='auto', cmap='gray')
        ax.set_xlabel('Velocity (cm/s)')
        ax.set_ylabel('Range (cm)')
        ax.set_title(f'Binary Map + Tracking Frame {frame_id}')

        for t in tracks:
            color = t['color']
            rect = patches.Rectangle((t['doppler'] - 0.01, t['range'] - 0.8), 0.02, 1.6,
                                     linewidth=1.5, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(t['doppler'], t['range'] - 1.2, f"ID {t['id']}", color=color, fontsize=8)

        plt.pause(0.1)

    plt.close()

run_tracking("hand.h5")