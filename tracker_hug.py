import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# 彩色軌跡與 ID 顏色
TRACK_COLORS = [
    'deepskyblue', 'orange', 'yellowgreen', 'magenta', 'aqua',
    'gold', 'orchid', 'turquoise', 'lightcoral', 'plum'
]

class TentativeTrack:
    def __init__(self, init_pos, init_bbox):
        self.init_pos = init_pos
        self.init_bbox = init_bbox
        self.age = 1  # 第1次出現
        self.max_age = 15  # 若超過這個年齡還未被確認，就移除
        self.required_hits = 5  # 連續出現5次才變正式track
        self.matched = True  # 本幀被配對到

    def increment_age(self):
        self.age += 1
        self.matched = False

    def confirm(self):
        return self.age >= self.required_hits

class KalmanTracker:
    def __init__(self, id, init_pos, init_bbox):
        self.id = id
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0

        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])

        self.kf.R *= 10
        self.kf.P *= 100
        self.kf.Q *= 0.1

        self.kf.x[:2] = np.reshape(init_pos, (2, 1))
        self.age = 0
        self.missed = 0
        self.max_missed = 10

        self.history = [init_pos]
        self.predicted_pos = init_pos
        self.latest_observation = None
        self.latest_bbox = init_bbox
        self.total_visible = 1  # 初始化為 1，因為剛建立時被觀測到

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.missed += 1
        self.predicted_pos = self.kf.x[:2].reshape(-1)
        self.history.append(self.predicted_pos)
        return self.predicted_pos

    def update(self, measurement, bbox):
        self.kf.update(measurement)
        self.missed = 0
        self.latest_observation = measurement
        self.latest_bbox = bbox
        self.history[-1] = self.kf.x[:2].reshape(-1)
        self.total_visible += 1

    def get_pos(self):
        return self.kf.x[:2].reshape(-1)

    def is_dead(self):
        return self.missed > self.max_missed


class MultiTargetTracker:
    def __init__(self):
        self.trackers = []
        self.next_id = 1
        self.log = []  # 儲存所有目標的出生與離場記錄
        self.tentatives = []  # 暫存候選軌跡


    def update(self, detections, frame_id):
        det_pos = [det[0] for det in detections]
        predicted = [t.predict() for t in self.trackers]
        matched = set()
        unmatched_dets = list(range(len(detections)))

        if predicted and det_pos:
            D = cdist(predicted, det_pos)
            row_ind, col_ind = linear_sum_assignment(D)
            matched = set()
            unmatched_dets = list(range(len(detections)))
            
            for t_idx, det_idx in zip(row_ind, col_ind):
                if D[t_idx, det_idx] > 5.0:  # 可選擇是否加距離門檻
                    continue
                self.trackers[t_idx].update(det_pos[det_idx], detections[det_idx][1])
                matched.add(det_idx)
                if det_idx in unmatched_dets:
                    unmatched_dets.remove(det_idx)
        # 找出要刪除的 tracker（missed 太多幀）
        dead_trackers = []
        for t in self.trackers:
            if t.is_dead():
                t.death_frame = frame_id
                self.log.append({
                    'id': t.id,
                    'birth': t.birth_frame,
                    'death': t.death_frame,
                    'total_visible': t.total_visible
                })
                dead_trackers.append(t)
        self.trackers = [t for t in self.trackers if not t.is_dead()]

        for i in unmatched_dets:
            det_pos_i = detections[i][0]
            det_bbox_i = detections[i][1]

            # 嘗試與現有候選軌跡比對（距離小於門檻就累積）
            matched = False
            for t in self.tentatives:
                if np.linalg.norm(np.array(t.init_pos) - np.array(det_pos_i)) < 5.0:
                    t.age += 1
                    t.init_pos = det_pos_i  # 更新位置
                    t.init_bbox = det_bbox_i
                    t.matched = True
                    matched = True
                    break

            # 若沒有匹配任何候選軌跡，就新增一個新的候選
            if not matched:
                self.tentatives.append(TentativeTrack(det_pos_i, det_bbox_i))

        # 清除老化的候選軌跡或轉為正式 Kalman tracker
        new_tentatives = []
        for t in self.tentatives:
            if t.matched:
                if t.confirm():
                    new_tracker = KalmanTracker(self.next_id, t.init_pos, t.init_bbox)
                    new_tracker.birth_frame = frame_id
                    self.trackers.append(new_tracker)
                    self.next_id += 1
                else:
                    new_tentatives.append(t)
            else:
                t.increment_age()
                if t.age < t.max_age:
                    new_tentatives.append(t)
        self.tentatives = new_tentatives

    def draw(self, ax):
        for t in self.trackers:
            color = TRACK_COLORS[(t.id - 1) % len(TRACK_COLORS)]
            r_pred, d_pred = t.predicted_pos
            h, w = t.latest_bbox[2], t.latest_bbox[3]

            # 🔴 預測框 - 紅色虛線
            ax.add_patch(plt.Rectangle((d_pred - w/2, r_pred - h/2), w, h,
                                       edgecolor='red', facecolor='none',
                                       lw=2, linestyle='--'))

            # 🟢 觀測框 - 綠色實線
            if t.latest_observation is not None:
                r_obs, d_obs = t.latest_observation
                h, w = t.latest_bbox[2], t.latest_bbox[3]
                ax.add_patch(plt.Rectangle((d_obs - w/2, r_obs - h/2), w, h,
                                           edgecolor='lime', facecolor='none',
                                           lw=2))

            # 軌跡（淡出）
            history = np.array(t.history[-20:])
            num_points = len(history)
            for i in range(1, num_points):
                alpha = i / num_points
                ax.plot(history[i-1:i+1,1], history[i-1:i+1,0],
                        linestyle='--', linewidth=2,
                        color=color, alpha=alpha)

            # ID 顯示
            id_x = d_pred - w/2 + 0.2
            id_y = r_pred - h/2 + 0.8
            ax.text(id_x, id_y, f'ID {t.id}',
                    color=color, fontsize=14, weight='bold')
