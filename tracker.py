
# import numpy as np
# from filterpy.kalman import KalmanFilter
# from scipy.spatial.distance import cdist
# from scipy.optimize import linear_sum_assignment
# import matplotlib.pyplot as plt

# TRACK_COLORS = [
#     'deepskyblue', 'orange', 'yellowgreen', 'magenta', 'aqua',
#     'gold', 'orchid', 'turquoise', 'lightcoral', 'plum'
# ]

# def bbox_iou(b1, b2):
#     x1, y1, w1, h1 = b1
#     x2, y2, w2, h2 = b2
#     xa = max(x1 - w1/2, x2 - w2/2)
#     ya = max(y1 - h1/2, y2 - h2/2)
#     xb = min(x1 + w1/2, x2 + w2/2)
#     yb = min(y1 + h1/2, y2 + h2/2)
#     inter = max(0, xb - xa) * max(0, yb - ya)
#     area1 = w1 * h1
#     area2 = w2 * h2
#     union = area1 + area2 - inter
#     return inter / union if union != 0 else 0

# class KalmanTracker:
#     def __init__(self, id, init_pos, init_bbox):
#         self.id = id
#         self.kf = KalmanFilter(dim_x=4, dim_z=2)
#         dt = 1.0
#         self.kf.F = np.array([[1, 0, dt, 0],
#                               [0, 1, 0, dt],
#                               [0, 0, 1, 0],
#                               [0, 0, 0, 1]])
#         self.kf.H = np.array([[1, 0, 0, 0],
#                               [0, 1, 0, 0]])
#         self.kf.R *= 10
#         self.kf.P *= 100
#         self.kf.Q *= 0.1
#         self.kf.x[:2] = np.reshape(init_pos, (2, 1))

#         self.age = 0
#         self.missed = 0
#         self.max_missed = 15
#         self.birth_frame = None
#         self.death_frame = None
#         self.total_visible = 0
#         self.predicted_pos = init_pos
#         self.latest_observation = None
#         self.latest_bbox = init_bbox
#         self.history = []

#     def predict(self):
#         self.kf.predict()
#         self.age += 1
#         self.missed += 1
#         self.predicted_pos = self.kf.x[:2].reshape(-1)
#         return self.predicted_pos

#     def update(self, measurement, bbox):
#         if measurement is None:
#             return
#         self.kf.update(measurement)
#         self.missed = 0
#         self.latest_observation = measurement
#         self.latest_bbox = bbox
#         self.history.append(self.kf.x[:2].reshape(-1))
#         self.total_visible += 1

#     def get_pos(self):
#         return self.kf.x[:2].reshape(-1)

#     def is_dead(self):
#         return self.missed > self.max_missed


# class MultiTargetTracker:
#     def __init__(self, match_threshold=12.0):
#         self.trackers = []
#         self.next_id = 1
#         self.log = []
#         self.match_threshold = match_threshold

#     def update(self, detections, frame_id):
#         if not detections:
#             for t in self.trackers:
#                 t.predict()
#             return

#         det_pos = [det[0] for det in detections]
#         predicted = [t.predict() for t in self.trackers]
#         matched_trackers = set()
#         matched_dets = set()

#         if predicted and det_pos:
#             D = cdist(predicted, det_pos)
#             row_ind, col_ind = linear_sum_assignment(D)
#             for t_idx, det_idx in zip(row_ind, col_ind):
#                 if D[t_idx, det_idx] > self.match_threshold:
#                     continue
#                 self.trackers[t_idx].update(det_pos[det_idx], detections[det_idx][1])
#                 matched_trackers.add(t_idx)
#                 matched_dets.add(det_idx)

#         MAX_DRIFT = 6.0
#         alive_trackers = []
#         for t in self.trackers:
#             drift = np.linalg.norm(t.predicted_pos - t.get_pos())
#             if t.is_dead() or drift > MAX_DRIFT:
#                 t.death_frame = frame_id
#                 self.log.append({
#                     'id': t.id,
#                     'birth': t.birth_frame,
#                     'death': t.death_frame,
#                     'total_visible': t.total_visible
#                 })
#             else:
#                 alive_trackers.append(t)

#         self.trackers = alive_trackers

#         unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
#         for i in unmatched_dets:
#             detection_pos = np.array(detections[i][0])
#             detection_bbox = detections[i][1]
#             too_close = any(np.linalg.norm(detection_pos - np.array(t.predicted_pos)) < 5.0 for t in self.trackers)
#             iou_overlap = any(bbox_iou(detection_bbox, t.latest_bbox) > 0.5 for t in self.trackers)
#             if too_close or iou_overlap:
#                 continue
#             new_tracker = KalmanTracker(self.next_id, detections[i][0], detections[i][1])
#             new_tracker.birth_frame = frame_id
#             self.trackers.append(new_tracker)
#             self.next_id += 1

#     def draw(self, ax):
#         for t in self.trackers:
#             color = TRACK_COLORS[(t.id - 1) % len(TRACK_COLORS)]
#             r_pred, d_pred = t.predicted_pos
#             h, w = t.latest_bbox[2], t.latest_bbox[3]
#             ax.add_patch(plt.Rectangle((d_pred - w/2, r_pred - h/2), w, h,
#                                        edgecolor='red', facecolor='none',
#                                        lw=2, linestyle='--'))
#             if t.latest_observation is not None:
#                 r_obs, d_obs = t.latest_observation
#                 h, w = t.latest_bbox[2], t.latest_bbox[3]
#                 ax.add_patch(plt.Rectangle((d_obs - w/2, r_obs - h/2), w, h,
#                                            edgecolor='lime', facecolor='none',
#                                            lw=2))

#             history = np.array(t.history[-20:])
#             num_points = len(history)
#             for i in range(1, num_points):
#                 alpha = i / num_points
#                 ax.plot(history[i-1:i+1,1], history[i-1:i+1,0],
#                         linestyle='--', linewidth=2,
#                         color=color, alpha=alpha)

#             id_x = d_pred - w/2 + 0.2
#             id_y = r_pred - h/2 + 0.8
#             ax.text(id_x, id_y, f'ID {t.id}',
#                     color=color, fontsize=14, weight='bold')
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# 彩色軌跡與 ID 顏色
TRACK_COLORS = [
    'deepskyblue', 'orange', 'yellowgreen', 'magenta', 'aqua',
    'gold', 'orchid', 'turquoise', 'lightcoral', 'plum'
]

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


    def update(self, detections, frame_id):
        det_pos = [det[0] for det in detections]
        predicted = [t.predict() for t in self.trackers]
        matched = set()
        unmatched_dets = list(range(len(detections)))

        if predicted and det_pos:
            D = cdist(predicted, det_pos)
            for t_idx, det_idx in zip(*np.where(D < 5.0)):
                if det_idx in matched: continue
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
            new_tracker = KalmanTracker(self.next_id, detections[i][0], detections[i][1])
            new_tracker.birth_frame = frame_id  # ⬅️ 加這行
            self.trackers.append(new_tracker)
            self.next_id += 1

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
