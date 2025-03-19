import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# Radar parameters
range_resolution = 0.6 / 32
angle_resolution = np.deg2rad(180 / 32)

# Load data
with h5py.File('rdi2.h5', 'r') as file:
    rdi_data = file['DS1'][0]
    phd_data = file['DS1'][1]

# Kalman Filter initialization
def create_kf(x, y):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([x, y, 0, 0])
    kf.F = np.array([[1, 0, 0.033, 0],
                     [0, 1, 0, 0.033],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 0.1
    kf.R *= 0.01
    return kf

# Target detection function
def detect_targets(rdi_frame, phd_frame, threshold_ratio=0.9):
    threshold = threshold_ratio * np.max(rdi_frame)
    target_bins = np.argwhere(rdi_frame >= threshold)
    targets = []

    for bin_idx in target_bins:
        range_bin = bin_idx[0]
        distance = range_bin * range_resolution
        angle_bin = np.argmax(phd_frame[range_bin])
        angle = (angle_bin - 16) * angle_resolution
        X = distance * np.sin(angle)
        Y = distance * np.cos(angle)

        targets.append((X, Y))

    return targets

# Initialize tracking
kfs = []
max_distance = 0.1
track_history = []
skipped_frames = []

for frame_idx in range(500):
    rdi_frame = rdi_data[:, :, frame_idx]
    phd_frame = phd_data[:, :, frame_idx]
    current_targets = detect_targets(rdi_frame, phd_frame)

    predicted_positions = []
    for kf in kfs:
        kf.predict()
        predicted_positions.append(kf.x[:2])

    assigned = set()

    if len(kfs) > 0 and len(current_targets) > 0:
        cost = np.linalg.norm(np.expand_dims(predicted_positions, 1) - np.expand_dims(current_targets, 0), axis=2)
        row_idx, col_idx = linear_sum_assignment(cost)

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < max_distance:
                kfs[r].update(current_targets[c])
                assigned.add(c)
                skipped_frames[r] = 0

    # Handle new tracks
    for i, target in enumerate(current_targets):
        if i not in assigned and len(kfs) < 3:
            kfs.append(create_kf(*target))
            skipped_frames.append(0)

    # Handle missed tracks
    for i in range(len(kfs)):
        if i not in assigned:
            skipped_frames[i] += 1

    # Delete tracks not updated
    for i in reversed(range(len(kfs))):
        if skipped_frames[i] > 3:
            del kfs[i]
            del skipped_frames[i]

    track_history.append([kf.x[:2].copy() for kf in kfs])

# Animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(0, 0.6)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.grid(True)

scatter = ax.scatter([], [], c='blue', s=40)

def animate(i):
    if track_history[i]:
        scatter.set_offsets(track_history[i])
    else:
        scatter.set_offsets(np.empty((0, 2)))  # 修正這一行

    ax.set_title(f'Frame: {i}')
    return scatter,


ani = animation.FuncAnimation(fig, animate, frames=len(track_history), interval=33, blit=True)

# Save video
ani.save('multi_target_tracking_kf.mp4', writer='ffmpeg', fps=30)

plt.close()
