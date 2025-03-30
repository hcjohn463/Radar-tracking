import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 雷達參數（已確認）
range_resolution = 0.6 / 32  # 1.875 cm per bin
angle_resolution = np.deg2rad(180 / 32)  # -90°到+90°

# 讀取資料函式
def read_h5_data(file_path):
    with h5py.File(file_path, 'r') as file:
        rdi_data = file['DS1'][0]
        phd_data = file['DS1'][1]
    return rdi_data, phd_data

# 計算物體位置函式 (X:左右, Y:前方距離)
def calculate_positions(rdi_data, phd_data, frame_indices):
    positions = []
    for idx in frame_indices:
        # RDI計算距離
        rdi_frame = rdi_data[:, :, idx]
        rdi_max_idx = np.unravel_index(np.argmax(rdi_frame), rdi_frame.shape)
        distance_bin_rdi = rdi_max_idx[0]
        distance_rdi = distance_bin_rdi * range_resolution

        # PHD計算角度
        phd_frame = phd_data[:, :, idx]
        phd_max_idx = np.unravel_index(np.argmax(phd_frame), phd_frame.shape)
        distance_bin_phd, angle_bin = phd_max_idx
        distance_phd = distance_bin_phd * range_resolution

        # 若RDI與PHD距離差異過大，以RDI為主（通常RDI較可靠）
        if abs(distance_phd - distance_rdi) > range_resolution:
            distance = distance_rdi
        else:
            distance = (distance_rdi + distance_phd) / 2

        # 計算角度（角度從-90°到+90°）
        angle = (angle_bin - 16) * angle_resolution

        # 實際位置
        X = distance * np.sin(angle)  # 左右
        Y = distance * np.cos(angle)  # 正前方距離

        positions.append((idx, X, Y, distance, np.rad2deg(angle)))

    return positions

# 主程式
if __name__ == "__main__":
    file_path = 'rdi1.h5'  # 替換你的檔案路徑
    rdi_data, phd_data = read_h5_data(file_path)

    # 計算所有frame的位置
    all_frames = range(500)
    all_positions = calculate_positions(rdi_data, phd_data, all_frames)

    # 提取座標
    X = [pos[1] for pos in all_positions]
    Y = [pos[2] for pos in all_positions]

    # 動畫繪製
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True)

    scatter, = ax.plot([], [], 'ro', markersize=8)

    def animate(i):
        scatter.set_data(X[i], Y[i])
        ax.set_title(f'Frame: {i}, X: {X[i]:.3f}m, Y: {Y[i]:.3f}m')
        return scatter,

    ani = animation.FuncAnimation(fig, animate, frames=len(all_frames), interval=1000/30)

    # 儲存影片
    video_path = 'radar_positions.mp4'
    ani.save(video_path, writer='ffmpeg', fps=30)

    plt.close()
    # frames = [0, 100, 250, 400, 499]

    # rdi_data, phd_data = read_h5_data(file_path)
    # positions = calculate_positions(rdi_data, phd_data, frames)

    # print("Frame |    X(m)    |    Y(m)    | Distance(m) | Angle(degree)")
    # print("--------------------------------------------------------------")
    # for idx, X, Y, distance, angle in positions:
    #     print(f"{idx:<5} | {X:>9.3f} | {Y:>9.3f} | {distance:>11.3f} | {angle:>13.2f}°")
