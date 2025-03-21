import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# 讀取 .h5 檔案
def read_h5_data(file_path):
    with h5py.File(file_path, 'r') as file:
        rdi_data = file['DS1'][0]  # 取得 RDI 數據
    return rdi_data

# 設定存放圖片的資料夾
output_dir = 'rdi_framess'
os.makedirs(output_dir, exist_ok=True)

# 檔案路徑
file_path = "bin2.h5"

# 讀取 RDI 數據
rdi_data = read_h5_data(file_path)
num_frames = rdi_data.shape[2]

# 儲存每個 frame 圖像
for frame in range(num_frames):
    plt.figure(figsize=(6, 6))
    plt.imshow(rdi_data[:, :, frame], cmap='jet', aspect='auto', origin='upper')
    plt.axis('off')  # 不顯示任何座標軸和標籤
    plt.tight_layout(pad=0)
    
    # 儲存圖像
    frame_filename = os.path.join(output_dir, f'rdi_frame_{frame + 500:04d}.png')
    plt.savefig(frame_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

print(f"所有RDI圖已儲存至資料夾: {output_dir}")