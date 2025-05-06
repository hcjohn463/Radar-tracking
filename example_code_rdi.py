
# -*- coding: utf-8 -*-
"""
Created by ChatGPT - Generate RDI video with velocity axis
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# Radar constants
fc = 60e9  # center frequency (Hz)
c = 3e8  # speed of light (m/s)
lambda_radar = c / fc  # wavelength (m)
Tc = 102.4e-6  # chirp duration (s)
chirp_num = 32

# Velocity axis calculation (in cm/s)
v_max = (lambda_radar / (4 * Tc)) * 100  # max velocity in cm/s
velocity_axis = np.linspace(-v_max, v_max, chirp_num)

if __name__ == '__main__':
    raw = h5py.File('./Test_pattern/hand.h5', 'r')
    cubeRaw = np.transpose(raw['DS1'][:], (2, 1, 0, 3))  # shape: 128 × 32 × 2 × 500

    sample_num = cubeRaw.shape[0]
    chirp_num = cubeRaw.shape[1]
    frame_num = cubeRaw.shape[3]

    fig, ax = plt.subplots()
    writer = FFMpegWriter(fps=10)

    with writer.saving(fig, "test_raw_rdi.mp4", dpi=100):
        for idxFrame in range(frame_num):
            current_raw = cubeRaw[:, :, :, idxFrame]  # shape: 128 × 32 × 2
            complex_cube = current_raw[:, :, 0] + 1j * current_raw[:, :, 1]  # shape: 128 × 32

            # Range FFT
            range_fft = np.fft.fft(complex_cube, axis=0)
            range_fft = range_fft[:64, :]  # take half (Nyquist limit)

            # Doppler FFT
            doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=1), axes=1)
            rdm = np.abs(doppler_fft)

            ax.clear()
            ax.pcolormesh(velocity_axis, np.arange(64), rdm)
            ax.set_xlabel("Velocity (cm/s)")
            ax.set_ylabel("Range Bin")
            ax.set_title(f"Frame {idxFrame + 1}/{frame_num}")
            writer.grab_frame()
