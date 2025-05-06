# -*- coding: utf-8 -*-
"""
Created on Thu June 20 2024
Modified on Fri June 21 2024

@author: Ken Liu (Original), Claude (Modification)

Generates Range-Doppler maps and saves them as a video.
Axes are converted to physical units (cm and cm/s).
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os # For path joining

# --- Confirmed Radar Parameters ---
F_START = 57e9  # Start Frequency (Hz)
F_END = 64e9    # End Frequency (Hz)
FC = (F_START + F_END) / 2 # Center Frequency (Hz) ~60.5 GHz
BW = F_END - F_START      # Bandwidth (Hz) = 7 GHz
TC_UP = 102.4e-6          # Up-chirp sweep time (s)
# Assuming symmetrical triangular chirp and no gap
PRI = 2 * TC_UP           # Pulse Repetition Interval (s) ~204.8 us
FS = 625e3                # ADC Sampling Rate during Tc_up (Hz) = 625 kHz
NS_UP = 64                # Number of samples during up-chirp (used for FFT)
# chirp_num will be read from data, expected to be 32
# used_fft_num will be calculated, expected to be 32

C = 3e8                   # Speed of light (m/s)
LAMBDA = C / FC           # Wavelength (m)

# Calculated Performance Metrics
D_RES_M = C / (2 * BW)    # Range Resolution (m)
S = BW / TC_UP            # Chirp Slope (Hz/s)
# Max range is limited by Fs in this configuration
D_MAX_M = FS * C / (2 * S) # Max Unambiguous Range (m)

# Max unambiguous velocity based on PRI
# V_MAX_MPS = LAMBDA / (2 * PRI) # This is the full unambiguous velocity range width
# Doppler resolution
DELTA_FD = 1 / (32 * PRI) # Using chirp_num=32 read from data
DELTA_V_MPS = DELTA_FD * LAMBDA / 2 # Velocity resolution (m/s)

# --- Video Saving Setup ---
VIDEO_FILENAME = 'raw_data_rdi_video.mp4'
FPS = 10 # Frames per second for the video
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='RDI Video', artist='Matplotlib',
                comment='Radar Range-Doppler Plot')
writer = FFMpegWriter(fps=FPS, metadata=metadata)

# --- Main Code ---
if __name__ == '__main__':

    # Define input file path relative to script location for robustness
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'Test_pattern', 'hand.h5')

    try:
        raw = h5py.File(file_path, 'r')
    except FileNotFoundError:
        print(f"Error: HDF5 file not found at {file_path}")
        exit()
    except Exception as e:
        print(f"Error opening HDF5 file: {e}")
        exit()

    cubeRaw = np.transpose(raw['DS1'][:], (2, 1, 0, 3))

    # parameters read from data shape
    sample_num_raw = cubeRaw.shape[0] # Should be 128 (Up + Down)
    chirp_num = cubeRaw.shape[1]      # Should be 32
    antenna_num = cubeRaw.shape[2]    # Should be 2
    frame_num = cubeRaw.shape[3]      # Should be 500

    # Parameters for processing (based on confirmed understanding)
    up_sample_num = int(sample_num_raw * 0.5) # Use only the up-chirp part = 64
    if up_sample_num != NS_UP:
        print(f"Warning: Calculated up_sample_num ({up_sample_num}) does not match expected NS_UP ({NS_UP})")
        # Continue using calculated up_sample_num, but be aware

    used_fft_num = int(up_sample_num * 0.5) # Use half of range FFT bins = 32

    # Phase Compensate parameters
    try:
        rf_config = dict(raw['RF_CONFIG'].attrs)
        RX1_image_compansate = rf_config.get('RX1_image_compansate')
        RX1_real_compansate = rf_config.get('RX1_real_compansate')
        if RX1_image_compansate is None or RX1_real_compansate is None:
             raise KeyError("Compensation values not found in RF_CONFIG")
        print("Phase compensation parameters loaded.")
    except Exception as e:
         print(f"Warning: Could not load phase compensation parameters. Using 1+0j. Error: {e}")
         RX1_image_compansate = 0
         RX1_real_compansate = 1024 # Assuming compensation factor is applied relative to 1024

    # --- Calculate Axes for Plotting ---
    # Y-axis: Distance (cm)
    # Range bins correspond to frequencies from 0 Hz up to Fs/2 (max beat freq)
    # Max beat freq Fs/2 corresponds to dmax_m
    # Create 32 points from 0 to dmax_m
    distance_axis_m = np.linspace(0, D_MAX_M, used_fft_num, endpoint=True)
    distance_axis_cm = distance_axis_m * 100

    # X-axis: Velocity (cm/s)
    # Doppler FFT performed over chirp_num points. Frequency resolution is 1/(chirp_num * PRI)
    # Corresponding velocity for a frequency bin df is v = df * lambda / 2
    # np.fft.fftfreq gives frequencies from 0 to Fs/2 then -Fs/2 to 0 (where Fs=1/PRI for Doppler)
    doppler_freq_hz = np.fft.fftfreq(chirp_num, d=PRI) # Frequencies for each bin
    # Apply fftshift to center zero frequency
    doppler_freq_hz_shifted = np.fft.fftshift(doppler_freq_hz)
    # Convert frequency to velocity
    velocity_axis_mps = doppler_freq_hz_shifted * LAMBDA / 2
    velocity_axis_cmps = velocity_axis_mps * 100

    print(f"Calculated Parameters for Plotting:")
    print(f"  Distance axis (Y): 0 to {distance_axis_cm[-1]:.2f} cm")
    print(f"  Velocity axis (X): {velocity_axis_cmps[0]:.2f} to {velocity_axis_cmps[-1]:.2f} cm/s")
    print(f"  Saving video to: {VIDEO_FILENAME}")

    # --- Setup Plotting Figure ---
    fig, ax = plt.subplots(figsize=(10, 8)) # Create figure and axes ONCE

    # --- Processing and Video Saving Loop ---
    
    print("Scanning all frames to get global max value...")
    global_max = 0
    for idxFrame in range(frame_num):
        current_raw = cubeRaw[:up_sample_num, :, :, idxFrame]
        current_raw_antenna0 = current_raw[:, :, 0]
        fast_fft_matrix_tmp = np.fft.fft(current_raw_antenna0, up_sample_num, axis=0)
        compensate_factor = (RX1_real_compansate - 1j * RX1_image_compansate) / 1024.0
        fast_fft_matrix_tmp = fast_fft_matrix_tmp * compensate_factor
        fast_fft_matrix = fast_fft_matrix_tmp[:used_fft_num, :]
        rdi_map_complex = np.fft.fftshift(np.fft.fft(fast_fft_matrix, chirp_num, axis=1), axes=1)
        rdi_map = np.abs(rdi_map_complex)
        max_val = np.max(rdi_map)
        if max_val > global_max:
            global_max = max_val
    print(f"Global max magnitude for fixed color scale: {global_max:.2f}")

    print("Starting video generation...")
    with writer.saving(fig, VIDEO_FILENAME, dpi=100): # Use context manager
        for idxFrame in range(frame_num):
            if (idxFrame + 1) % 50 == 0: # Print progress every 50 frames
                 print(f"  Processing frame {idxFrame + 1} / {frame_num}")

            # Select only the up-chirp samples for the current frame
            current_raw = cubeRaw[:up_sample_num, :, :, idxFrame] # dim 64*32*2

            # Process only the first antenna for this example
            current_raw_antenna0 = current_raw[:, :, 0] # dim 64*32

            # Fast FFT (Range FFT)
            fast_fft_matrix_tmp = np.fft.fft(current_raw_antenna0, up_sample_num, axis=0) # dim 64*32

            # Phase compensate (using complex number representation)
            compensate_factor = (RX1_real_compansate - 1j * RX1_image_compansate) / 1024.0
            fast_fft_matrix_tmp = fast_fft_matrix_tmp * compensate_factor

            # Select positive frequency range bins
            fast_fft_matrix = fast_fft_matrix_tmp[:used_fft_num, :] # dim 32*32

            # Slow FFT (Doppler FFT) across chirps, then shift zero frequency to center
            rdi_map_complex = np.fft.fftshift(np.fft.fft(fast_fft_matrix, chirp_num, axis=1), axes=1) # dim 32*32

            # Take absolute value for intensity map
            rdi_map = np.abs(rdi_map_complex) # dim 32*32

            # --- Plotting Section (for video frame) ---
            # Clear the ENTIRE figure before drawing the new frame
            fig.clf()

            # Re-add the main axes to the figure for this frame
            ax = fig.add_subplot(111) # 111 means 1 row, 1 column, 1st subplot

            # Use pcolormesh with specified coordinates
            pcm = ax.pcolormesh(velocity_axis_cmps, distance_axis_cm, rdi_map, shading='auto', cmap='jet', vmin=0, vmax=global_max)

            ax.set_xlabel("Velocity (cm/s)")
            ax.set_ylabel("Distance (cm)")
            ax.set_title(f"Range-Doppler Map - Frame {idxFrame + 1}")
            ax.set_ylim(0, 60)
            ax.set_xlim(-0.15, 0.15)

            # Add the colorbar for the current frame (now it won't stack)
            fig.colorbar(pcm, ax=ax, label='Signal Magnitude (a.u.)')

            # Grab the current figure state as a video frame
            writer.grab_frame()

    print(f"Video generation complete. Saved as {VIDEO_FILENAME}")
    raw.close() # Close the HDF5 file
    print("HDF5 file closed.")