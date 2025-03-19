import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt  # Use matplotlib for the 3D plot creation
from mpl_toolkits.mplot3d import Axes3D
import random

def create_3d_phd(phd_frame, radar_position, azimuth_range, elevation_range,
                 range_resolution, velocity_resolution, num_azimuth_bins, num_elevation_bins):
    """
    Converts a 2D PHD frame to a 3D PHD in Cartesian coordinates.
    """

    num_range_bins, num_velocity_bins = phd_frame.shape
    range_values = np.arange(num_range_bins) * range_resolution
    velocity_values = (np.arange(num_velocity_bins) - num_velocity_bins // 2) * velocity_resolution
    azimuth_values = np.linspace(azimuth_range[0], azimuth_range[1], num_azimuth_bins)
    elevation_values = np.linspace(elevation_range[0], elevation_range[1], num_elevation_bins)

    x_coords = []
    y_coords = []
    z_coords = []
    phd_values_list = []

    for i in range(num_range_bins):
        for j in range(num_velocity_bins):
            azimuth_index = i % num_azimuth_bins
            elevation_index = j % num_elevation_bins

            azimuth_rad = np.radians(azimuth_values[azimuth_index])
            elevation_rad = np.radians(elevation_values[elevation_index])

            r = range_values[i]
            x = radar_position[0] + r * np.cos(elevation_rad) * np.sin(azimuth_rad)
            y = radar_position[1] + r * np.cos(elevation_rad) * np.cos(azimuth_rad)
            z = radar_position[2] + r * np.sin(elevation_rad)

            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            phd_values_list.append(phd_frame[i, j])

    return np.array(x_coords), np.array(y_coords), np.array(z_coords), np.array(phd_values_list)

def track_3d_targets(phd_frames, radar_position, azimuth_range, elevation_range,
                     range_resolution, velocity_resolution, num_azimuth_bins, num_elevation_bins,
                     distance_threshold = 20, R = 5, Q_var_pos = 0.1, Q_var_vel = 0.1):
    """
    Tracks targets across a sequence of 3D PHD frames using a Kalman filter.
    """

    tracks = []
    kalman_filters = []
    previous_peaks = []

    for frame_index, phd_frame in enumerate(phd_frames):
        x, y, z, phd_values = create_3d_phd(phd_frame, radar_position, azimuth_range, elevation_range,
                                             range_resolution, velocity_resolution, num_azimuth_bins, num_elevation_bins)

        max_index = np.argmax(phd_values)
        peak_location = (x[max_index], y[max_index], z[max_index])

        if previous_peaks:
            distances = [np.sqrt((peak_location[0] - px)**2 + (peak_location[1] - py)**2 + (peak_location[2] - pz)**2) for px, py, pz in previous_peaks]
            min_distance_index = np.argmin(distances)
            if distances[min_distance_index] < distance_threshold:
                kf = kalman_filters[min_distance_index]
                kf.predict()
                kf.update(np.array([[peak_location[0]], [peak_location[1]], [peak_location[2]]]))
                smoothed_x, smoothed_y, smoothed_z = kf.x[0, 0], kf.x[1, 0], kf.x[2, 0]
                tracks[min_distance_index].append((frame_index, smoothed_x, smoothed_y, smoothed_z))
                previous_peaks[min_distance_index] = (smoothed_x, smoothed_y, smoothed_z)
            else:
                kf = KalmanFilter(dim_x=6, dim_z=3)
                kf.F = np.array([[1, 0, 0, 1, 0, 0],
                                 [0, 1, 0, 0, 1, 0],
                                 [0, 0, 1, 0, 0, 1],
                                 [0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1]])
                kf.H = np.array([[1, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0]])
                kf.P *= 1000.
                kf.R = R
                kf.Q = np.diag([Q_var_pos, Q_var_pos, Q_var_pos, 0, 0, 0])

                kf.x = np.array([[peak_location[0]], [peak_location[1]], [peak_location[2]], [0.], [0.], [0.]])

                kalman_filters.append(kf)
                smoothed_x, smoothed_y, smoothed_z = kf.x[0, 0], kf.x[1, 0], kf.x[2, 0]
                tracks.append([(frame_index, smoothed_x, smoothed_y, smoothed_z)])
                previous_peaks.append((smoothed_x, smoothed_y, smoothed_z))
        else:
            kf = KalmanFilter(dim_x=6, dim_z=3)
            kf.F = np.array([[1, 0, 0, 1, 0, 0],
                             [0, 1, 0, 0, 1, 0],
                             [0, 0, 1, 0, 0, 1],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1]])
            kf.H = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0]])
            kf.P *= 1000.
            kf.R = R
            kf.Q = np.diag([Q_var_pos, Q_var_pos, Q_var_pos, 0, 0, 0])

            kf.x = np.array([[peak_location[0]], [peak_location[1]], [peak_location[2]], [0.], [0.], [0.]])

            kalman_filters.append(kf)
            smoothed_x, smoothed_y, smoothed_z = kf.x[0, 0], kf.x[1, 0], kf.x[2, 0]
            tracks.append([(frame_index, smoothed_x, smoothed_y, smoothed_z)])
            previous_peaks.append((smoothed_x, smoothed_y, smoothed_z))
    return tracks

def visualize_tracks_opencv(phd_frames, radar_position, azimuth_range, elevation_range,
                     range_resolution, velocity_resolution, num_azimuth_bins, num_elevation_bins,
                     tracks, output_video_path="3d_tracking_animation.mp4", fps=10):
    """
    Visualizes the tracks in 3D space and creates a video using OpenCV.
    """
    # Set up the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (fig.canvas.get_width_height()))

    # Iterate through each frame
    for frame_index in range(len(phd_frames)):
        ax.clear()

        phd_frame = phd_frames[frame_index]
        x, y, z, phd_values = create_3d_phd(phd_frame, radar_position, azimuth_range, elevation_range,
                                             range_resolution, velocity_resolution, num_azimuth_bins, num_elevation_bins)

        scatter = ax.scatter(x, y, z, c=phd_values, cmap='viridis', marker='o', s=5, alpha=0.5)

        for track in tracks:
            x_coords = [t[1] for t in track if t[0] <= frame_index]
            y_coords = [t[2] for t in track if t[0] <= frame_index]
            z_coords = [t[3] for t in track if t[0] <= frame_index]
            ax.plot(x_coords, y_coords, z_coords, marker='o', linestyle='-', markersize=3, label='Track')

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D PHD Representation with Tracked Trajectory - Frame {}'.format(frame_index))
        ax.view_init(elev=30, azim=45)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:1], labels[:1], loc='upper right')

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        frame = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved to {output_video_path}")

if __name__ == '__main__':
    #--- 1. Load your 32x32x500 PHD data ---
    phd_data_path = "path/to/your/phd_data.npy"
    try:
        phd_frames = np.load(phd_data_path)
        if phd_frames.shape != (500, 32, 32):
            raise ValueError("PHD data has incorrect shape. Expected (500, 32, 32).")
        phd_frames = phd_frames.tolist()
        print("PHD data loaded successfully.")

    except FileNotFoundError:
        print(f"Error: PHD data file not found at {phd_data_path}")
        exit()
    except Exception as e:
        print(f"Error loading PHD data: {e}")
        exit()

    num_frames = len(phd_frames)
    num_range_bins = 32
    num_velocity_bins = 32

    #--- 2. Define Radar and Environment Parameters ---
    radar_position = (0, 0, 0)

    azimuth_range = (-90, 90)
    elevation_range = (-90, 90)

    num_azimuth_bins = 32
    num_elevation_bins = 32

    range_resolution = 5
    velocity_resolution = 1

    # Define the movement directions, these are applied to the object after
    # The initial position is found by create_3d_phd.
    possible_movements = [
        [0.5, 0, 0], # Move right
        [-0.5, 0, 0], # Move left
        [0, 0.5, 0], # Move up
        [0, -0.5, 0]  # Move down
    ]
    #This gives the scale of the movements.
    movement_persistence = 10

    for i in range(num_frames):

        #Object position in 3D, at init the range values
        x, y, z, phd_values = create_3d_phd(phd_frames[i], radar_position, azimuth_range, elevation_range,
                                            range_resolution, velocity_resolution, num_azimuth_bins, num_elevation_bins)

        # Find the peak by converting to 3D and finding the max
        max_index = np.argmax(phd_values)
        movement = possible_movements[i % len(possible_movements)] #Select direction based on what frame
        #Move the index
        new_x = x[max_index] + movement[0] * movement_persistence
        new_y = y[max_index] + movement[1] * movement_persistence
        new_z = z[max_index] + movement[2] * movement_persistence # keep at original range

        phd_frames[i] = np.zeros((num_range_bins, num_velocity_bins)) #Clear last pos
        range_value = np.linalg.norm(np.array([new_x,new_y,new_z]) - radar_position) #Range, distance in 3D coordinates.
        azimuth_value = np.degrees(np.arctan2(new_x, new_y)) #Angle in the XY plane
        elevation_value = np.degrees(np.arcsin(new_z / range_value)) if range_value != 0 else 0#The angle relative to the XY plane

        # Create 2D PHD frame
        range_index = int(range_value/5) #range res is 5. tune that
        velocity_index = int((movement[0])/1)+num_velocity_bins//2 #velocity res is 1 tune that

        if(0<=range_index<num_range_bins) and (0<=velocity_index<num_velocity_bins):
            phd_frames[i][range_index, velocity_index] = 10 #Make the object position

    #--- 3. Run the Tracking ---
    tracks = track_3d_targets(phd_frames, radar_position, azimuth_range, elevation_range,
                                 range_resolution, velocity_resolution, num_azimuth_bins, num_elevation_bins)

    #--- 4. Visualization ---
    visualize_tracks_opencv(phd_frames, radar_position, azimuth_range, elevation_range,
                     range_resolution, velocity_resolution, num_azimuth_bins, num_elevation_bins,
                     tracks)