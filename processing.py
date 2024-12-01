import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import math
import cv2
import pandas as pd
from datetime import datetime, timedelta

# Load the model
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

def quaternion_to_rotation_matrix(w, x, y, z):
    """Convert quaternion to 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    return R

def create_video_df(metadata):
    duration_s = float(metadata['streams'][0]['duration'])
    numerator, denominator = map(float, metadata['streams'][0]['avg_frame_rate'].split('/'))
    fps = numerator / denominator

    creation_time = metadata['format']['tags']['creation_time']
    dt_time = datetime.strptime(creation_time, "%Y-%m-%dT%H:%M:%S.%fZ")
    dt_time = dt_time + timedelta(hours=2)  # Adjust for time zone if needed
    dt_time_sec = float(dt_time.timestamp())

    n_frames = int(metadata['streams'][0]['nb_frames'])
    timestamps = np.linspace(dt_time_sec, dt_time_sec + duration_s, n_frames)
    elap_time = np.linspace(0, duration_s, n_frames)

    vid_df = {
        "Frame": np.linspace(1, n_frames, n_frames),
        "Timestamp": timestamps,
        "Elapsed time (s)": elap_time
    }

    video_df = pd.DataFrame(vid_df)
    return video_df

def initial_IMU_reading(imu_data, quart_data):
    acceleration_df = imu_data
    quaternion_df = quart_data
    global_accelerations = []
    for i in range(len(quaternion_df)):
        # Get quaternion
        w, x, y, z = quaternion_df.iloc[i][3:]
        
        # Compute rotation matrix
        R = quaternion_to_rotation_matrix(w, x, y, z)
        
        # Get local acceleration
        a_local = acceleration_df.iloc[i][3:].values
        
        # Transform to global frame
        a_global = np.dot(R, a_local)

        elp = acceleration_df.iloc[i, 2]
        timestamp_s = datetime.strptime(acceleration_df.iloc[i, 1], "%Y-%m-%dT%H:%M:%S.%f").timestamp()
        a_global_time = np.append(a_global, [elp, timestamp_s])

        global_accelerations.append(a_global_time)
    global_acceleration_df = pd.DataFrame(global_accelerations, columns=['g_x', 'g_y', 'g_z', 'elapsed', 'timestamp_s'])
    return global_acceleration_df

def synch_IMU_video(global_acceleration_df, video_df):
    IMU_time_offset = 1.22
    max_acc_z = max(global_acceleration_df["g_z"])
    max_acc_z_ind = list(global_acceleration_df["g_z"]).index(max_acc_z)
    max_acc_time = global_acceleration_df['timestamp_s'][max_acc_z_ind] - IMU_time_offset

    closest_time_vid = min(video_df["Timestamp"], key=lambda date: abs(date - max_acc_time))

    video_start = video_df["Timestamp"][0]
    IMU_start = global_acceleration_df["timestamp_s"][0]

    if IMU_start >= video_start:
        closest_time_vid = min(video_df["Timestamp"], key=lambda date: abs(date - IMU_start))
    else:
        closest_time_vid = min(global_acceleration_df["timestamp_s"], key=lambda date: abs(date - video_start))

    if abs(closest_time_vid - IMU_start) < 1:
        synch_offset = IMU_start - video_start - 3.15
        synched = True
    elif abs(closest_time_vid - video_start) < 1:
        synch_offset = IMU_start - video_start
        synched = True
    else:
        synched = False
        synch_offset = None

    return synched, synch_offset

def pose_estimation(frame):
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 480, 800)
    input_img = tf.cast(img, dtype=tf.int32)
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
    return keypoints_with_scores

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    color_nodes = [
    (115, 121, 4), (115, 121, 4), (115, 121, 4), (115, 121, 4),
    (115, 121, 4), (115, 121, 4), (115, 121, 4), (115, 121, 4),
    (115, 121, 4), (115, 121, 4), (115, 121, 4), (15, 15, 122),
    (140, 99, 28), (15, 15, 122), (140, 99, 28), (15, 15, 122),
    (140, 99, 28)
    ]
    color_links = {
        (0, 1): (174, 175, 134), (0, 2): (174, 175, 134), (1, 3): (174, 175, 134),
        (2, 4): (174, 175, 134), (0, 5): (174, 175, 134), (0, 6): (174, 175, 134),
        (5, 7): (174, 175, 134), (7, 9): (174, 175, 134), (6, 8): (174, 175, 134),
        (8, 10): (174, 175, 134), (5, 6): (174, 175, 134), (5, 11): (174, 175, 134),
        (6, 12): (174, 175, 134), (11, 12): (174, 175, 134), (11, 13): (45, 45, 150),
        (13, 15): (45, 45, 150), (12, 14): (149, 113, 49), (14, 16): (149, 113, 49)
    }
    EDGES = {
        (0, 1): 'nose-leye', (0, 2): 'nose-reye', (1, 3): 'leye-lnose', (2, 4): 'reye-rnose',
        (0, 5): 'nose-lshoulder', (0, 6): 'nose-rshoulder', (5, 7): 'lshoulder-lelbow',
        (7, 9): 'lelbow-lwrist', (6, 8): 'rshoulder-relbow', (8, 10): 'relbow-rwrist',
        (5, 6): 'lshoulder-rshoulder', (5, 11): 'lshoulder-lhip', (6, 12): 'rshoulder-rhip',
        (11, 12): 'lhip-rhip', (11, 13): 'lhip-lknee', (13, 15): 'lknee-lankle',
        (12, 14): 'rhip-rknee', (14, 16): 'rknee-rankle'
    }

    # Draw edges
    for edge, color in EDGES.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) and (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_links[(p1, p2)], 4)

    # Draw keypoints
    for i, kp in enumerate(shaped):
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, color_nodes[i], -1)
            
    # Angle calculations
    def unit_vec(vector):
        return vector / np.linalg.norm(vector) if np.linalg.norm(vector) != 0 else vector

    right_knee_angle = None
    left_knee_angle = None

    # Right knee angle calculation
    # Right hip (12), right knee (14), right ankle (16)
    y1, x1, c1 = shaped[12]
    y2, x2, c2 = shaped[14]
    y3, x3, c3 = shaped[16]

    node1 = (x1, y1)
    node2 = (x2, y2)
    node3 = (x3, y3)

    vector1 = unit_vec((node1[0] - node2[0], node1[1] - node2[1]))
    vector2 = unit_vec((node3[0] - node2[0], node3[1] - node2[1]))

    angle1 = math.atan2(vector1[1], vector1[0]) * 180 / math.pi + 180
    angle2 = math.atan2(vector2[1], vector2[0]) * 180 / math.pi + 180

    start_angle = min(angle1, angle2)
    end_angle = max(angle1, angle2)
    res_angle = end_angle - start_angle
    if res_angle > 180:
        start_angle, end_angle = end_angle, start_angle
        res_angle = 360 - res_angle

    if start_angle > 180:
        start_angle -= 180
    else:
        start_angle += 180

    if end_angle > 180:
        end_angle -= 180
    else:
        end_angle += 180

    if start_angle > end_angle:
        start_angle -= 360

    right_knee_angle = res_angle

    # Draw the angle arc on the frame
    node2_int = (int(node2[0]), int(node2[1]))
    cv2.ellipse(frame, node2_int, (40, 40), 0, start_angle, end_angle, color_nodes[14], 3)


    # Left knee angle calculation
    
    # Left hip (11), left knee (13), left ankle (15)
    y1, x1, c1 = shaped[11]
    y2, x2, c2 = shaped[13]
    y3, x3, c3 = shaped[15]

    node1 = (x1, y1)
    node2 = (x2, y2)
    node3 = (x3, y3)

    vector1 = unit_vec((node1[0] - node2[0], node1[1] - node2[1]))
    vector2 = unit_vec((node3[0] - node2[0], node3[1] - node2[1]))

    angle1 = math.atan2(vector1[1], vector1[0]) * 180 / math.pi + 180
    angle2 = math.atan2(vector2[1], vector2[0]) * 180 / math.pi + 180

    start_angle = min(angle1, angle2)
    end_angle = max(angle1, angle2)
    res_angle = end_angle - start_angle
    if res_angle > 180:
        start_angle, end_angle = end_angle, start_angle
        res_angle = 360 - res_angle

    if start_angle > 180:
        start_angle -= 180
    else:
        start_angle += 180

    if end_angle > 180:
        end_angle -= 180
    else:
        end_angle += 180

    if start_angle > end_angle:
        start_angle -= 360

    left_knee_angle = res_angle

    # Draw the angle arc on the frame
    node2_int = (int(node2[0]), int(node2[1]))
    cv2.ellipse(frame, node2_int, (40, 40), 0, start_angle, end_angle, color_nodes[13], 3)

    return frame, right_knee_angle, left_knee_angle
