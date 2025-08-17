import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Hilfsfunktion: 4x4-Transform aus Translation + RPY
def make_transform(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
    T[:3, 3] = xyz
    return T

# --- Basis-Posen aus URDF ---
T_world_left = make_transform(
    [0.3682, -0.1842, 0.7014],   # Position (m)
    [0.0039, -0.0030, -0.0161]   # RPY (rad)
)

T_world_right = make_transform(
    [0.3743,  0.1816, 0.7048],   # Position (m)
    [-0.0012, 0.0001, -0.0158]   # RPY (rad)
)

# --- Transformationen ---
T_left_right = np.linalg.inv(T_world_right) @ T_world_left
T_right_left = np.linalg.inv(T_left_right)

# --- CSV einlesen ---
df = pd.read_csv("cppflow/paths/flappy_bird.csv", skiprows=1, header=None)
df.columns = ['t', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']

# --- Transformation ---
rows_out = []
for _, row in df.iterrows():
    pos_left = np.array([row['x'], row['y'], row['z']])
    quat_left = np.array([row['qx'], row['qy'], row['qz'], row['qw']])
    rot_left = R.from_quat(quat_left).as_matrix()

    T_left_obj = np.eye(4)
    T_left_obj[:3, :3] = rot_left
    T_left_obj[:3, 3] = pos_left

    T_right_obj = T_right_left @ T_left_obj

    pos_right = T_right_obj[:3, 3]
    quat_right = R.from_matrix(T_right_obj[:3, :3]).as_quat()  # (qx, qy, qz, qw)

    rows_out.append([
        row['t'],
        pos_right[0], pos_right[1], pos_right[2],
        quat_right[3], quat_right[0], quat_right[1], quat_right[2]
    ])

# --- Speichern ---
df_out = pd.DataFrame(rows_out, columns=['t', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
# Dummy-Zeile oben einfügen
df_out.to_csv("cppflow/paths/flappy_bird_right.csv", index=False)
print("✅ Trajektorie in rechte Basis transformiert und gespeichert")
