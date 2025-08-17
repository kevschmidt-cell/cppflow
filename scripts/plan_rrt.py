import argparse
import numpy as np
import pandas as pd
import torch
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from ikflow.config import DEVICE
from scipy.spatial.transform import Rotation as R

set_seed()

# -----------------------------
# Beispiel-Kollisionscheck
# -----------------------------
def env_collision(joint_angles):
    """
    Dummy-Implementierung.
    Ersetze das hier mit deinem echten Collision-Checker (JRL, PyBullet, etc.).
    joint_angles: numpy array mit Gelenkwinkeln
    """
    return False  # aktuell keine Hindernisse


# -----------------------------
# Sampling im Cartesian Space
# -----------------------------
def sample_pose(bounds, fixed_orientation):
    """
    bounds: ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    Gibt eine zufällige Pose als [x, y, z, qw, qx, qy, qz] zurück.
    """
    pos = np.array([
        np.random.uniform(bounds[0][0], bounds[0][1]),
        np.random.uniform(bounds[1][0], bounds[1][1]),
        np.random.uniform(bounds[2][0], bounds[2][1])
    ])
    # Zufalls-Orientierung
    #quat = R.random().as_quat()  # [qx, qy, qz, qw]
    #qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
    return [pos[0], pos[1], pos[2]] + list(fixed_orientation)  # [x, y, z, qw, qx, qy, qz]


# -----------------------------
# RRT Planner
# -----------------------------
def rrt_plan(ik_solver, start_pose, goal_pose, bounds, max_iters=500, step_size=0.05, batch_size=64):
    tree = [start_pose]
    parents = {0: None}

    for it in range(max_iters):
        # mit einer kleinen Wahrscheinlichkeit direkt zum Ziel ziehen
        if np.random.rand() < 0.1:
            rand_pose = goal_pose
        else:
            rand_pose = sample_pose(bounds, fixed_orientation=start_pose[3:])

        # Nächsten Knoten im Baum finden
        dists = [np.linalg.norm(np.array(p[:3]) - np.array(rand_pose[:3])) for p in tree]
        nearest_idx = int(np.argmin(dists))
        nearest_pose = tree[nearest_idx]

        # Schritt in Richtung Zufallspose
        dir_vec = np.array(rand_pose[:3]) - np.array(nearest_pose[:3])
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm > step_size:
            dir_vec = dir_vec / dir_norm * step_size
        new_pos = np.array(nearest_pose[:3]) + dir_vec
        new_pose = list(new_pos) + nearest_pose[3:]  # Orientierung erstmal behalten

        # Batch sammeln
        if it % batch_size == 0:
            batch = [new_pose]
        else:
            batch.append(new_pose)

        # Batch-IK ausführen, wenn voll oder letzter Iterationsschritt
        if len(batch) >= batch_size or it == max_iters - 1:
            poses_tensor = torch.tensor(batch, device=DEVICE, dtype=torch.float32)

            sols, pos_err, rot_err, joint_limits, self_col, runtime = ik_solver.generate_ik_solutions(
                poses_tensor, n=1, refine_solutions=False, return_detailed=True
            )

            for pose, q, jl, sc in zip(batch, sols.cpu().numpy(), joint_limits, self_col):
                if jl or sc:
                    continue
                if env_collision(q):
                    continue
                # Pose in Baum einfügen
                tree.append(pose)
                parents[len(tree) - 1] = nearest_idx

                # Prüfen ob Ziel erreicht
                if np.linalg.norm(np.array(pose[:3]) - np.array(goal_pose[:3])) < step_size:
                    parents[len(tree) - 1] = nearest_idx
                    print(f"Ziel nach {it} Iterationen erreicht!")
                    return reconstruct_path(tree, parents, len(tree) - 1)

            batch = []

    print("Kein Pfad gefunden.")
    return []


# -----------------------------
# Pfad rekonstruieren
# -----------------------------
def reconstruct_path(tree, parents, goal_idx):
    path = []
    idx = goal_idx
    while idx is not None:
        path.append(tree[idx])
        idx = parents[idx]
    path.reverse()
    return path


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="planned_path.csv")
    args = parser.parse_args()

    # IKFlow laden
    ik_solver, hyper_parameters = get_ik_solver(args.model_name)

    # Start- und Zielpose definieren
    start_pose = [0.3, 0.0, 0.5, 1, 0, 0, 0]  # [x,y,z,qw,qx,qy,qz]
    goal_pose = [0.6, 0.2, 0.5, 1, 0, 0, 0]

    # Workspace-Bounds
    bounds = ((0.2, 0.7), (-0.3, 0.3), (0.3, 0.8))

    path = rrt_plan(ik_solver, start_pose, goal_pose, bounds)

    # CSV speichern
    if path:
        df = pd.DataFrame(
            [[i] + p for i, p in enumerate(path)],
            columns=["t", "x", "y", "z", "qw", "qx", "qy", "qz"]
        )
        df.to_csv(args.output_csv, index=False)
        print(f"Pfad in {args.output_csv} gespeichert.")

