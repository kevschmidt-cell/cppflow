#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from klampt import WorldModel, vis
from klampt.model import collide
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from ikflow.config import DEVICE
import torch
from plan_dualarm_rrt import parse_object_urdf_for_offsets_and_mesh
set_seed(42)

# ---------------------------
# Transform-Helfer
# ---------------------------
def xyzrpy_to_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = np.asarray(xyz, float)
    return T

def draw_frame(T, name, size=0.05):
    origin = T[:3,3].tolist()
    x_dir = (T[:3,3] + T[:3,0]*size).tolist()
    y_dir = (T[:3,3] + T[:3,1]*size).tolist()
    z_dir = (T[:3,3] + T[:3,2]*size).tolist()
    vis.add("frame_"+name+"_x", [origin, x_dir], color=(1,0,0,1))
    vis.add("frame_"+name+"_y", [origin, y_dir], color=(0,1,0,1))
    vis.add("frame_"+name+"_z", [origin, z_dir], color=(0,0,1,1))

def T_to_posevec(T):
    """
    Konvertiert eine 4x4-Transformationsmatrix in [x,y,z,qw,qx,qy,qz].
    """
    p = T[:3, 3]
    qx, qy, qz, qw = R.from_matrix(T[:3, :3]).as_quat()  # scipy gibt (x,y,z,w) zurück
    return [float(p[0]), float(p[1]), float(p[2]), float(qw), float(qx), float(qy), float(qz)]

def posevec_to_T(v):
    """
    Konvertiert [x,y,z,qw,qx,qy,qz] zurück in eine 4x4-Transformationsmatrix.
    """
    x, y, z, qw, qx, qy, qz = map(float, v)
    T = np.eye(4)
    # scipy erwartet (x,y,z,w)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T
def expand_q_to_full(robot, q_partial):
    """
    Baut aus einer Teilkonfiguration (nur bewegliche Gelenke)
    eine volle Robot-Konfiguration inkl. Fixed Joints.
    """
    full_config = robot.getConfig()[:]   # aktuelle volle Konfiguration als Basis
    assert len(q_partial) == robot.numDrivers(), \
        f"q_partial hat {len(q_partial)} Werte, erwartet {robot.numDrivers()}"

    for i in range(robot.numDrivers()):
        drv = robot.driver(i)
        full_config[drv.getAffectedLink()] = float(q_partial[i])

    return full_config

# ---------------------------
# Batch IK
# ---------------------------
def batch_ik_and_filter(ik_solver, poses_batch):
    if not poses_batch:
        return []
    poses_t = torch.tensor([p[:7] for p in poses_batch], dtype=torch.float32, device=DEVICE)
    sols, pos_err, rot_err, jl_exceeded, self_col, runtime = ik_solver.generate_ik_solutions(
        poses_t, n=1, refine_solutions=False, return_detailed=True
    )
    sols_np = sols.detach().cpu().numpy()
    jl = jl_exceeded.detach().cpu().numpy()
    sc = self_col.detach().cpu().numpy()

    out = []
    for i in range(sols_np.shape[0]):
        ok = (not bool(jl[i])) and (not bool(sc[i]))
        out.append((ok, sols_np[i]))
    return out

# ---------------------------
# Batch IK: mehrere Lösungen
# ---------------------------
def batch_ik_and_filter_multi(ik_solver, poses_batch, n_solutions=2):
    if not poses_batch:
        return []

    poses_t = torch.tensor([p[:7] for p in poses_batch], dtype=torch.float32, device=DEVICE)
    sols, pos_err, rot_err, jl_exceeded, self_col, runtime = ik_solver.generate_ik_solutions(
        poses_t, n=n_solutions, refine_solutions=False, return_detailed=True
    )

    sols_np = sols.detach().cpu().numpy()
    jl = jl_exceeded.detach().cpu().numpy()
    sc = self_col.detach().cpu().numpy()

    # Falls nur (n_solutions, dof) zurückkommt → Batch-Dimension hinzufügen
    if sols_np.ndim == 2:
        sols_np = sols_np[None, :, :]  # (1, n_solutions, dof)

    out = []
    for i in range(sols_np.shape[0]):  # batch
        valid_sols = []
        for j in range(sols_np.shape[1]):  # n_solutions
            ok = True
            if jl.ndim == 1:
                if jl[i]:
                    ok = False
            else:
                if jl[i, j]:
                    ok = False
            if sc.ndim == 1:
                if sc[i]:
                    ok = False
            else:
                if sc[i, j]:
                    ok = False
            if ok:
                q_sol = np.array(sols_np[i, j], dtype=float).ravel()
                valid_sols.append(q_sol)
        out.append(valid_sols)
    return out



# ---------------------------
# Weltklasse
# ---------------------------
class WorldCollision:
    def __init__(self, urdf_left, urdf_right, obj_mesh_path):
        self.world = WorldModel()
        assert self.world.readFile(urdf_left)
        assert self.world.readFile(urdf_right)

        ok = self.world.loadRigidObject(obj_mesh_path.replace(".urdf", ".stl"))
        if not ok: raise RuntimeError(f"Objekt-Mesh konnte nicht geladen werden: {obj_mesh_path}")

        self.robot_L = self.world.robot(0)
        self.robot_R = self.world.robot(1)
        self.obj = self.world.rigidObject(0)
        self.collider = collide.WorldCollider(self.world)

    def set_q(self, qL, qR):
        qL_full = expand_q_to_full(self.robot_L, qL)
        qR_full = expand_q_to_full(self.robot_R, qR)
        self.robot_L.setConfig(qL_full)
        self.robot_R.setConfig(qR_full)
        print("qL_partial:", qL)
        print("qL_full:", qL_full)

    def set_object_T(self, T_world_obj):
        R_flat = T_world_obj[:3, :3].T.flatten().tolist()
        t_list = T_world_obj[:3, 3].tolist()
        self.obj.setTransform(R_flat, t_list)

    def view(self):
        vis.add("world", self.world)
        vis.setAttribute("world", "drawTransparency", 0.3)
        vis.setAttribute("world", "drawContacts", True)
        vis.show()
        vis.loop()

# ---------------------------
# Visualisierung der Pose
# ---------------------------
from klampt import vis

def setup_visualization(world):
    vis.add("world", world)
    vis.show()

def visualize_pose2(wc, ik_left, ik_right, T_obj, T_left_offset, T_right_offset, n_ik_solutions=10000):
    # Ziel-Trafos
    T_left_global = T_obj @ T_left_offset
    T_right_global = T_obj @ T_right_offset

    left_pose_vec = T_to_posevec(T_left_global)
    right_pose_vec = T_to_posevec(T_right_global)

    # Multi-IK
    left_sols = batch_ik_and_filter_multi(ik_left, [left_pose_vec], n_solutions=n_ik_solutions)[0]
    right_sols = batch_ik_and_filter_multi(ik_right, [right_pose_vec], n_solutions=n_ik_solutions)[0]

    if not left_sols or not right_sols:
        print("⚠️ Keine gültige IK gefunden")
        return

    # Hilfsfunktion zur Distanzmessung
    def pose_distance(q, T_target, robot, ee_link_idx):
        robot.setConfig(expand_q_to_full(robot, q))
        pos = robot.link(ee_link_idx).getWorldPosition([0, 0, 0])
        return np.linalg.norm(np.array(pos) - T_target[:3, 3])

    # Index Endeffektoren (angenommen: letzter Link)
    ee_left_idx = wc.robot_L.numLinks() 
    ee_right_idx = wc.robot_R.numLinks() 

    # Beste Lösung auswählen
    best_left = min(left_sols, key=lambda q: pose_distance(q, T_left_global, wc.robot_L, ee_left_idx))
    best_right = min(right_sols, key=lambda q: pose_distance(q, T_right_global, wc.robot_R, ee_right_idx))

    # Konfiguration setzen
    wc.set_q(best_left, best_right)
    wc.set_object_T(T_obj)

    # Frames
    draw_frame(T_obj, "obj_com", size=0.1)
    draw_frame(T_left_global, "left_ee_target", size=0.05)
    draw_frame(T_right_global, "right_ee_target", size=0.05)

    wc.view()



def visualize_pose(wc, ik_left, ik_right, T_obj, T_left_offset, T_right_offset):
    # Greifpunkte global
    T_left_global = T_obj @ T_left_offset
    T_right_global = T_obj @ T_right_offset

    # Basis-Trafos
    T_base_left = np.eye(4)
    T_base_left[:3, 3] = wc.robot_L.link(0).getWorldPosition([0,0,0])
    T_base_right = np.eye(4)
    T_base_right[:3, 3] = wc.robot_R.link(0).getWorldPosition([0,0,0])

    # Zielpose relativ zur Roboterbasis
    T_left_rel = T_base_left @ T_left_global
    T_right_rel = T_base_right @ T_right_global

    left_pose_vec = T_to_posevec(T_left_global)
    right_pose_vec = T_to_posevec(T_right_global)

    okL, qL_partial = batch_ik_and_filter(ik_left, [left_pose_vec])[0]
    okR, qR_partial = batch_ik_and_filter(ik_right, [right_pose_vec])[0]

    if not (okL and okR):
        print("⚠️ Keine gültige IK")
        return

    # Vollständige Konfigurationen setzen
    wc.set_q(qL_partial, qR_partial)
    wc.set_object_T(T_obj)

    # Frames visualisieren
    draw_frame(T_obj, "obj_com", size=0.1)
    draw_frame(T_left_global, "left_ee_target", size=0.05)
    draw_frame(T_right_global, "right_ee_target", size=0.05)

    wc.view()




# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ik_left_model", required=True)
    parser.add_argument("--ik_right_model", required=True)
    parser.add_argument("--urdf_left", required=True)
    parser.add_argument("--urdf_right", required=True)
    parser.add_argument("--urdf_object", required=True)
    parser.add_argument("--pose", required=True, help="x,y,z,qw,qx,qy,qz")
    args = parser.parse_args()

    pose_vals = [float(x) for x in args.pose.split(",")]
    T_obj = posevec_to_T(pose_vals)

    # Offsets und Mesh
    T_left_off, T_right_off, mesh_abs = parse_object_urdf_for_offsets_and_mesh(args.urdf_object)

    # IK Solver laden
    ik_left, _ = get_ik_solver(args.ik_left_model)
    ik_right, _ = get_ik_solver(args.ik_right_model)

    # Welt initialisieren
    wc = WorldCollision(args.urdf_left, args.urdf_right, args.urdf_object)

    visualize_pose2(wc, ik_left, ik_right, T_obj, T_left_off, T_right_off)

if __name__=="__main__":
    main()



"""
python scripts/debug_dual_arm_pose.py \
    --ik_left_model iiwa7_left_arm_0.75m \
    --ik_right_model iiwa7_right_arm_0.1m \
    --urdf_left urdfs/iiwa7_L/iiwa7_L_updated.urdf \
    --urdf_right urdfs/iiwa7_R/iiwa7_R_updated.urdf \
    --urdf_object urdfs/object/se3_object.urdf \
    --pose "1,0.0,0.9,1,0,0,0"

"""