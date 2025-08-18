#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from klampt import WorldModel, vis
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from ikflow.config import DEVICE
import xml.etree.ElementTree as ET
from plan_dualarm_rrt import parse_object_urdf_for_offsets_and_mesh, WorldCollision, expand_q_to_full

set_seed(42)

# ---------------------------
# Transform-Helfer
# ---------------------------
def xyzrpy_to_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = np.asarray(xyz, float)
    return T

def T_to_posevec(T):
    p = T[:3, 3]
    q = R.from_matrix(T[:3, :3]).as_quat()
    return [float(p[0]), float(p[1]), float(p[2]), float(q[3]), float(q[0]), float(q[1]), float(q[2])]

def posevec_to_T(v):
    x, y, z, qw, qx, qy, qz = map(float, v)
    T = np.eye(4)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T

# ---------------------------
# Objekt-URDF Offsets
# ---------------------------
def parse_object_urdf_for_offsets(urdf_object_path, child_left="obj_p_01", child_right="obj_p_02", base="obj_com"):
    tree = ET.parse(urdf_object_path)
    root = tree.getroot()

    def find_origin_to(child_name):
        for joint in root.findall("joint"):
            child_elem = joint.find("child")
            if child_elem is not None and child_elem.get("link") == child_name:
                org = joint.find("origin")
                xyz = [0,0,0]; rpy=[0,0,0]
                if org is not None:
                    if org.get("xyz"): xyz = [float(s) for s in org.get("xyz").split()]
                    if org.get("rpy"): rpy = [float(s) for s in org.get("rpy").split()]
                return xyzrpy_to_T(xyz,rpy)
        raise RuntimeError(f"Kein Joint zu child '{child_name}' gefunden.")
    
    return find_origin_to(child_left), find_origin_to(child_right)



# ---------------------------
# Visualisierung
# ---------------------------
def visualize_pose(world, wc, ik_left, ik_right, T_obj, T_left_offset, T_right_offset):
    # Arme initial setzen
    left_pose = T_to_posevec(T_obj @ T_left_offset)
    right_pose = T_to_posevec(T_obj @ T_right_offset)

    okL, qL = batch_ik_and_filter(ik_left, [left_pose])[0]
    okR, qR = batch_ik_and_filter(ik_right, [right_pose])[0]

    if not (okL and okR):
        print("Keine gültige IK für diese Pose!")
        return

    wc.set_q(qL, qR)
    wc.set_object_T(T_obj)
    wc.view()



# ---------------------------
# IKFlow Batch (klein)
# ---------------------------
def batch_ik_and_filter(ik_solver, poses_batch):
    import torch
    if not poses_batch: return []
    poses_t = torch.tensor([p[:7] for p in poses_batch], dtype=torch.float32, device=DEVICE)
    sols, pos_err, rot_err, jl_exceeded, self_col, runtime = ik_solver.generate_ik_solutions(
        poses_t, n=1, refine_solutions=False, return_detailed=True
    )
    sols_np = sols.detach().cpu().numpy()
    jl = jl_exceeded.detach().cpu().numpy()
    sc = self_col.detach().cpu().numpy()
    out=[]
    for i in range(sols_np.shape[0]):
        ok = (not bool(jl[i])) and (not bool(sc[i]))
        out.append((ok, sols_np[i]))
    return out

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

    # Welt + Roboter laden
    world = WorldModel()
    assert world.readFile(args.urdf_left)
    assert world.readFile(args.urdf_right)

    T_left_off, T_right_off, mesh_abs = parse_object_urdf_for_offsets_and_mesh(args.urdf_object)

    # RigidObject (Mesh) laden
    obj_loaded = world.loadRigidObject(mesh_abs)

    # Prüfen, ob es ein Index oder schon ein RigidObjectModel ist
    if isinstance(obj_loaded, int):
        if obj_loaded < 0:
            raise RuntimeError(f"Objekt-Mesh konnte nicht geladen werden: {mesh_abs}")
        obj = world.rigidObject(obj_loaded)
    elif obj_loaded is None:
        raise RuntimeError(f"Objekt-Mesh konnte nicht geladen werden: {mesh_abs}")
    else:
        obj = obj_loaded  # schon ein RigidObjectModel

    print("RigidObjects:", world.numRigidObjects(), " -> obj =", obj)
    
    print("T_left_off:\n", T_left_off)
    print("T_right_off:\n", T_right_off)
    
    ik_left, _ = get_ik_solver(args.ik_left_model)
    ik_right, _ = get_ik_solver(args.ik_right_model)

    wc = WorldCollision(args.urdf_left, args.urdf_right, args.urdf_object)
    visualize_pose(world, wc, ik_left, ik_right, T_obj, T_left_off, T_right_off)


if __name__=="__main__":
    main()

"""
python scripts/debug_dual_arm_pose.py \
    --ik_left_model iiwa7_left_arm_0.75m \
    --ik_right_model iiwa7_right_arm_0.1m \
    --urdf_left urdfs/iiwa7_L/iiwa7_L_updated.urdf \
    --urdf_right urdfs/iiwa7_R/iiwa7_R_updated.urdf \
    --urdf_object urdfs/object/se3_object.urdf \
    --pose "0.45,0.00,1.75,1,0,0,0"

"""