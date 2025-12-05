#!/usr/bin/env python3
"""
debug_dual_arm_pose_fixed.py

Usage example:
python scripts/debug_dual_arm_pose_fixed.py \
    --urdf_left urdfs/iiwa7_L/iiwa7_L_updated.urdf \
    --urdf_right urdfs/iiwa7_R/iiwa7_R_updated.urdf \
    --urdf_object urdfs/object/se3_object.urdf \
    --pose "1,0.0,0.9,1,0,0,0"
"""
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from klampt import WorldModel, vis
from klampt.model import collide, ik
from klampt.robotsim import IKSolver
import torch
from ikflow.utils import set_seed
from ikflow.config import DEVICE
from plan_dualarm_rrt import parse_object_urdf_for_offsets_and_mesh
from klampt.math import so3
from klampt.model.ik import IKSolver, objective

set_seed(42)

# Try to import your jrl Robot class (preferred IK backend)
try:
    from jrl.robot import Robot as JRLRobot  # adjust import if your path differs
    HAVE_JRL = True
except Exception:
    JRLRobot = None
    HAVE_JRL = False

# ---------------------------
# Helper: transforms and utils
# ---------------------------
def posevec_to_T(v):
    x, y, z, qw, qx, qy, qz = map(float, v)
    T = np.eye(4)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T

def T_to_posevec(T):
    p = T[:3,3]
    qx, qy, qz, qw = R.from_matrix(T[:3,:3]).as_quat()
    return [float(p[0]), float(p[1]), float(p[2]), float(qw), float(qx), float(qy), float(qz)]

def draw_frame(T, name, size=0.05):
    origin = T[:3,3].tolist()
    x_dir = (T[:3,3] + T[:3,0]*size).tolist()
    y_dir = (T[:3,3] + T[:3,1]*size).tolist()
    z_dir = (T[:3,3] + T[:3,2]*size).tolist()
    vis.add("frame_"+name+"_x", [origin, x_dir], color=(1,0,0,1))
    vis.add("frame_"+name+"_y", [origin, y_dir], color=(0,1,0,1))
    vis.add("frame_"+name+"_z", [origin, z_dir], color=(0,0,1,1))

def expand_q_to_full_klampt(robot_model, q_partial):
    """Klampt: Teilkonfiguration -> volle Konfiguration (inkl. fixed joints)."""
    full = robot_model.getConfig()[:]
    # In vielen URDF->Klampt-Setups sind die 'drivers' contiguous; fallbacks:
    if len(q_partial) == robot_model.numDrivers():
        # set by drivers
        for i in range(robot_model.numDrivers()):
            drv = robot_model.driver(i)
            link_idx = drv.getAffectedLink()
            full[link_idx] = float(q_partial[i])
        return full
    # fallback: if ndof equals numDrivers or direct mapping
    if len(q_partial) <= len(full):
        full[:len(q_partial)] = [float(x) for x in q_partial]
        return full
    raise RuntimeError("expand_q_to_full_klampt: unexpected q length")

# ---------------------------
# IK wrappers
# ---------------------------
def klampt_ik_solver(klampt_robot_model, ee_link_idx, pose_vec, positional_tolerance=1e-3, n_tries=50):
    """
    Führt numerische IK aus und gibt q_partial oder None zurück.
    pose_vec: [x,y,z,qw,qx,qy,qz]
    """
    assert len(pose_vec) == 7
    t = pose_vec[0:3]
    qw, qx, qy, qz = pose_vec[3:]
    Robj = so3.from_quaternion([qx, qy, qz, qw])

    # LINK statt Index !!
    ee_link = klampt_robot_model.link(ee_link_idx)

    # IK objective
    obj = ik.objective(ee_link, t=t, R=Robj)

    for _ in range(n_tries):
        solver = IKSolver(klampt_robot_model)
        solver.add(obj)
        active_dofs = list(range(klampt_robot_model.numDrivers()))

        solver.setMaxIters(150)
        solver.setTolerance(positional_tolerance)

        solver.sampleInitial()

        res = solver.solve()
        if res:
            return klampt_robot_model.getConfig()

    return None


def batch_ik_and_filter_klampt_backend(jrl_robot_or_none, klampt_robot_model, ee_link_idx, poses_batch):
    """
    Unified batch-IK function:
      - if jrl_robot_or_none provided and has inverse_kinematics_klampt -> use that (preferred)
      - else fallback to klampt_ik_solver on klampt_robot_model
    Returns list of (ok, q_partial) with q_partial length == numDrivers
    """
    out = []
    for pose in poses_batch:
        pose_arr = np.asarray(pose, dtype=float)
        q = None
        if jrl_robot_or_none is not None:
            try:
                # jrl Robot expects np.array and returns x in its internal representation; adapt if necessary
                q = jrl_robot_or_none.inverse_kinematics_klampt(pose_arr)
            except Exception:
                q = None
        if q is None:
            # fallback to klampt solver
            q = klampt_ik_solver(klampt_robot_model, ee_link_idx, pose_arr)
        ok = q is not None
        if not ok:
            out.append((False, None))
        else:
            out.append((True, np.asarray(q, dtype=float)))
    return out

# ---------------------------
# World + visualization container
# ---------------------------
class DualWorld:
    def __init__(self, urdf_left, urdf_right, urdf_object):
        # Klampt world for visualization & collision
        self.world = WorldModel()
        # load robots into klampt world
        okL = self.world.readFile(urdf_left)
        okR = self.world.readFile(urdf_right)
        if not okL or not okR:
            raise RuntimeError("Klampt: Failed to read robot URDFs")
        # load object mesh (expect {urdf_object}.stl adjacent)
        # parse object URDF to get mesh name (we used parse_object_urdf_for_offsets_and_mesh earlier)
        _, _, mesh_abs = parse_object_urdf_for_offsets_and_mesh(urdf_object)
        stl_path = mesh_abs.replace(".obj", ".stl").replace(".dae", ".stl")
        # try loadRigidObject with the provided mesh path (or the .stl variant)
        if not self.world.loadRigidObject(mesh_abs.replace(".urdf", ".stl")):
            # try stl_path or fallback to mesh_abs (if already .stl)
            self.world.loadRigidObject(mesh_abs)

        # keep references
        self.klampt_robot_L = self.world.robot(0)
        self.klampt_robot_R = self.world.robot(1)
        self.klampt_obj = self.world.rigidObject(0)
        self.collider = collide.WorldCollider(self.world)

        # try to create jrl Robot wrappers if available
        self.jrl_left = None
        self.jrl_right = None
        if HAVE_JRL:
            try:
                # JRLRobot constructor: try common patterns
                try:
                    self.jrl_left = JRLRobot(urdf_left)
                    self.jrl_right = JRLRobot(urdf_right)
                except Exception:
                    # maybe classmethod from_urdf
                    self.jrl_left = JRLRobot.from_urdf(urdf_left)
                    self.jrl_right = JRLRobot.from_urdf(urdf_right)
            except Exception:
                # If creating JRLRobot fails, just leave None (we'll use klampt fallback)
                self.jrl_left = None
                self.jrl_right = None

    def set_object_T(self, T_world_obj):
        R_flat = T_world_obj[:3, :3].T.flatten().tolist()
        t_list = T_world_obj[:3, 3].tolist()
        self.klampt_obj.setTransform(R_flat, t_list)

    def set_qs_for_vis(self, qL_partial, qR_partial):
        qL_full = expand_q_to_full_klampt(self.klampt_robot_L, qL_partial)
        qR_full = expand_q_to_full_klampt(self.klampt_robot_R, qR_partial)
        self.klampt_robot_L.setConfig(qL_full)
        self.klampt_robot_R.setConfig(qR_full)

    def show(self):
        vis.add("world", self.world)
        vis.setAttribute("world", "drawTransparency", 0.3)
        vis.setAttribute("world", "drawContacts", True)
        vis.show()
        vis.loop()

# ---------------------------
# Visualize a single dual-arm pose using numeric IK
# ---------------------------
def visualize_dual_arm_pose(dual_world: DualWorld, pose_vec, T_left_off, T_right_off, n_solutions=1):
    """
    pose_vec: [x,y,z,qw,qx,qy,qz] for object in world frame
    Will compute IK for left and right end-effectors and visualize best found solution.
    """
    T_obj = posevec_to_T(pose_vec)
    T_left_global = T_obj @ T_left_off
    T_right_global = T_obj @ T_right_off

    left_pose_vec = T_to_posevec(T_left_global)
    # compute pose for right EE in right-arm base coordinates expected by your IK (here we pass world-space)
    right_pose_vec = T_to_posevec(T_right_global)

    # choose ee link index for Klampt IK objective
    ee_left_idx = dual_world.klampt_robot_L.numLinks() - 1
    ee_right_idx = dual_world.klampt_robot_R.numLinks() - 1

    # Batch call (single-pose lists)
    left_results = batch_ik_and_filter_klampt_backend(dual_world.jrl_left, dual_world.klampt_robot_L, ee_left_idx, [left_pose_vec])
    right_results = batch_ik_and_filter_klampt_backend(dual_world.jrl_right, dual_world.klampt_robot_R, ee_right_idx, [right_pose_vec])

    okL, qL = left_results[0]
    okR, qR = right_results[0]

    if not (okL and okR):
        print("⚠️ Keine gültige IK gefunden für beide Arme (links/rechts):", okL, okR)
        return False

    # set and visualize
    dual_world.set_qs_for_vis(qL, qR)
    dual_world.set_object_T(T_obj)

    # draw frames
    draw_frame(T_obj, "obj_com", size=0.08)
    draw_frame(T_left_global, "left_ee_target", size=0.05)
    draw_frame(T_right_global, "right_ee_target", size=0.05)

    dual_world.show()
    return True

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf_left", required=True)
    ap.add_argument("--urdf_right", required=True)
    ap.add_argument("--urdf_object", required=True)
    ap.add_argument("--pose", required=True, help="x,y,z,qw,qx,qy,qz")
    args = ap.parse_args()

    pose_vals = [float(x) for x in args.pose.split(",")]
    if len(pose_vals) != 7:
        raise RuntimeError("pose must have 7 values: x,y,z,qw,qx,qy,qz")
    T_obj = posevec_to_T(pose_vals)

    # parse object to get offsets and mesh
    T_left_off, T_right_off, mesh_abs = parse_object_urdf_for_offsets_and_mesh(args.urdf_object)

    # build world (klampt + optional jrl robots)
    dw = DualWorld(args.urdf_left, args.urdf_right, args.urdf_object)

    ok = visualize_dual_arm_pose(dw, pose_vals, T_left_off, T_right_off)
    if not ok:
        print("Visualisierung fehlgeschlagen (keine IK).")

if __name__ == "__main__":
    main()


"""
python scripts/debug_dual_arm_pose.py \
    --urdf_left urdfs/iiwa7_L/iiwa7_L_updated.urdf \
    --urdf_right urdfs/iiwa7_R/iiwa7_R_updated.urdf \
    --urdf_object urdfs/object/se3_object.urdf \
    --pose "1,0.0,0.9,1,0,0,0"

"""