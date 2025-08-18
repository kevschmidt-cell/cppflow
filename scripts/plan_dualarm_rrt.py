#!/usr/bin/env python3
"""
plan_dualarm_rrt.py

Dual-arm RRT in object-pose space (x,y,z,qw,qx,qy,qz).
- Lädt linken Arm, rechten Arm und Objekt jeweils aus separaten URDFs.
- Liest Grasp-Offsets (obj_com -> obj_p_01 / obj_p_02) aus dem Objekt-URDF.
- Verwendet IKFlow für IK der einzelnen Arme.
- Kollisionen über Klampt (robot self, robot-robot, robot-object).

Beispiel:
python scripts/plan_dualarm_rrt.py \
  --ik_left_model iiwa7_left_arm_0.75m \
  --ik_right_model iiwa7_right_arm_0.1m \
  --urdf_left urdfs/iiwa7_L/iiwa7_L_updated.urdf \
  --urdf_right urdfs/iiwa7_R/iiwa7_R_updated.urdf \
  --urdf_object urdfs/object/se3_object.urdf \
  --start_obj "0.45,0.00,0.75,1,0,0,0" \
  --goal_obj  "0.60,0.20,0.75,1,0,0,0" \
  --out_obj_csv planned_object_path.csv \
  --out_joint_csv planned_joints.csv
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from time import time

import numpy as np
import pandas as pd
from sympy import root
import torch
from scipy.spatial.transform import Rotation as R

# IKFlow
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from ikflow.config import DEVICE

# Klampt
from klampt import WorldModel, vis
from klampt.model import collide

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
    """4x4 -> [x,y,z,qw,qx,qy,qz]"""
    p = T[:3, 3]
    q = R.from_matrix(T[:3, :3]).as_quat()  # (qx,qy,qz,qw)
    return [float(p[0]), float(p[1]), float(p[2]), float(q[3]), float(q[0]), float(q[1]), float(q[2])]

def posevec_to_T(v):
    """[x,y,z,qw,qx,qy,qz] -> 4x4"""
    x, y, z, qw, qx, qy, qz = map(float, v)
    T = np.eye(4)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T

def visualize_dual_arm_pose(wc, T_obj, T_left_offset, T_right_offset, ik_left, ik_right):
    # IK berechnen
    left_ee_pose = T_to_posevec(T_obj @ T_left_offset)
    right_ee_pose = T_to_posevec(T_obj @ T_right_offset)
    
    okL, qL = batch_ik_and_filter(ik_left, [left_ee_pose])[0]
    okR, qR = batch_ik_and_filter(ik_right, [right_ee_pose])[0]
    
    if not (okL and okR):
        print("Keine gültige IK für diese Pose!")
        return
    
    wc.set_q(qL, qR)
    wc.set_object_T(T_obj)
    
    wc.view()

# ---------------------------
# Objekt-URDF: Grasp-Offsets & Meshpfad
# ---------------------------
def parse_object_urdf_for_offsets_and_mesh(urdf_object_path,
                                           child_left="obj_p_01",
                                           child_right="obj_p_02",
                                           base="obj_com"):
    """
    - Findet Joints base->child_left und base->child_right und liefert deren origin als 4x4.
    - Sucht im Link 'base' eine Visual-Mesh-Referenz und gibt deren Pfad zurück.
    """
    tree = ET.parse(urdf_object_path)
    root = tree.getroot()
    print("Gefundene Joints im URDF:")
    for joint in root.findall("joint"):
        print(joint.get("name"), "-> child:", joint.get("child"))


    def find_origin_to(child_name):
        for joint in root.findall("joint"):
            child_elem = joint.find("child")
            if child_elem is not None and child_elem.get("link") == child_name:
                org = joint.find("origin")
                xyz = [0, 0, 0]
                rpy = [0, 0, 0]
                if org is not None:
                    if org.get("xyz"):
                        xyz = [float(s) for s in org.get("xyz").split()]
                    if org.get("rpy"):
                        rpy = [float(s) for s in org.get("rpy").split()]
                return xyzrpy_to_T(xyz, rpy)
        raise RuntimeError(f"Kein Joint zu child '{child_name}' im Objekt-URDF gefunden.")


    # Grasp-Offsets bezogen auf base
    T_obj_to_left = find_origin_to(child_left)
    T_obj_to_right = find_origin_to(child_right)

    # Visual-Mesh suchen
    mesh_path = None
    for link in root.findall("link"):
        if link.get("name") == base:
            vis = link.find("visual")
            if vis is not None:
                geom = vis.find("geometry")
                if geom is not None:
                    mesh = geom.find("mesh")
                    if mesh is not None and mesh.get("filename"):
                        mesh_path = mesh.get("filename")
            break

    # Falls kein Mesh im Base-Link, irgendein anderes Visual-Mesh
    if mesh_path is None:
        for link in root.findall("link"):
            vis = link.find("visual")
            if vis is not None:
                geom = vis.find("geometry")
                if geom is not None:
                    mesh = geom.find("mesh")
                    if mesh is not None and mesh.get("filename"):
                        mesh_path = mesh.get("filename")
                        break

    if mesh_path is None:
        raise RuntimeError("Konnte kein Visual-Mesh im Objekt-URDF finden (für Kollision).")

    # Absoluten Pfad zurückgeben
    urdf_dir = os.path.dirname(os.path.abspath(urdf_object_path))
    mesh_abs = mesh_path if os.path.isabs(mesh_path) else os.path.join(urdf_dir, mesh_path)

    return T_obj_to_left, T_obj_to_right, mesh_abs

def expand_q_to_full(robot, q_partial):
    """
    q_partial: Array der Treiber-Gelenke (7 DoF)
    Gibt eine volle Konfiguration zurück (inkl. Fixed Joints)
    """
    full_config = robot.getConfig()[:]       # aktuelle volle Konfiguration als Basis
    for i in range(robot.numDrivers()):
        drv = robot.driver(i)                # Treiber i abrufen
        full_config[drv.index] = float(q_partial[i])
    return full_config

# ---------------------------
# Kollisionen (Klampt)
# ---------------------------
class WorldCollision:
    def __init__(self, urdf_left, urdf_right, obj_mesh_path):
        self.world = WorldModel()

        assert self.world.readFile(urdf_left), f"URDF (links) konnte nicht geladen werden: {urdf_left}"
        assert self.world.readFile(urdf_right), f"URDF (rechts) konnte nicht geladen werden: {urdf_right}"

        # Objekt als RigidObject (Mesh) laden
        ok = self.world.loadRigidObject(obj_mesh_path.replace(".urdf", ".stl"))
        if not ok:
            raise RuntimeError(f"Objekt-Mesh konnte nicht geladen werden: {obj_mesh_path}")

        # Indizes: Reihenfolge: robot(0)=links, robot(1)=rechts, rigidObject(0)=obj
        self.robot_L = self.world.robot(0)
        self.robot_R = self.world.robot(1)
        self.obj = self.world.rigidObject(0)

        self.collider = collide.WorldCollider(self.world)

    def view(self):
        """Zeigt die aktuelle Szene in einem interaktiven Klampt-Fenster"""
        vis.add("world", self.world)
        vis.setAttribute("world", "drawTransparency", 0.3)
        vis.setAttribute("world", "drawContacts", True)  # hebt Kontaktpunkte hervor
        vis.show()
        vis.loop()

    def set_q(self, qL, qR):
        print("qL shape:", np.shape(qL))
        print("robot_L dofs:", self.robot_L.numDrivers())
        print("robot_L config length:", self.robot_L.numLinks())
        print("Expected config length:", self.robot_L.numLinks())
        print("robot_L initial config length:", len(self.robot_L.getConfig()))

        qL_full = expand_q_to_full(self.robot_L, qL)
        qR_full = expand_q_to_full(self.robot_R, qR)

        self.robot_L.setConfig(list(qL_full))
        self.robot_R.setConfig(list(qR_full))

    def set_object_T(self, T_world_obj):
        Rm = T_world_obj[:3, :3]
        t = T_world_obj[:3, 3]
        # Flatten R in row-major order
        R_list = Rm.flatten().tolist()
        t_list = t.tolist()
        self.obj.setTransform(R_list, t_list)

    def any_collision(self):
        # Selbstkollision prüfen
        if self.collider.robotSelfCollisions(self.robot_L):
            print("Linker Arm kollidiert mit sich selbst")
            return True
        if self.collider.robotSelfCollisions(self.robot_R):
            print("Rechter Arm kollidiert mit sich selbst")
            return True

        # Roboter ↔ Objekt Kollision prüfen
        if self.collider.robotObjectCollisions(self.robot_L, self.obj):
            print("Linker Arm kollidiert mit Objekt")
            return True
        if self.collider.robotObjectCollisions(self.robot_R, self.obj):
            print("Rechter Arm kollidiert mit Objekt")
            return True

        # Arm ↔ Arm Kollision prüfen
        for l_link in self.robot_L.links():
            for r_link in self.robot_R.links():
                if self.collider.linkLinkCollision(l_link, r_link):
                    print(f"Kollision zwischen {l_link.getName()} und {r_link.getName()}")
                    return True

        # Keine Kollision gefunden
        return False



# ---------------------------
# IKFlow Batch
# ---------------------------
def batch_ik_and_filter(ik_solver, poses_batch):
    """
    poses_batch: Liste von [x,y,z,qw,qx,qy,qz]
    Return: Liste (ok:bool, q:np.ndarray)
    """
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
# RRT Hilfen im Objekt-Pose-Raum
# ---------------------------
def pose_distance(a, b, pos_w=1.0, rot_w=0.2):
    da = np.linalg.norm(np.asarray(a[:3]) - np.asarray(b[:3]))
    qa = R.from_quat([a[4], a[5], a[6], a[3]])
    qb = R.from_quat([b[4], b[5], b[6], b[3]])
    ang = np.linalg.norm((qa.inv() * qb).as_rotvec())
    return pos_w * da + rot_w * ang

def steer_towards(a, b, step):
    ap = np.asarray(a[:3]); bp = np.asarray(b[:3])
    d = bp - ap
    L = np.linalg.norm(d)
    if L <= step:
        p = bp
    else:
        p = ap + d / L * step
    # Orientierung konstant (vom Ziel b)
    return [float(p[0]), float(p[1]), float(p[2]), b[3], b[4], b[5], b[6]]

# ---------------------------
# RRT Kern
# ---------------------------
def rrt_dual(
    ik_left,
    ik_right,
    wc: WorldCollision,
    T_obj_left_offset,
    T_obj_right_offset,
    start_obj_pose,
    goal_obj_pose,
    bounds,                   # ((xmin,xmax),(ymin,ymax),(zmin,zmax))
    max_iters=20000,
    step_size=0.03,
    batch_size=64,
):
    tree = [start_obj_pose]
    parents = {0: None}

    def sample_pose():
        x = float(np.random.uniform(*bounds[0]))
        y = float(np.random.uniform(*bounds[1]))
        z = float(np.random.uniform(*bounds[2]))
        return [x, y, z, goal_obj_pose[3], goal_obj_pose[4], goal_obj_pose[5], goal_obj_pose[6]]

    cand_batch = []

    for it in range(max_iters):
        target = goal_obj_pose if (np.random.rand() < 0.1) else sample_pose()
        # nearest
        dists = [pose_distance(n, target) for n in tree]
        idx_near = int(np.argmin(dists))
        new_pose = steer_towards(tree[idx_near], target, step_size)
        cand_batch.append((idx_near, new_pose))

        if len(cand_batch) >= batch_size or it == max_iters - 1:
            mats_obj = [posevec_to_T(p) for _, p in cand_batch]
            mats_left_ee = [m @ T_obj_left_offset for m in mats_obj]
            mats_right_ee = [m @ T_obj_right_offset for m in mats_obj]
            poses_left = [T_to_posevec(m) for m in mats_left_ee]
            poses_right = [T_to_posevec(m) for m in mats_right_ee]

            resL = batch_ik_and_filter(ik_left, poses_left)
            resR = batch_ik_and_filter(ik_right, poses_right)

            # Debug: Kandidaten prüfen mit detaillierter Kollisionserkennung
            MAX_DEBUG = 5
            debug_count = 0

            from klampt import vis

            contact_markers = []

            for idx, ((idx_parent, cand_pose), (okL, qL), (okR, qR)) in enumerate(zip(cand_batch, resL, resR)):
                if not (okL and okR):
                    continue  # Überspringe Kandidaten ohne gültige IK
                
                wc.set_q(qL, qR)
                wc.set_object_T(posevec_to_T(cand_pose))
                wc.view()
                collisions = []

                # Selbstkollisionen
                if wc.collider.robotSelfCollisions(wc.robot_L):
                    collisions.append("left_arm_self")
                if wc.collider.robotSelfCollisions(wc.robot_R):
                    collisions.append("right_arm_self")

                # Arm-Objekt-Kollision
                if wc.collider.robotObjectCollisions(wc.robot_L, wc.obj):
                    collisions.append("left_arm_obj")
                if wc.collider.robotObjectCollisions(wc.robot_R, wc.obj):
                    collisions.append("right_arm_obj")

                if collisions:
                    print(f"[Debug {idx}] collisions: {collisions}")
                    continue  # Kandidat verwerfen

                # Node akzeptieren
                tree.append(cand_pose)
                parents[len(tree)-1] = idx_parent

                # Zielprüfung
                if pose_distance(cand_pose, goal_obj_pose) < step_size:
                    path = []
                    k = len(tree)-1
                    while k is not None:
                        path.append(tree[k])
                        k = parents[k]
                    path.reverse()

                    # Gelenkpfade
                    mats = [posevec_to_T(p) for p in path]
                    Lposes = [T_to_posevec(m @ T_obj_left_offset) for m in mats]
                    Rposes = [T_to_posevec(m @ T_obj_right_offset) for m in mats]
                    JL = batch_ik_and_filter(ik_left, Lposes)
                    JR = batch_ik_and_filter(ik_right, Rposes)
                    qL_list = [q for ok, q in JL]
                    qR_list = [q for ok, q in JR]
                    return path, qL_list, qR_list

        cand_batch.clear()

    return None, None, None


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ik_left_model", required=True)
    ap.add_argument("--ik_right_model", required=True)
    ap.add_argument("--urdf_left", required=True)
    ap.add_argument("--urdf_right", required=True)
    ap.add_argument("--urdf_object", required=True,
                    help="Objekt-URDF (liefert Grasp-Offsets + Visual-Mesh für Kollision)")
    ap.add_argument("--start_obj", required=True, help="x,y,z,qw,qx,qy,qz")
    ap.add_argument("--goal_obj", required=True, help="x,y,z,qw,qx,qy,qz")
    ap.add_argument("--bounds", default=None, help="xmin,xmax,ymin,ymax,zmin,zmax")
    ap.add_argument("--max_iters", type=int, default=20000)
    ap.add_argument("--step_size", type=float, default=0.03)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_obj_csv", default="planned_object_path.csv")
    ap.add_argument("--out_joint_csv", default="planned_joints.csv")
    args = ap.parse_args()

    def parse_pose(s):
        v = [float(x) for x in s.split(",")]
        if len(v) != 7:
            raise ValueError("Pose muss 7 Werte haben: x,y,z,qw,qx,qy,qz")
        return v

    start_pose = parse_pose(args.start_obj)
    goal_pose  = parse_pose(args.goal_obj)

    if args.bounds:
        b = [float(x) for x in args.bounds.split(",")]
        bounds = ((b[0], b[1]), (b[2], b[3]), (b[4], b[5]))
    else:
        # Box um Start/Goal ± 0.4 m
        mins = np.minimum(np.asarray(start_pose[:3]), np.asarray(goal_pose[:3])) - 0.4
        maxs = np.maximum(np.asarray(start_pose[:3]), np.asarray(goal_pose[:3])) + 0.4
        bounds = ((float(mins[0]), float(maxs[0])),
                  (float(mins[1]), float(maxs[1])),
                  (float(mins[2]), float(maxs[2])))

    print(">> IKFlow-Modelle laden …")
    ik_left, _  = get_ik_solver(args.ik_left_model)
    ik_right, _ = get_ik_solver(args.ik_right_model)

    print(">> Objekt-URDF parsen (Grasp-Offsets + Mesh) …")
    T_obj_left_off, T_obj_right_off, obj_mesh_abs = parse_object_urdf_for_offsets_and_mesh(args.urdf_object)

    print(">> Welt/Kollision (Klampt) initialisieren …")
    wc = WorldCollision(args.urdf_left, args.urdf_right, obj_mesh_abs)

    print(">> Starte RRT-Planung …")
    t0 = time()
    path, qL_list, qR_list = rrt_dual(
        ik_left, ik_right, wc,
        T_obj_left_off, T_obj_right_off,
        start_pose, goal_pose, bounds,
        max_iters=args.max_iters, step_size=args.step_size, batch_size=args.batch_size
    )
    dt = time() - t0

    if path is None:
        print("!! Kein Pfad gefunden.")
        sys.exit(1)

    print(f">> Pfad gefunden in {dt:.2f}s, #Waypoints = {len(path)}")

    # Objektpfad speichern
    df_obj = pd.DataFrame(path, columns=["x","y","z","qw","qx","qy","qz"])
    df_obj.insert(0, "t", np.arange(len(path), dtype=int))
    df_obj.to_csv(args.out_obj_csv, index=False)
    print("Gespeichert:", args.out_obj_csv)

    # Gelenkpfade speichern (t, qL0.., qR0..)
    nqL = len(qL_list[0]); nqR = len(qR_list[0])
    rows = []
    for i, (qL, qR) in enumerate(zip(qL_list, qR_list)):
        rows.append([i] + list(qL) + list(qR))
    cols = ["t"] + [f"qL{i}" for i in range(nqL)] + [f"qR{i}" for i in range(nqR)]
    df_j = pd.DataFrame(rows, columns=cols)
    df_j.to_csv(args.out_joint_csv, index=False)
    print("Gespeichert:", args.out_joint_csv)

    print("Fertig.")

if __name__ == "__main__":
    main()

