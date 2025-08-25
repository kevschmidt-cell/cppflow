#!/usr/bin/env python3
"""
plan_dualarm_rrt_with_obstacles.py

Dual-arm RRT in object-pose space (x,y,z,qw,qx,qy,qz) mit Hindernissen.
- Lädt linken Arm, rechten Arm und Objekt jeweils aus URDF.
- Liest Grasp-Offsets (obj_com -> obj_p_01 / obj_p_02) aus Objekt-URDF.
- Verwendet IKFlow für IK der Arme.
- Kollisionen über Klampt (self, arm-arm, arm-object, Hindernisse).

Beispiel:
python scripts/plan_dualarm_rrt.py \
  --ik_left_model iiwa7_left_arm_0.75m \
  --ik_right_model iiwa7_right_arm_0.1m \
  --urdf_left urdfs/iiwa7_L/iiwa7_L_updated.urdf \
  --urdf_right urdfs/iiwa7_R/iiwa7_R_updated.urdf \
  --urdf_object urdfs/object/se3_object.urdf \
  --start_obj "1,-0.5,0.9,1,0,0,0" \
  --goal_obj  "1,-0.3,1.1,1,0,0,0" \
  --urdf_obstacles "urdfs/obstacle/obstacle.urdf" \
  --out_obj_csv planned_object_path.csv \
  --out_joint_csv planned_joints.csv
"""

#!/usr/bin/env python3
"""
plan_dualarm_rrt_with_obstacles.py

Dual-arm RRT in object-pose space (x,y,z,qw,qx,qy,qz) mit Hindernissen.
- Lädt linken Arm, rechten Arm und Objekt (als RigidObject) in eine Klampt-Welt.
- Grasp-Offsets aus Objekt-URDF (obj_com -> obj_p_01 / obj_p_02).
- IK per IKFlow.
- Kollisionen per Klampt WorldCollider:
    left arm <-> obstacles, right arm <-> obstacles, object <-> obstacles.
"""

import argparse, os, sys, xml.etree.ElementTree as ET
from time import time
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R

# IKFlow
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from ikflow.config import DEVICE

# Klampt
from klampt import WorldModel
from klampt.model import collide
from klampt.math import se3
from klampt.robotsim import RobotModelLink, RigidObjectModel

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
    p = T[:3,3]
    q = R.from_matrix(T[:3,:3]).as_quat()  # (qx,qy,qz,qw)
    return [float(p[0]), float(p[1]), float(p[2]), float(q[3]), float(q[0]), float(q[1]), float(q[2])]

def posevec_to_T(v):
    x,y,z,qw,qx,qy,qz = map(float,v)
    T = np.eye(4)
    T[:3,:3] = R.from_quat([qx,qy,qz,qw]).as_matrix()
    T[:3,3] = [x,y,z]
    return T

def T_to_se3(T):
    Rm = T[:3,:3]; t = T[:3,3]
    return (Rm.reshape(-1).tolist(), t.tolist())

def expand_q_to_full(robot, q_partial):
    """Teilkonfiguration -> volle Robot-Konfiguration (inkl. fixed joints)."""
    full_config = robot.getConfig()[:]   # aktuelle volle Konfiguration als Basis
    assert len(q_partial) == robot.numDrivers(), \
        f"q_partial hat {len(q_partial)} Werte, erwartet {robot.numDrivers()}"
    for i in range(robot.numDrivers()):
        drv = robot.driver(i)
        full_config[drv.getAffectedLink()] = float(q_partial[i])
    return full_config

# ---------------------------
# Objekt-URDF
# ---------------------------
def parse_object_urdf_for_offsets_and_mesh(urdf_object_path, child_left="obj_p_01", child_right="obj_p_02", base="obj_com"):
    tree = ET.parse(urdf_object_path)
    root = tree.getroot()

    def find_origin_to(child_name):
        for joint in root.findall("joint"):
            child_elem = joint.find("child")
            if child_elem is not None and child_elem.get("link")==child_name:
                org = joint.find("origin")
                xyz = [0,0,0]; rpy=[0,0,0]
                if org is not None:
                    if org.get("xyz"): xyz=[float(s) for s in org.get("xyz").split()]
                    if org.get("rpy"): rpy=[float(s) for s in org.get("rpy").split()]
                return xyzrpy_to_T(xyz,rpy)
        raise RuntimeError(f"Kein Joint zu child '{child_name}' gefunden.")

    T_obj_to_left = find_origin_to(child_left)
    T_obj_to_right = find_origin_to(child_right)

    mesh_path = None
    for link in root.findall("link"):
        if link.get("name")==base:
            vis = link.find("visual")
            if vis is not None:
                geom = vis.find("geometry")
                if geom is not None:
                    mesh = geom.find("mesh")
                    if mesh is not None and mesh.get("filename"):
                        mesh_path = mesh.get("filename")
            break
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
        raise RuntimeError("Kein Visual-Mesh im Objekt-URDF gefunden.")

    urdf_dir = os.path.dirname(os.path.abspath(urdf_object_path))
    mesh_abs = mesh_path if os.path.isabs(mesh_path) else os.path.join(urdf_dir, mesh_path)
    return T_obj_to_left, T_obj_to_right, mesh_abs

# ---------------------------
# Batch-IK
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
# CollisionChecker (nur Arm/Objekt <-> Hindernisse)
# ---------------------------
class DualArmCollisionChecker:
    """
    Prüft:
      - linker Arm   <-> Obstacles (Roboter-URDFs)
      - rechter Arm  <-> Obstacles
      - Objekt       <-> Obstacles
    KEINE Arm-Arm- oder Arm-Objekt-Checks.
    """
    def __init__(self, world, robot_left, robot_right, grasped_object, obstacles_robots):
        self.world = world
        self.left = robot_left
        self.right = robot_right
        self.obj = grasped_object           # RigidObjectModel
        self.obstacles = obstacles_robots   # Liste von RobotModel (Hindernis-URDFs)
        self.cw = collide.WorldCollider(world)

        # IDs für schnelle Filter
        self.left_id = self.left.getID()
        self.right_id = self.right.getID()
        self.obs_ids = {r.getID() for r in self.obstacles}
        self.obj_id = self.obj.getID()

    def _is_links_of(self, robot_id):
        def f(o):
            return isinstance(o, RobotModelLink) and o.robot().getID()==robot_id
        return f

    def _is_links_of_obstacles(self, o):
        return isinstance(o, RobotModelLink) and (o.robot().getID() in self.obs_ids)

    def _is_the_object(self, o):
        return isinstance(o, RigidObjectModel) and (o.getID()==self.obj_id)

    def set_object_T(self, T_obj):
        Rm, t = T_to_se3(T_obj)
        self.obj.setTransform(Rm, t)

    def any_collision(self, qL_full, qR_full, T_obj):
        # setze Konfigurationen
        self.left.setConfig(qL_full)
        self.right.setConfig(qR_full)
        self.set_object_T(T_obj)

        # Left vs Obstacles
        if any(self.cw.collisions(self._is_links_of(self.left_id), self._is_links_of_obstacles)):
            return True
        # Right vs Obstacles
        if any(self.cw.collisions(self._is_links_of(self.right_id), self._is_links_of_obstacles)):
            return True
        # Object vs Obstacles
        if any(self.cw.collisions(self._is_the_object, self._is_links_of_obstacles)):
            return True

        return False

# ---------------------------
# RRT Hilfen
# ---------------------------
def pose_distance(a,b,pos_w=1.0,rot_w=0.2):
    da = np.linalg.norm(np.asarray(a[:3])-np.asarray(b[:3]))
    qa = R.from_quat([a[4],a[5],a[6],a[3]])
    qb = R.from_quat([b[4],b[5],b[6],b[3]])
    ang = np.linalg.norm((qa.inv()*qb).as_rotvec())
    return pos_w*da + rot_w*ang

def steer_towards(a,b,step):
    ap=np.asarray(a[:3]); bp=np.asarray(b[:3])
    d=bp-ap; L=np.linalg.norm(d)
    p=bp if L<=step else ap+d/L*step
    return [float(p[0]),float(p[1]),float(p[2]),b[3],b[4],b[5],b[6]]

# ---------------------------
# RRT Kern
# ---------------------------
def rrt_dual(ik_left, ik_right, checker, T_obj_left_offset, T_obj_right_offset,
             start_obj_pose, goal_obj_pose, bounds, max_iters=20000, step_size=0.03, batch_size=64):
    tree = [start_obj_pose]
    parents={0:None}
    cand_batch=[]
    def sample_pose():
        x = float(np.random.uniform(*bounds[0]))
        y = float(np.random.uniform(*bounds[1]))
        z = float(np.random.uniform(*bounds[2]))
        return [x,y,z,goal_obj_pose[3],goal_obj_pose[4],goal_obj_pose[5],goal_obj_pose[6]]
    for it in range(max_iters):
        target = goal_obj_pose if np.random.rand()<0.1 else sample_pose()
        dists = [pose_distance(n,target) for n in tree]
        idx_near = int(np.argmin(dists))
        new_pose = steer_towards(tree[idx_near], target, step_size)
        cand_batch.append((idx_near,new_pose))
        if len(cand_batch)>=batch_size or it==max_iters-1:
            mats_obj = [posevec_to_T(p) for _,p in cand_batch]
            mats_left = [m@T_obj_left_offset for m in mats_obj]
            mats_right = [m@T_obj_right_offset for m in mats_obj]
            poses_left = [T_to_posevec(m) for m in mats_left]
            poses_right = [T_to_posevec(m) for m in mats_right]
            resL=batch_ik_and_filter(ik_left, poses_left)
            resR=batch_ik_and_filter(ik_right, poses_right)
            for (idx_parent,cand_pose), M_obj, (okL,qL),(okR,qR) in zip(cand_batch, mats_obj, resL, resR):
                if not(okL and okR): continue
                # IK-Teilkonfig -> volle Konfig
                qL_full = expand_q_to_full(checker.left, qL)
                qR_full = expand_q_to_full(checker.right, qR)
                if checker.any_collision(qL_full, qR_full, M_obj): 
                    continue
                tree.append(cand_pose)
                parents[len(tree)-1]=idx_parent
                if pose_distance(cand_pose,goal_obj_pose)<step_size:
                    # Rückverfolgung
                    path=[]
                    k=len(tree)-1
                    while k is not None:
                        path.append(tree[k])
                        k=parents[k]
                    path.reverse()
                    # Gelenkpfade (einfach nochmal IK evaluieren)
                    mats=[posevec_to_T(p) for p in path]
                    JL=[q for ok,q in batch_ik_and_filter(ik_left, [T_to_posevec(m@T_obj_left_offset) for m in mats])]
                    JR=[q for ok,q in batch_ik_and_filter(ik_right,[T_to_posevec(m@T_obj_right_offset) for m in mats])]
                    return path, JL, JR
            cand_batch.clear()
    return None,None,None

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ik_left_model", required=True)
    ap.add_argument("--ik_right_model", required=True)
    ap.add_argument("--urdf_left", required=True)
    ap.add_argument("--urdf_right", required=True)
    ap.add_argument("--urdf_object", required=True)
    ap.add_argument("--urdf_obstacles", default="", help="Komma-getrennte URDFs von Hindernissen (als Roboter geladen)")
    ap.add_argument("--start_obj", required=True)
    ap.add_argument("--goal_obj", required=True)
    ap.add_argument("--bounds", default=None)
    ap.add_argument("--max_iters", type=int, default=20000)
    ap.add_argument("--step_size", type=float, default=0.03)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_obj_csv", default="planned_object_path.csv")
    ap.add_argument("--out_joint_csv", default="planned_joints.csv")
    args = ap.parse_args()

    def parse_pose(s): 
        v=[float(x) for x in s.split(",")]; 
        assert len(v)==7; 
        return v

    start_pose=parse_pose(args.start_obj)
    goal_pose=parse_pose(args.goal_obj)
    
    if args.bounds:
        b=[float(x) for x in args.bounds.split(",")]
        bounds=((b[0],b[1]),(b[2],b[3]),(b[4],b[5]))
    else:
        mins=np.minimum(start_pose[:3],goal_pose[:3])-0.4
        maxs=np.maximum(start_pose[:3],goal_pose[:3])+0.4
        bounds=((float(mins[0]),float(maxs[0])),(float(mins[1]),float(maxs[1])),(float(mins[2]),float(maxs[2])))

    print(">> IKFlow-Modelle laden …")
    ik_left,_=get_ik_solver(args.ik_left_model)
    ik_right,_=get_ik_solver(args.ik_right_model)

    print(">> Objekt-URDF parsen …")
    T_obj_left_off, T_obj_right_off, obj_mesh_abs = parse_object_urdf_for_offsets_and_mesh(args.urdf_object)

    print(">> Klampt World initialisieren …")
    world=WorldModel()
    robot_left=world.loadRobot(args.urdf_left)
    robot_right=world.loadRobot(args.urdf_right)

    # Objekt als RigidObject in die Welt
    obj = world.makeRigidObject("grasped_object")
    if not obj.geometry().loadFile(obj_mesh_abs):
        raise RuntimeError(f"Konnte Mesh für Objekt nicht laden: {obj_mesh_abs}")
    # Startpose für Visualisierung / erste Kollisionen
    obj.setTransform(*T_to_se3(posevec_to_T(start_pose)))

    # Hindernisse als Roboter laden (dein obstacle.urdf ist ein 'robot')
    obstacles=[]
    if args.urdf_obstacles:
        for path in args.urdf_obstacles.split(","):
            path = path.strip()
            if not path: 
                continue
            r = world.loadRobot(path)
            if r is None:
                raise RuntimeError(f"Hindernis-URDF konnte nicht als Robot geladen werden: {path}")
            obstacles.append(r)

    checker = DualArmCollisionChecker(world, robot_left, robot_right, obj, obstacles)

    print(">> Starte RRT-Planung …")
    t0=time()
    path,qL_list,qR_list=rrt_dual(
        ik_left,ik_right,checker,
        T_obj_left_off,T_obj_right_off,
        start_pose,goal_pose,bounds,
        max_iters=args.max_iters, step_size=args.step_size,batch_size=args.batch_size)
    dt=time()-t0

    if path is None:
        print("!! Kein Pfad gefunden.")
        sys.exit(1)

    print(f">> Pfad gefunden in {dt:.2f}s, #Waypoints = {len(path)}")

    # Objektpfad speichern
    df_obj = pd.DataFrame(path, columns=["x", "y", "z", "qw", "qx", "qy", "qz"])
    df_obj.insert(0, "t", np.arange(len(path), dtype=int))
    df_obj.to_csv(args.out_obj_csv, index=False)
    print("Gespeichert:", args.out_obj_csv)

    # Endeffektorpfade berechnen
    poses_left, poses_right = [], []
    for p in path:
        # Objektpose als 4x4 Matrix
        T_obj = np.eye(4)
        T_obj[:3,3] = p[:3]
        T_obj[:3,:3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()  # (qx,qy,qz,qw)

        # Endeffektor-Transformationen
        T_left  = T_obj @ T_obj_left_off
        T_right = T_obj @ T_obj_right_off

        poses_left.append(T_to_posevec(T_left))
        poses_right.append(T_to_posevec(T_right))

    # CSV speichern
    df_left = pd.DataFrame(poses_left, columns=["x", "y", "z", "qw", "qx", "qy", "qz"])
    df_left.insert(0, "t", np.arange(len(poses_left), dtype=int))
    df_left.to_csv("cppflow/paths/end_eff_left.csv", index=False)

    df_right = pd.DataFrame(poses_right, columns=["x", "y", "z", "qw", "qx", "qy", "qz"])
    df_right.insert(0, "t", np.arange(len(poses_right), dtype=int))
    df_right.to_csv("cppflow/paths/end_eff_right.csv", index=False)

    print("Gespeichert: end_eff_left.csv, end_eff_right.csv")

    print("Fertig.")

if __name__=="__main__":
    main()
