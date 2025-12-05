"""
python scripts/rrt_ompl.py \
  --ik_left_model iiwa7_left_arm\
  --urdf_left urdfs/iiwa7_L/iiwa7_L_updated.urdf \
  --urdf_right urdfs/iiwa7_R/iiwa7_R_updated.urdf \
  --urdf_object urdfs/object/se3_object.urdf \
  --urdf_obstacles "urdfs/obstacle/passage.urdf,urdfs/obstacle/passage2.urdf" \
  --start_obj "1.05,0.0,0.9,1,0,0,0" \
  --goal_obj  "0.8,0.0,0.9,1,0,0,0" \
  --bounds "0.65,1.1,-0.2,0.2,0.6,1.5" \
  --time_limit 10 \
  --max_rot_deg 35 \
  --save_prefix run1
"""

#!/usr/bin/env python3
import argparse, os, sys, xml.etree.ElementTree as ET
from time import time
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R

# OMPL
from ompl import base as ob
from ompl import geometric as og

# IKFlow
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from ikflow.config import DEVICE

# Klampt
from klampt import WorldModel
from klampt.model import collide
from klampt.math import se3
from klampt.robotsim import RobotModelLink, RigidObjectModel

# --- GLOBAL METRICS ---
METRICS = {
    "validity_checks": 0,
    "ik_calls_left": 0,
    "ik_calls_right": 0,
    "ik_success_left": 0,
    "ik_success_right": 0,
    "collision_checks": 0,
    "collisions_found": 0,
    "path_length": 0.0,
}

def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])

    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])

    Rx = np.array([[1, 0,  0],
                   [0, cr, -sr],
                   [0, sr,  cr]])

    return Rz @ Ry @ Rx

xyz = np.array([0.3682, -0.1842, 0.7014])
rpy = np.array([0.0039, -0.0030, -0.0161])

xyz_R = np.array([0.3743, 0.1816, 0.7048])
rpy_R = np.array([-0.0012, 0.0001, -0.0158])
R_L = rpy_to_matrix(*rpy)
R_R = rpy_to_matrix(*rpy_R)

T_world_left = np.eye(4)
T_world_left[:3,:3] = R_L
T_world_left[:3, 3] = xyz

T_world_right = np.eye(4)
T_world_right[:3,:3] = R_R
T_world_right[:3, 3] = xyz_R

ROBOT_TO_BASE_TRANSFORM = {
    # example (identity = no base offset)
    "iiwa7": np.eye(4),
    "iiwa7_L": np.eye(4),
    #iiwa7_L": T_world_left, 
    "iiwa7_R": np.linalg.inv(T_world_left) @ T_world_right, 
    "iiwa7_N": np.eye(4),
}
def normalize_quat(w,x,y,z):
    q = np.array([w,x,y,z], dtype=float)
    n = np.linalg.norm(q)
    if n == 0: 
        return 1.0,0.0,0.0,0.0
    q /= n
    # optional: w>=0 für konsistente Richtung (kürzt 360°-Flips)
    if q[0] < 0:
        q = -q
    return tuple(q.tolist())

def so3_angle_between_q(qw1,qx1,qy1,qz1, qw2,qx2,qy2,qz2):
    # Winkel zwischen zwei Quaternions (0..pi)
    dot = qw1*qw2 + qx1*qx2 + qy1*qy2 + qz1*qz2
    dot = max(-1.0, min(1.0, dot))
    return 2.0*np.arccos(abs(dot))

# ---------------------------
# Utils
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
    Rm = T[:3,:3].T; t = T[:3,3]
    return (Rm.reshape(-1).tolist(), t.tolist())

def expand_q_to_full(robot, q_partial):
    """Teilkonfiguration -> volle Konfiguration (inkl. fixed joints)."""
    full_config = robot.getConfig()[:]   # aktuelle volle Konfiguration als Basis
    assert len(q_partial) == robot.numDrivers(), \
        f"q_partial hat {len(q_partial)} Werte, erwartet {robot.numDrivers()}"
    for i in range(robot.numDrivers()):
        drv = robot.driver(i)
        # konservativ: setze den 'leading' Gelenkwinkel des Drivers
        # (funktioniert für weit verbreitete URDFs gut genug)
        link_idx = drv.getAffectedLink()
        full_config[link_idx] = float(q_partial[i])
    return full_config

def parse_object_urdf_for_offsets_and_mesh(urdf_object_path,
                                           child_left="obj_p_01",
                                           child_right="obj_p_02",
                                           base="obj_com"):
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
# CollisionChecker (Arm/Object vs. Obstacles)
# ---------------------------
class DualArmCollisionChecker:
    """
    Prüft:
      - linker Arm   <-> Obstacles (Roboter-URDFs)
      - rechter Arm  <-> Obstacles
      - Objekt       <-> Obstacles
    KEINE Arm-Arm- oder Arm-Objekt-Checks.
    """
    def __init__(self, world, robot_left, robot_right, grasped_object, obstacles_robots,
                 ignore_links_left=None, ignore_links_right=None, ignore_links_obstacles=None,
                 check_object_vs_obstacles=True, collision_padding=0.0):
        self.world = world
        self.left = robot_left
        self.right = robot_right
        self.obj = grasped_object
        self.obstacles = obstacles_robots
        self.ignore_links_left = set(ignore_links_left or [])
        self.ignore_links_right = set(ignore_links_right or [])
        self.ignore_links_obstacles = set(ignore_links_obstacles or [])
        self.check_object_vs_obstacles = check_object_vs_obstacles
        self.cw = collide.WorldCollider(world)

        self.left_id = self.left.getID()
        self.right_id = self.right.getID()
        self.obs_ids = {r.getID() for r in self.obstacles}
        self.obj_id = self.obj.getID()
        self.collision_padding = float(collision_padding)

    def _is_links_of(self, robot_id, ignored):
        def f(o):
            return isinstance(o, RobotModelLink) and o.robot().getID()==robot_id and o.getName() not in ignored
        return f

    def _is_links_of_obstacles(self, o):
        return isinstance(o, RobotModelLink) and (o.robot().getID() in self.obs_ids) and (o.getName() not in self.ignore_links_obstacles)

    def _is_the_object(self, o):
        return isinstance(o, RigidObjectModel) and (o.getID()==self.obj_id)

    def set_object_T(self, T_obj):
        Rm, t = T_to_se3(T_obj)
        self.obj.setTransform(Rm, t)

    def _collides(self, f1, f2):
        """Collision with optional negative padding (allow small penetrations)."""
        if self.collision_padding <= 0.0:
            return any(self.cw.collisions(f1, f2))
        try:
            d = self.cw.distance(f1, f2)
            # distance < 0 => penetration; allow up to -padding before declaring collision
            return d is not None and d < -self.collision_padding
        except Exception:
            return any(self.cw.collisions(f1, f2))

    def any_collision(self, qL_full, qR_full, T_obj):
        self.left.setConfig(qL_full)
        self.right.setConfig(qR_full)
        self.set_object_T(T_obj)
        if self._collides(self._is_links_of(self.left_id, self.ignore_links_left), self._is_links_of_obstacles):
            return True
        if self._collides(self._is_links_of(self.right_id, self.ignore_links_right), self._is_links_of_obstacles):
            return True
        if self.check_object_vs_obstacles and self._collides(self._is_the_object, self._is_links_of_obstacles):
            return True
        return False

# ---------------------------
# OMPL State Validity Checker
# ---------------------------
class DualArmOMPLChecker:
    def __init__(self, ik_left, checker, T_obj_left_off, T_obj_right_off,
                 R_start, max_rot_angle=np.deg2rad(35), base_offset_right=np.eye(4)):
        self.ik_left = ik_left
        self.checker = checker
        self.T_obj_left_off = T_obj_left_off
        self.T_obj_right_off = T_obj_right_off
        self.R_start = R_start
        self.max_rot_angle = max_rot_angle
        self.base_offset_right = base_offset_right

    def __call__(self, state):

        # --- METRIC: validity checks ---
        METRICS["validity_checks"] += 1

        # Extrahiere Objektpose aus SE3-State
        xyz = [state.getX(), state.getY(), state.getZ()]
        rot = state.rotation()
        quat = [rot.x, rot.y, rot.z, rot.w]  # OMPL format

        T_obj = posevec_to_T([*xyz, quat[3], quat[0], quat[1], quat[2]])

        # Rotationsabweichung prüfen
        R_curr = R.from_matrix(T_obj[:3, :3])
        R_rel = self.R_start.inv() * R_curr
        angle = R_rel.magnitude()
        if angle > self.max_rot_angle:
            return False

        # --- IK für linken Arm ---
        METRICS["ik_calls_left"] += 1
        pose_left = T_to_posevec(T_obj @ self.T_obj_left_off)
        okL, qL = batch_ik_and_filter(self.ik_left, [pose_left])[0]
        if okL:
            METRICS["ik_success_left"] += 1
        if not okL:
            return False

        # --- IK für rechten Arm ---
        METRICS["ik_calls_right"] += 1
        T_ee_right_leftbasis = T_obj @ self.T_obj_right_off
        T_ee_right = np.linalg.inv(self.base_offset_right) @ T_ee_right_leftbasis
        pose_right = T_to_posevec(T_ee_right)
        okR, qR = batch_ik_and_filter(self.ik_left, [pose_right])[0]
        if okR:
            METRICS["ik_success_right"] += 1
        if not okR:
            return False

        # --- Vollständige q-Vektoren ---
        qL_full = expand_q_to_full(self.checker.left, qL)
        qR_full = expand_q_to_full(self.checker.right, qR)

        # --- Kollision ---
        METRICS["collision_checks"] += 1
        collision = self.checker.any_collision(qL_full, qR_full, T_obj)
        if collision:
            METRICS["collisions_found"] += 1
            return False

        return True


# ---------------------------
# OMPL Planung
# ---------------------------
def plan_with_ompl(start_pose, goal_pose, bounds, ik_left, checker,
                   T_obj_left_off, T_obj_right_off, R_start,
                   time_limit=60.0, simplify=True, range_hint=None, max_rot_deg=35,
                   sv_resolution=0.02, goal_bias=None, interp_res=0.01, planner_range=None):
    
    for k in METRICS:
        METRICS[k] = 0

    space = ob.SE3StateSpace()

    # Positionsgrenzen
    b = ob.RealVectorBounds(3)
    b.setLow(0, bounds[0][0]); b.setHigh(0, bounds[0][1])
    b.setLow(1, bounds[1][0]); b.setHigh(1, bounds[1][1])
    b.setLow(2, bounds[2][0]); b.setHigh(2, bounds[2][1])
    space.setBounds(b)

    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ob.StateValidityCheckerFn(
        DualArmOMPLChecker(ik_left, checker, T_obj_left_off, T_obj_right_off,
                           R_start, max_rot_angle=np.deg2rad(max_rot_deg), base_offset_right=ROBOT_TO_BASE_TRANSFORM["iiwa7_R"])
    ))
    # etwas gröbere Auflösung spart IK-Aufrufe; bei Bedarf feiner machen (z. B. 0.01)
    si.setStateValidityCheckingResolution(float(sv_resolution))
    si.setup()

    # Startzustand
    start = ob.State(space)
    start().setX(start_pose[0])
    start().setY(start_pose[1])
    start().setZ(start_pose[2])
    start().rotation().x = start_pose[4]
    start().rotation().y = start_pose[5]
    start().rotation().z = start_pose[6]
    start().rotation().w = start_pose[3]

    # Zielzustand
    goal = ob.State(space)
    goal().setX(goal_pose[0])
    goal().setY(goal_pose[1])
    goal().setZ(goal_pose[2])
    goal().rotation().x = goal_pose[4]
    goal().rotation().y = goal_pose[5]
    goal().rotation().z = goal_pose[6]
    goal().rotation().w = goal_pose[3]

    # Problemdefinition
    pdef = ob.ProblemDefinition(si)
    goal_tolerance = 0.01  # Position ~1 cm, Rotation ~2°
    pdef.setStartAndGoalStates(start, goal, goal_tolerance)

    pdef.setOptimizationObjective(
        ob.PathLengthOptimizationObjective(si)
    )
    # --- RRT* statt RRTConnect ---
    planner = og.RRTConnect(si)     # oder RRTstar, egal
    if range_hint is not None:
        planner.setRange(float(range_hint))
    elif planner_range is not None:
        planner.setRange(float(planner_range))
    if goal_bias is not None:
        planner.setGoalBias(float(goal_bias))

    #planner.setGoalBias(0.05)
    planner.setProblemDefinition(pdef)
    planner.setup()   # nur EINMAL aufrufen

    solved = planner.solve(time_limit)
    if not solved:
        return None

    path_geom = pdef.getSolutionPath()
    if simplify:
        og.PathSimplifier(si).simplifyMax(path_geom)

    # dynamische Interpolation: z. B. alle 0.01 m ein Wegpunkt
    resolution = float(interp_res)
    n_states = int(path_geom.length() / resolution)
    if n_states > 0:
        path_geom.interpolate(n_states)

    # Pfadlänge direkt aus OMPL
    path_length = path_geom.length()
    METRICS['path_length'] = path_length

    # States -> Posevec (x,y,z, qw,qx,qy,qz)
    out = []
    for s in path_geom.getStates():
        xyz = [s.getX(), s.getY(), s.getZ()]
        r = s.rotation()
        out.append([*xyz, r.w, r.x, r.y, r.z])

    # explizit Ressourcen freigeben (OMPL/C++)
    del path_geom
    del planner
    del pdef
    del si
    del space

    print("\n========== METRICS ==========")
    print(f"Validity checks: {METRICS['validity_checks']}")
    print(f"IK calls left/right: {METRICS['ik_calls_left']} / {METRICS['ik_calls_right']}")
    print(f"IK success left/right: {METRICS['ik_success_left']} / {METRICS['ik_success_right']}")
    print(f"Collision checks: {METRICS['collision_checks']}")
    print(f"Collisions found: {METRICS['collisions_found']}")
    print(f"Path length: {METRICS['path_length']:.4f} m")
    print("================================\n")
    
    return out

def visualize_path_animated(world, robot_left, robot_right, obj, path, T_obj_left_off, T_obj_right_off, dt=0.05):
    """
    Animierte Visualisierung des Objektpfads mit Endeffektorframes.
    dt = Zeit zwischen Frames in Sekunden
    """
    from klampt import vis
    import time
    import numpy as np
    vis.add("world", world)
    vis.setAttribute("world", "drawTransparency", 0.3)
    vis.setAttribute("world", "drawContacts", True)

    # Hilfsfunktion für Frames
    def draw_frame(T, name, size=0.05):
        origin = T[:3,3].tolist()
        x_dir = (T[:3,3] + T[:3,0]*size).tolist()
        y_dir = (T[:3,3] + T[:3,1]*size).tolist()
        z_dir = (T[:3,3] + T[:3,2]*size).tolist()
        vis.add("frame_"+name+"_x", [origin, x_dir], color=(1,0,0,1))
        vis.add("frame_"+name+"_y", [origin, y_dir], color=(0,1,0,1))
        vis.add("frame_"+name+"_z", [origin, z_dir], color=(0,0,1,1))

    # Animation
    for p in path:
        T_obj = posevec_to_T(p)
        Rm, t = T_to_se3(T_obj)
        obj.setTransform(Rm, t)

        # Frames für Endeffektoren
        T_left_ee  = T_obj @ T_obj_left_off
        T_right_ee = T_obj @ T_obj_right_off
        draw_frame(T_obj, "obj")
        draw_frame(T_left_ee, "left_ee")
        draw_frame(T_right_ee, "right_ee")

        vis.update()
        time.sleep(dt)

    vis.loop()

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ik_left_model", required=True)
    #ap.add_argument("--ik_right_model", required=True)
    ap.add_argument("--urdf_left", required=True)
    ap.add_argument("--urdf_right", required=True)
    ap.add_argument("--urdf_object", required=True)
    ap.add_argument("--urdf_obstacles", default="", help="Komma-getrennte URDFs von Hindernissen (als Roboter geladen)")
    ap.add_argument("--start_obj", required=True, help="x,y,z,qw,qx,qy,qz")
    ap.add_argument("--goal_obj",  required=True, help="x,y,z,qw,qx,qy,qz")
    ap.add_argument("--bounds", default=None, help="xmin,xmax,ymin,ymax,zmin,zmax (optional)")
    ap.add_argument("--time_limit", type=float, default=60.0)
    ap.add_argument("--goal_tol", type=float, default=0.01)
    ap.add_argument("--save_prefix", default="ompl_dual")
    ap.add_argument("--seed", type=int, default=42)
    # neues Argument
    ap.add_argument("--max_rot_deg", type=float, default=35.0,
                    help="Maximal erlaubte Abweichung der Objektrotation in Grad (Standard: 35°)")
    ap.add_argument("--ignore_links_left", default="", help="Komma-getrennte Linknamen, die bei Kollision (links) ignoriert werden")
    ap.add_argument("--ignore_links_right", default="", help="Komma-getrennte Linknamen, die bei Kollision (rechts) ignoriert werden")
    ap.add_argument("--ignore_links_obstacles", default="", help="Komma-getrennte Linknamen der Hindernisroboter, die ignoriert werden")
    ap.add_argument("--skip_object_vs_obstacles", action="store_true", help="Objekt gegen Hindernisse nicht prüfen")
    ap.add_argument("--collision_padding", type=float, default=0.0,
                    help="Erlaube Penetration bis zu diesem Wert (m), z.B. 0.01 für 1 cm Lockerung")
    ap.add_argument("--fast_exit", action="store_true",
                    help="Nutze os._exit(0), um Destruktor-bedingte double-free beim Shutdown zu vermeiden")
    ap.add_argument("--sv_resolution", type=float, default=0.02,
                    help="State-validity Auflösung (größer = weniger Checks, schneller)")
    ap.add_argument("--planner_range", type=float, default=None,
                    help="RRTConnect Range; wenn None wird OMPL-Autotuning genutzt")
    ap.add_argument("--goal_bias", type=float, default=None,
                    help="Goal bias für RRTConnect (0..1). Höher = schnellere, aber evtl. schlechtere Abdeckung")
    ap.add_argument("--interp_res", type=float, default=0.01,
                    help="Interpolation-Auflösung für Ausgabewegpunkte (m)")
    ap.add_argument("--skip_simplify", action="store_true", help="Pfad-Simplifier überspringen (schneller)")

    args = ap.parse_args()
    set_seed(args.seed)

    def parse_pose(s):
        v=[float(x) for x in s.split(",")]
        assert len(v)==7
        # (qw,qx,qy,qz) normalisieren
        qw,qx,qy,qz = normalize_quat(v[3],v[4],v[5],v[6])
        v[3],v[4],v[5],v[6] = qw,qx,qy,qz
        return v

    def parse_ignore_list(s):
        s = s.strip()
        if not s:
            return []
        return [x.strip() for x in s.split(",") if x.strip()]


    start_pose=parse_pose(args.start_obj)
    # Startrotation speichern
    T_start = posevec_to_T(start_pose)
    R_start = R.from_matrix(T_start[:3,:3])

    goal_pose=parse_pose(args.goal_obj)

    if args.bounds:
        b=[float(x) for x in args.bounds.split(",")]
        bounds=((b[0],b[1]),(b[2],b[3]),(b[4],b[5]))
    else:
        mins=np.minimum(start_pose[:3],goal_pose[:3])-0.4
        maxs=np.maximum(start_pose[:3],goal_pose[:3])+0.4
        bounds=((float(mins[0]),float(maxs[0])),
                (float(mins[1]),float(maxs[1])),
                (float(mins[2]),float(maxs[2])))

    print(">> IKFlow-Modelle laden …")
    ik_left,_  = get_ik_solver(args.ik_left_model)
    #ik_right,_ = get_ik_solver(args.ik_right_model)

    print(">> Objekt-URDF Offsets/Mesh …")
    T_obj_left_off, T_obj_right_off, obj_mesh_abs = parse_object_urdf_for_offsets_and_mesh(args.urdf_object)
    base_offset_right = ROBOT_TO_BASE_TRANSFORM["iiwa7_R"]

    print(">> Klampt Welt …")
    world=WorldModel()
    robot_left=world.loadRobot(args.urdf_left);  assert robot_left is not None
    robot_right=world.loadRobot(args.urdf_right); assert robot_right is not None

    # Objekt als RigidObject
    obj = world.makeRigidObject("grasped_object")
    if not obj.geometry().loadFile(obj_mesh_abs):
        raise RuntimeError(f"Konnte Mesh für Objekt nicht laden: {obj_mesh_abs}")
    obj.setTransform(*T_to_se3(posevec_to_T(start_pose)))

    obstacles=[]
    if args.urdf_obstacles:
        for path in args.urdf_obstacles.split(","):
            path = path.strip()
            if not path: 
                continue
            r = world.loadRobot(path)
            if r is None:
                raise RuntimeError(f"Hindernis-URDF konnte nicht geladen werden: {path}")
            obstacles.append(r)

    ignore_left = parse_ignore_list(args.ignore_links_left)
    ignore_right = parse_ignore_list(args.ignore_links_right)
    ignore_obs = parse_ignore_list(args.ignore_links_obstacles)

    checker = DualArmCollisionChecker(
        world, robot_left, robot_right, obj, obstacles,
        ignore_links_left=ignore_left,
        ignore_links_right=ignore_right,
        ignore_links_obstacles=ignore_obs,
        check_object_vs_obstacles=not args.skip_object_vs_obstacles,
        collision_padding=args.collision_padding,
    )
    # --- Pre-Check Start/Goal ---
    space = ob.SE3StateSpace()
    si_tmp = ob.SpaceInformation(space)  # nur für State-Objekterzeugung
    checker_fn = DualArmOMPLChecker(
        ik_left, checker, 
        T_obj_left_off, T_obj_right_off, R_start, max_rot_angle=np.deg2rad(args.max_rot_deg),
        base_offset_right=ROBOT_TO_BASE_TRANSFORM["iiwa7_R"])

    def mk_state_from_pose(pose):
        s = ob.State(space)
        s().setX(pose[0]); s().setY(pose[1]); s().setZ(pose[2])
        s().rotation().x = pose[4]; s().rotation().y = pose[5]
        s().rotation().z = pose[6]; s().rotation().w = pose[3]
        return s

    start_state = mk_state_from_pose(start_pose)
    goal_state  = mk_state_from_pose(goal_pose)

    def diagnose_pose(tag, pose):
        T_obj = posevec_to_T(pose)
        pose_left  = T_to_posevec(T_obj @ T_obj_left_off)
        T_ee_right_leftbasis = T_obj @ T_obj_right_off
        T_ee_right = np.linalg.inv(base_offset_right) @ T_ee_right_leftbasis
        pose_right = T_to_posevec(T_ee_right)
        (okL,qL),(okR,qR) = batch_ik_and_filter(ik_left,[pose_left])[0], batch_ik_and_filter(ik_left,[pose_right])[0]
        print(f"[CHECK] {tag}: IK_L={okL}, IK_R={okR}")
        if not okL or not okR:
            return False
        qL_full = expand_q_to_full(robot_left, qL)
        qR_full = expand_q_to_full(robot_right, qR)
        coll = checker.any_collision(qL_full, qR_full, T_obj)
        print(f"[CHECK] {tag}: collision={coll}")
        return (not coll)

    ok_start = diagnose_pose("START", start_pose)
    ok_goal  = diagnose_pose("GOAL", goal_pose)
    if not ok_start:
        print("!! Startpose ist ungültig (IK/Kollision).")
        sys.exit(2)
    if not ok_goal:
        print("!! Zielpose ist (so) ungültig (IK/Kollision). Versuche Toleranz/Orientierung anzupassen.")

    print(">> OMPL-Planung startet …")
    t0=time()
    path = plan_with_ompl(start_pose, goal_pose, bounds,
                          ik_left, checker,
                          T_obj_left_off, T_obj_right_off, R_start,
                          time_limit=args.time_limit,
                          simplify=not args.skip_simplify,
                          max_rot_deg=args.max_rot_deg,
                          sv_resolution=args.sv_resolution,
                          goal_bias=args.goal_bias,
                          interp_res=args.interp_res,
                          range_hint=args.planner_range if hasattr(args, "planner_range") else None)
    dt=time()-t0

    if path is None:
        print("!! Kein Pfad gefunden.")
        sys.exit(2)

    print(f">> Pfad gefunden in {dt:.2f}s, #Waypoints = {len(path)}")

    # Objektpfad speichern
    df_obj = pd.DataFrame(path, columns=["x","y","z","qw","qx","qy","qz"])
    df_obj.insert(0, "t", np.arange(len(path), dtype=int))
    obj_csv = f"{args.save_prefix}_object.csv"
    df_obj.to_csv(obj_csv, index=False)
    print("Gespeichert:", obj_csv)

    # Endeffektorpfade und Gelenkpfade entlang des Pfads
    mats=[posevec_to_T(p) for p in path]
    poses_left  = [T_to_posevec(m @ T_obj_left_off)  for m in mats]
    poses_right = [T_to_posevec(m @ T_obj_right_off) for m in mats]

    """# IK entlang des Pfades (je 1 Lösung pro Wegpunkt)
    resL = batch_ik_and_filter(ik_left, poses_left)
    resR = batch_ik_and_filter(ik_left, poses_right)

    qL_list=[]; qR_list=[]
    for (okL,qL),(okR,qR) in zip(resL,resR):
        if not(okL and okR):
            # Sollte selten passieren, da der State-Checker das schon filtert.
            # Zur Robustheit: abbrechen (oder alternativ: lokale Re-Planung)
            print("Warnung: IK-Versagen an einem Wegpunkt — Abbruch.")
            sys.exit(3)
        qL_list.append(qL); qR_list.append(qR)

    # CSVs
    left_csv  = f"{args.save_prefix}_ik_left.csv"
    right_csv = f"{args.save_prefix}_ik_right.csv"
    pd.DataFrame(qL_list).to_csv(left_csv, index=False, header=False)
    pd.DataFrame(qR_list).to_csv(right_csv, index=False, header=False)
    print("Gespeichert:", left_csv, right_csv)"""

    # Endeffektor-Posepfade (für Visualisierung)
    ee_left_csv  = f"{args.save_prefix}_ee_left.csv"
    ee_right_csv = f"{args.save_prefix}_ee_right.csv"
    df_left = pd.DataFrame(poses_left, columns=["x","y","z","qw","qx","qy","qz"])
    df_left.insert(0, "t", np.arange(len(poses_left), dtype=int))
    df_left.to_csv(ee_left_csv, index=False)
    df_right = pd.DataFrame(poses_right, columns=["x","y","z","qw","qx","qy","qz"])
    df_right.insert(0, "t", np.arange(len(poses_right), dtype=int))
    df_right.to_csv(ee_right_csv, index=False)
    print("Gespeichert:", ee_left_csv, ee_right_csv)

    print("Fertig.")

    # native Ressourcen freigeben
    try:
        world.destroy()
    except Exception:
        pass

    if args.fast_exit:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)

    #visualize_path_animated(world, robot_left, robot_right, obj, path, T_obj_left_off, T_obj_right_off, dt=0.05)

if __name__=="__main__":
    main()
