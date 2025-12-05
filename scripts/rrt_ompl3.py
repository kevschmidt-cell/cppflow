"""python scripts/rrt_ompl3.py \
  --urdf_left urdfs/iiwa7_L/iiwa7_L_updated.urdf \
  --urdf_right urdfs/iiwa7_R/iiwa7_R_updated.urdf \
  --urdf_object urdfs/object/se3_object.urdf \
  --urdf_obstacles "urdfs/obstacle/passage.urdf,urdfs/obstacle/passage2" \
  --start_obj "1.05,0.0,0.9,1,0,0,0" \
  --goal_obj  "0.8,0.0,0.9,1,0,0,0" \
  --bounds "0.65,1.1,-0.2,0.2,0.6,1.4" \
  --time_limit 10 \
  --max_rot_deg 35 \
  --save_prefix run1 """

#!/usr/bin/env python3
import argparse, os, sys, xml.etree.ElementTree as ET
import math
from time import time
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# OMPL
from ompl import base as ob
from ompl import geometric as og

# Optional: Seed für Reproduzierbarkeit (aus deinem IKFlow-Setup)
from ikflow.utils import set_seed

# Klampt
from klampt import WorldModel
from klampt.model import collide, ik
from klampt.math import se3, so3
from klampt.robotsim import RobotModelLink, RigidObjectModel, IKSolver

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

# ---------------------------
# Utils
# ---------------------------
def normalize_quat(w, x, y, z):
    q = np.array([w, x, y, z], dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return 1.0, 0.0, 0.0, 0.0
    q /= n
    if q[0] < 0:
        q = -q
    return tuple(q.tolist())

def xyzrpy_to_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = np.asarray(xyz, float)
    return T

def T_to_posevec(T):
    """4x4 -> [x,y,z,qw,qx,qy,qz]"""
    p = T[:3, 3]
    qx, qy, qz, qw = R.from_matrix(T[:3, :3]).as_quat()  # (x,y,z,w)
    return [float(p[0]), float(p[1]), float(p[2]), float(qw), float(qx), float(qy), float(qz)]

def posevec_to_T(v):
    """[x,y,z,qw,qx,qy,qz] -> 4x4"""
    x, y, z, qw, qx, qy, qz = map(float, v)
    T = np.eye(4)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T

def T_to_se3(T):
    Rm = T[:3, :3].T
    t = T[:3, 3]
    return (Rm.reshape(-1).tolist(), t.tolist())

def parse_object_urdf_for_offsets_and_mesh(urdf_object_path,
                                           child_left="obj_p_01",
                                           child_right="obj_p_02",
                                           base="obj_com"):
    tree = ET.parse(urdf_object_path)
    root = tree.getroot()

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
        raise RuntimeError(f"Kein Joint zu child '{child_name}' gefunden.")

    T_obj_to_left = find_origin_to(child_left)
    T_obj_to_right = find_origin_to(child_right)

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

from klampt.model import ik
from klampt.math import so3
from klampt.robotsim import IKSolver

def klampt_ik_solver(klampt_robot_model, ee_link_index, pose_vec,
                     positional_tolerance=1e-3, n_tries=50):
    """
    Numerische IK mit Klampt.
    pose_vec: [x,y,z,qw,qx,qy,qz]
    ee_link_index: int, Index des EE-Links (z.B. robot.numLinks()-1)
    Gibt volle Klampt-Konfiguration (list) oder None zurück.
    """
    assert len(pose_vec) == 7
    t = pose_vec[0:3]
    qw, qx, qy, qz = pose_vec[3:]
    Robj = so3.from_quaternion([qx, qy, qz, qw])  # Klampt: (x,y,z,w)

    ee_link = klampt_robot_model.link(ee_link_index)

    obj = ik.objective(ee_link, t=t, R=Robj)

    for _ in range(n_tries):
        solver = IKSolver(klampt_robot_model)
        solver.add(obj)
        solver.setMaxIters(150)
        solver.setTolerance(positional_tolerance)
        solver.sampleInitial()
        if solver.solve():
            # Vollständige Konfiguration zurückgeben
            return klampt_robot_model.getConfig()
    return None

def batch_ik_and_filter_klampt(robot, poses_batch, ee_link_index):
    """
    Wrapper: Liste von (ok, q_full) für jede Pose.
    q_full ist direkt die volle Klampt-Konfiguration (robot.getConfig()).
    """
    out = []
    for pose in poses_batch:
        q = klampt_ik_solver(robot, ee_link_index, pose)
        ok = q is not None
        out.append((ok, q if q is not None else [0.0]*robot.numLinks()))
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
    def __init__(self, world, robot_left, robot_right, grasped_object, obstacles_robots):
        self.world = world
        self.left = robot_left
        self.right = robot_right
        self.obj = grasped_object
        self.obstacles = obstacles_robots
        self.cw = collide.WorldCollider(world)

        self.left_id = self.left.getID()
        self.right_id = self.right.getID()
        self.obs_ids = {r.getID() for r in self.obstacles}
        self.obj_id = self.obj.getID()

    def _is_links_of(self, robot_id):
        def f(o):
            return isinstance(o, RobotModelLink) and o.robot().getID() == robot_id
        return f

    def _is_links_of_obstacles(self, o):
        return isinstance(o, RobotModelLink) and (o.robot().getID() in self.obs_ids)

    def _is_the_object(self, o):
        return isinstance(o, RigidObjectModel) and (o.getID() == self.obj_id)

    def set_object_T(self, T_obj):
        Rm, t = T_to_se3(T_obj)
        self.obj.setTransform(Rm, t)

    def any_collision(self, qL, qR, T_obj):
        self.left.setConfig(qL)
        self.right.setConfig(qR)
        self.set_object_T(T_obj)
        if any(self.cw.collisions(self._is_links_of(self.left_id), self._is_links_of_obstacles)):
            return True
        if any(self.cw.collisions(self._is_links_of(self.right_id), self._is_links_of_obstacles)):
            return True
        if any(self.cw.collisions(self._is_the_object, self._is_links_of_obstacles)):
            return True
        return False

# ---------------------------
# OMPL State Validity Checker
# ---------------------------
class DualArmOMPLChecker:
    def __init__(self,
                 robot_left,
                 robot_right,
                 ee_left_index,
                 ee_right_index,
                 checker,
                 T_obj_left_off,
                 T_obj_right_off,
                 R_start,
                 max_rot_angle=np.deg2rad(35)):
        self.robot_left = robot_left
        self.robot_right = robot_right
        self.ee_left_index = ee_left_index
        self.ee_right_index = ee_right_index
        self.checker = checker
        self.T_obj_left_off = T_obj_left_off
        self.T_obj_right_off = T_obj_right_off
        self.R_start = R_start
        self.max_rot_angle = max_rot_angle


    def __call__(self, state):
        METRICS["validity_checks"] += 1

        # Objektpose aus State
        xyz = [state.getX(), state.getY(), state.getZ()]
        rot = state.rotation()
        quat = [rot.x, rot.y, rot.z, rot.w]  # (x,y,z,w)
        T_obj = posevec_to_T([*xyz, quat[3], quat[0], quat[1], quat[2]])

        # Rotationsabweichung prüfen
        R_curr = R.from_matrix(T_obj[:3, :3])
        R_rel = self.R_start.inv() * R_curr
        if R_rel.magnitude() > self.max_rot_angle:
            return False

        # IK-Zielposes
        pose_left  = T_to_posevec(T_obj @ self.T_obj_left_off)
        pose_right = T_to_posevec(T_obj @ self.T_obj_right_off)

        METRICS["ik_calls_left"]  += 1
        METRICS["ik_calls_right"] += 1

        okL, qL = batch_ik_and_filter_klampt(self.robot_left,  [pose_left],  self.ee_left_index)[0]
        okR, qR = batch_ik_and_filter_klampt(self.robot_right, [pose_right], self.ee_right_index)[0]

        if okL: METRICS["ik_success_left"]  += 1
        if okR: METRICS["ik_success_right"] += 1
        if not (okL and okR):
            return False

        # Kollisionen
        METRICS["collision_checks"] += 1
        if self.checker.any_collision(qL, qR, T_obj):
            METRICS["collisions_found"] += 1
            return False

        return True


# ---------------------------
# OMPL Planung
# ---------------------------
def plan_with_ompl(start_pose, goal_pose, bounds,
                   robot_left, robot_right,
                   ee_left_index, ee_right_index,
                   checker,
                   T_obj_left_off, T_obj_right_off, R_start,
                   time_limit=60.0, simplify=True, range_hint=None, max_rot_deg=35):


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
        DualArmOMPLChecker(robot_left, robot_right,
                           ee_left_index, ee_right_index,
                           checker,
                           T_obj_left_off, T_obj_right_off,
                           R_start, max_rot_angle=np.deg2rad(max_rot_deg))
    ))

    si.setStateValidityCheckingResolution(0.02)
    si.setup()

    def set_state_from_pose(state_obj, pose):
        x, y, z, qw, qx, qy, qz = pose
        n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        if n == 0:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        else:
            qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
        state_obj().setX(x)
        state_obj().setY(y)
        state_obj().setZ(z)
        rot = state_obj().rotation()
        rot.x, rot.y, rot.z, rot.w = qx, qy, qz, qw  # OMPL quat is (x,y,z,w)

    # Start- und Zielzustand
    start = ob.State(space)
    goal = ob.State(space)
    set_state_from_pose(start, start_pose)
    set_state_from_pose(goal, goal_pose)

    def pose_within_bounds(pose, bounds_tuple):
        for i in range(3):
            if pose[i] < bounds_tuple[i][0] or pose[i] > bounds_tuple[i][1]:
                return False
        return True

    if (not pose_within_bounds(start_pose, bounds)) or (not pose_within_bounds(goal_pose, bounds)):
        raise ValueError("Start/Goal liegen außerhalb der Bounds; passe --bounds an oder erweitere sie.")

    # Problemdefinition
    pdef = ob.ProblemDefinition(si)
    goal_tolerance = 0.01  # Position ~1 cm, Rotation ~2°
    pdef.setStartAndGoalStates(start, goal, goal_tolerance)

    pdef.setOptimizationObjective(
        ob.PathLengthOptimizationObjective(si)
    )
    planner = og.RRTConnect(si)

    # Optional: Initial Stepsize / Range
    if range_hint is not None:
        planner.setRange(float(range_hint))

    # Wichtig bei RRT* – die Heuristik aktivieren
    #planner.setGoalBias(0.05)        # optional, Standard ~0.05
    #planner.setPruneThreshold(0.1)   # erlaubt aggressiveres Pruning
    #planner.setKNearest(False)       # True = k-nearest, False = r-disc (manchmal schneller)

    planner.setProblemDefinition(pdef)
    planner.setup()

    solved = planner.solve(time_limit)
    if not solved:
        return None

    path_geom = pdef.getSolutionPath()
    if simplify:
        og.PathSimplifier(si).simplifyMax(path_geom)

    # dynamische Interpolation: z. B. alle 0.01 m ein Wegpunkt
    resolution = 0.01  # 1 cm
    n_states = int(path_geom.length() / resolution)
    if n_states > 0:
        path_geom.interpolate(n_states)

    # Pfadlänge direkt aus OMPL
    METRICS["path_length"] = path_geom.length()

    # States -> Posevec (x,y,z, qw,qx,qy,qz)
    out = []
    for s in path_geom.getStates():
        xyz = [s.getX(), s.getY(), s.getZ()]
        r = s.rotation()
        out.append([*xyz, r.w, r.x, r.y, r.z])

    # explizit Ressourcen freigeben (OMPL/C++), bevor wir Python beenden
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

# ---------------------------
# CLI / main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf_left", required=True)
    ap.add_argument("--urdf_right", required=True)
    ap.add_argument("--urdf_object", required=True)
    ap.add_argument("--urdf_obstacles", default="",
                    help="Komma-getrennte URDFs von Hindernissen (als Roboter geladen)")
    ap.add_argument("--start_obj", required=True, help="x,y,z,qw,qx,qy,qz")
    ap.add_argument("--goal_obj",  required=True, help="x,y,z,qw,qx,qy,qz")
    ap.add_argument("--bounds", default=None, help="xmin,xmax,ymin,ymax,zmin,zmax (optional)")
    ap.add_argument("--time_limit", type=float, default=60.0)
    ap.add_argument("--goal_tol", type=float, default=0.01)
    ap.add_argument("--save_prefix", default="ompl_dual")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rot_deg", type=float, default=35.0,
                    help="Maximal erlaubte Abweichung der Objektrotation in Grad (Standard: 35°)")
    ap.add_argument("--fast_exit", action="store_true",
                    help="Benutze os._exit(0) statt regulärem Python-Shutdown (vermeidet double free)")

    args = ap.parse_args()
    set_seed(args.seed)

    def parse_pose(s):
        v = [float(x) for x in s.split(",")]
        assert len(v) == 7
        qw, qx, qy, qz = normalize_quat(v[3], v[4], v[5], v[6])
        v[3], v[4], v[5], v[6] = qw, qx, qy, qz
        return v

    start_pose = parse_pose(args.start_obj)
    goal_pose = parse_pose(args.goal_obj)

    # Startrotation für Winkelbeschränkung
    T_start = posevec_to_T(start_pose)
    R_start = R.from_matrix(T_start[:3, :3])

    if args.bounds:
        b = [float(x) for x in args.bounds.split(",")]
        bounds = ((b[0], b[1]), (b[2], b[3]), (b[4], b[5]))
    else:
        mins = np.minimum(start_pose[:3], goal_pose[:3]) - 0.4
        maxs = np.maximum(start_pose[:3], goal_pose[:3]) + 0.4
        bounds = ((float(mins[0]), float(maxs[0])),
                  (float(mins[1]), float(maxs[1])),
                  (float(mins[2]), float(maxs[2])))

    def include_points_in_bounds(bounds_tuple, pts, pad=1e-4):
        b = [list(bounds_tuple[0]), list(bounds_tuple[1]), list(bounds_tuple[2])]
        changed = False
        for p in pts:
            for i, v in enumerate(p):
                if v < b[i][0]:
                    b[i][0] = v - pad
                    changed = True
                if v > b[i][1]:
                    b[i][1] = v + pad
                    changed = True
        return tuple(tuple(x) for x in b), changed

    bounds, expanded = include_points_in_bounds(bounds, [start_pose[:3], goal_pose[:3]])
    if expanded:
        print("Hinweis: Bounds wurden erweitert, damit Start/Ziel enthalten sind:", bounds)

    print(">> Objekt-URDF Offsets/Mesh …")
    T_obj_left_off, T_obj_right_off, obj_mesh_abs = parse_object_urdf_for_offsets_and_mesh(args.urdf_object)

    print(">> Klampt Welt …")
    world = WorldModel()
    robot_left = world.loadRobot(args.urdf_left);  assert robot_left is not None
    robot_right = world.loadRobot(args.urdf_right); assert robot_right is not None

    # Objekt als RigidObject
    obj = world.makeRigidObject("grasped_object")
    if not obj.geometry().loadFile(obj_mesh_abs):
        raise RuntimeError(f"Konnte Mesh für Objekt nicht laden: {obj_mesh_abs}")
    obj.setTransform(*T_to_se3(posevec_to_T(start_pose)))

    obstacles = []
    if args.urdf_obstacles:
        for path in args.urdf_obstacles.split(","):
            path = path.strip()
            if not path:
                continue
            r = world.loadRobot(path)
            if r is None:
                raise RuntimeError(f"Hindernis-URDF konnte nicht geladen werden: {path}")
            obstacles.append(r)

    checker = DualArmCollisionChecker(world, robot_left, robot_right, obj, obstacles)

    # EE-Link-Namen: ggf. auf deine URDF anpassen!
    ee_left_index  = robot_left.link("lbr1_true_ee_link").getIndex()
    ee_right_index = robot_right.link("lbr2_true_ee_link").getIndex()
    print("EE-left index =", ee_left_index)
    print("EE-right index =", ee_right_index)

    # --- Diagnose Start/Ziel ---
    def diagnose_pose(tag, pose):
        T_obj = posevec_to_T(pose)
        pose_left  = T_to_posevec(T_obj @ T_obj_left_off)
        pose_right = T_to_posevec(T_obj @ T_obj_right_off)

        okL, qL = batch_ik_and_filter_klampt(robot_left, [pose_left], ee_left_index)[0]
        okR, qR = batch_ik_and_filter_klampt(robot_right, [pose_right], ee_right_index)[0]
        print(f"[CHECK] {tag}: IK_L={okL}, IK_R={okR}")
        if not (okL and okR):
            return False

        coll = checker.any_collision(qL, qR, T_obj)
        print(f"[CHECK] {tag}: collision={coll}")
        return not coll

    print("Initialized robot collision data structures in time ???")  # optional Timing

    ok_start = diagnose_pose("START", start_pose)
    ok_goal  = diagnose_pose("GOAL", goal_pose)

    if not ok_start:
        print("!! Startpose ist ungültig (IK/Kollision).")
        sys.exit(2)
    if not ok_goal:
        print("!! Zielpose ist (so) ungültig (IK/Kollision). Versuche Toleranz/Orientierung anzupassen.")

    print(">> OMPL-Planung startet …")
    t0 = time()
    path = plan_with_ompl(start_pose, goal_pose, bounds,
                          robot_left, robot_right, ee_left_index, ee_right_index,
                          checker,
                          T_obj_left_off, T_obj_right_off, R_start,
                          time_limit=args.time_limit, simplify=True,
                          max_rot_deg=args.max_rot_deg)
    dt = time() - t0

    if path is None:
        print("!! Kein Pfad gefunden.")
        sys.exit(2)

    print(f">> Pfad gefunden in {dt:.2f}s, #Waypoints = {len(path)}")

    # Objektpfad speichern
    df_obj = pd.DataFrame(path, columns=["x", "y", "z", "qw", "qx", "qy", "qz"])
    df_obj.insert(0, "t", np.arange(len(path), dtype=int))
    obj_csv = f"{args.save_prefix}_object.csv"
    df_obj.to_csv(obj_csv, index=False)
    print("Gespeichert:", obj_csv)

    # Endeffektorpfade für Visualisierung
    mats = [posevec_to_T(p) for p in path]
    poses_left  = [T_to_posevec(m @ T_obj_left_off)  for m in mats]
    poses_right = [T_to_posevec(m @ T_obj_right_off) for m in mats]

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

    # Native Ressourcen sauber freigeben, um "invalid pointer" am Interpreter-Ende zu vermeiden
    try:
        world.destroy()
    except Exception:
        pass

    # Einige Python/C++ Destruktoren (OMPL/Klampt) scheinen beim Interpreter-Shutdown
    # sporadisch double-free auszulösen. optional: hartes _exit.
    if args.fast_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

if __name__ == "__main__":
    main()
