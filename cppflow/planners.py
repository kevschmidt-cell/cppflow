from abc import abstractmethod
from typing import List, Tuple, Dict
from time import time
import warnings

from ikflow.ikflow_solver import IKFlowSolver
from ikflow.model_loading import get_ik_solver
from ikflow.model import IkflowModelParameters, TINY_MODEL_PARAMS
from jrl.robot import Robot
from jrl.robots import Panda, Fetch, FetchArm, Iiwa7, Iiwa7_L, Iiwa7_R, Iiwa7_N
import torch

from cppflow.utils import (
    TimerContext,
    print_v1,
    print_v2,
    _plot_self_collisions,
    _plot_env_collisions,
    make_text_green_or_red,
)
from cppflow.config import (
    DEBUG_MODE_ENABLED,
    DEVICE,
    SUCCESS_THRESHOLD_initial_q_norm_dist,
    OPTIMIZATION_CONVERGENCE_THRESHOLD,
)
from cppflow.search import dp_search
from cppflow.optimization import run_lm_optimization
from cppflow.data_types import TimingData, PlannerSettings, PlannerResult, Problem
from cppflow.data_type_utils import plan_from_qpath
from cppflow.collision_detection import qpaths_batched_self_collisions, qpaths_batched_env_collisions
from cppflow.evaluation_utils import get_mjacs

import numpy as np

# Mapping: robot.name -> 4x4 numpy array describing the robot base in world coordinates
# If you don't know the matrix yet, set to np.eye(4) and later fill the correct values.


ROBOT_TO_IKFLOW_MODEL = {
    # --- Panda
    Panda.name: "panda__full__lp191_5.25m",
    # --- Fetch
    Fetch.name: "fetch_full_temp_nsc_tpm",
    # --- FetchArm
    FetchArm.name: "fetch_arm__large__mh186_9.25m",
    # --- Iiwa7
    Iiwa7.name: "iiwa7__full__lp191_5.25m",
    Iiwa7_L.name: "iiwa7_left_arm",
    #Iiwa7_L.name: "iiwa7_neutral",
    Iiwa7_R.name: "iiwa7_left_arm",
    Iiwa7_N.name: "iiwa7_neutral",
}

MOCK_IKFLOW_PARAMS = IkflowModelParameters()
SINGLE_PT_ZERO = torch.zeros(1)

#
DEFAULT_RERUN_NEW_K = 125  # the existing k configs will be added to this so no need to go overboard

import numpy as np

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
R = rpy_to_matrix(*rpy)
R_R = rpy_to_matrix(*rpy_R)

T_world_left = np.eye(4)
T_world_left[:3,:3] = R
T_world_left[:3, 3] = xyz

T_world_right = np.eye(4)
T_world_right[:3,:3] = R_R
T_world_right[:3, 3] = xyz_R

ROBOT_TO_BASE_TRANSFORM = {
    # example (identity = no base offset)
    "iiwa7": np.eye(4),
    "iiwa7_L": np.eye(4),
    #iiwa7_L": T_world_left, 
    "iiwa7_R": np.linalg.inv(T_world_left) @ T_world_right, #<- hilf mir hier
    "iiwa7_N": np.eye(4),
}
def pose7_to_mat(pose7: torch.Tensor) -> torch.Tensor:
    # pose7: [7] -> 4x4 matrix. pose format: [x,y,z,qw, qx,qy,qz]
    pos = pose7[:3]
    q = pose7[3:7]  # w,x,y,z
    w, x, y, z = q.unbind(0)
    # build rotation matrix (normalized quaternion assumed)
    # formula uses x,y,z,w (x,y,z vector part, w scalar)
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; xz = x * z; yz = y * z
    xw = x * w; yw = y * w; zw = z * w
    R = torch.stack([
        torch.stack([1 - 2*(yy+zz),     2*(xy - zw),     2*(xz + yw)]),
        torch.stack([    2*(xy + zw), 1 - 2*(xx+zz),     2*(yz - xw)]),
        torch.stack([    2*(xz - yw),     2*(yz + xw), 1 - 2*(xx+yy)])
    ])
    T = torch.eye(4, device=pose7.device, dtype=pose7.dtype)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T

def mat_to_pose7(T: torch.Tensor) -> torch.Tensor:
    # T: 4x4 -> pose7 [x,y,z, qx,qy,qz,qw]
    pos = T[:3, 3]
    R = T[:3, :3]
    # convert rotmat -> quat (x,y,z,w)
    m00 = R[0,0]; m01 = R[0,1]; m02 = R[0,2]
    m10 = R[1,0]; m11 = R[1,1]; m12 = R[1,2]
    m20 = R[2,0]; m21 = R[2,1]; m22 = R[2,2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / (trace + 1.0)**0.5
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    else:
        if (m00 > m11) and (m00 > m22):
            s = 2.0 * (1.0 + m00 - m11 - m22)**0.5
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * (1.0 + m11 - m00 - m22)**0.5
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * (1.0 + m22 - m00 - m11)**0.5
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
    quat = torch.tensor([w, x, y, z], device=R.device, dtype=T.dtype)
    quat = quat / torch.norm(quat)
    return torch.cat([pos, quat], dim=0)

def test_base_transform(ee_path: torch.Tensor, base_T: torch.Tensor):
    base_T_inv = torch.linalg.inv(base_T)
    n = ee_path.shape[0]
    transformed = torch.empty_like(ee_path)
    recovered = torch.empty_like(ee_path)
    pos_diffs = []
    quat_diffs = []

    for i in range(n):
        Tw = pose7_to_mat(ee_path[i])
        T_model = base_T_inv @ Tw
        transformed[i] = mat_to_pose7(T_model)

        # recover
        T_recovered = base_T @ pose7_to_mat(transformed[i])
        recovered[i] = mat_to_pose7(T_recovered)

        pos_diff = torch.norm(T_recovered[:3,3] - Tw[:3,3]).item()
        quat_diff = torch.norm(recovered[i][3:7] - ee_path[i][3:7]).item()
        pos_diffs.append(pos_diff)
        quat_diffs.append(quat_diff)

    return transformed, recovered, pos_diffs, quat_diffs

def slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between quaternions q0 -> q1 at fraction t"""
    # ensure normalized
    q0 = q0 / q0.norm()
    q1 = q1 / q1.norm()
    
    dot = torch.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # Linear interpolation for very close quats
        result = q0 + t*(q1 - q0)
        return result / result.norm()
    
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * t
    sin_theta = torch.sin(theta)

    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q0) + (s1 * q1)

def smooth_target_path_inplace(path7: torch.Tensor, alpha: float = 0.5):
    """
    Glättet inplace einen Pfad von Pose7s (x,y,z,qx,qy,qz,qw).
    - alpha: Glättungsfaktor (0=kein Glätten, 1=voll auf vorherige Pose setzen)
    """
    n = path7.shape[0]
    if n < 2:
        return

    # Kopien der Positionen und Quats
    pos = path7[:, :3].clone()
    quat = path7[:, 3:7].clone()

    for i in range(1, n):
        # Positionen glätten (Low-pass)
        pos[i] = alpha * pos[i-1] + (1-alpha) * pos[i]

        # Quaternions glätten via SLERP
        quat[i] = slerp(quat[i-1], quat[i], 1-alpha)

    # zurückschreiben inplace
    path7[:, :3] = pos
    path7[:, 3:7] = quat

def transform_right_to_left_ikflow_path(right_path7: torch.Tensor,
                                        T_world_left: np.ndarray,
                                        T_world_right: np.ndarray) -> torch.Tensor:
    """
    Transformiert einen rechten Arm-Pfad in Weltkoordinaten so, dass
    er vom IKFlow-Modell des linken Arms interpretiert werden kann.

    Args:
        right_path7: [N,7] Tensor von Pose7s des rechten Arms (x,y,z,qw,qx,qy,qz)
        T_world_left: 4x4 np.array Base-Transform linker Arm in Welt
        T_world_right: 4x4 np.array Base-Transform rechter Arm in Welt

    Returns:
        left_ikflow_path: [N,7] Tensor transformiert für linkes IKFlow-Modell
    """
    N = right_path7.shape[0]
    left_path = torch.empty_like(right_path7)

    T_world_left = torch.tensor(T_world_left, dtype=right_path7.dtype, device=right_path7.device)
    T_world_right = torch.tensor(T_world_right, dtype=right_path7.dtype, device=right_path7.device)
    T_diff = torch.linalg.inv(T_world_left) @ T_world_right

    for i in range(N):
        T_right = pose7_to_mat(right_path7[i])
        T_left = T_diff @ T_right
        left_path[i] = mat_to_pose7(T_left)

    return left_path

def add_search_path_mjac(debug_info: Dict, problem: Problem, qpath_search: torch.Tensor):
    mjac_deg, mjac_cm = get_mjacs(problem.robot, qpath_search)
    debug_info["search_path_mjac-cm"] = mjac_cm
    debug_info["search_path_mjac-deg"] = mjac_deg
    _, qps_prismatic = problem.robot.split_configs_to_revolute_and_prismatic(qpath_search)

    search_path_min_dist_to_jlim_cm = -1
    search_path_min_dist_to_jlim_deg = 10000

    for i, (l, u) in enumerate(problem.robot.actuated_joints_limits):
        if i == 0 and problem.robot.has_prismatic_joints:
            search_path_min_dist_to_jlim_cm = 100 * min(
                torch.min(torch.abs(qps_prismatic - l)).item(), torch.min(torch.abs(qps_prismatic - u)).item()
            )
            continue
        search_path_min_dist_to_jlim_deg = min(
            torch.rad2deg(torch.min(torch.abs(qpath_search[:, i] - l))).item(),
            torch.rad2deg(torch.min(torch.abs(qpath_search[:, i] - u))).item(),
            search_path_min_dist_to_jlim_deg,
        )
    debug_info["search_path_min_dist_to_jlim_cm"] = search_path_min_dist_to_jlim_cm
    debug_info["search_path_min_dist_to_jlim_deg"] = search_path_min_dist_to_jlim_deg


class Planner:
    def __init__(self, settings: PlannerSettings, robot: Robot, is_mock: bool = False):
        if not is_mock:
            self._ikflow_model_name = ROBOT_TO_IKFLOW_MODEL[robot.name]
            self._ikflow_solver, _ = get_ik_solver(self._ikflow_model_name, robot=robot)
        else:
            print("Warning: Using a mocked IKFlow solver - this model has random weights")
            self._ikflow_model_name = "none - mocked"
            self._ikflow_solver = IKFlowSolver(TINY_MODEL_PARAMS, robot)

        self._network_width = self._ikflow_solver.network_width
        self._cfg = settings

    def set_settings(self, settings: PlannerSettings):
        self._cfg = settings

    # Public methods
    @property
    def ikflow_model_name(self) -> str:
        return self._ikflow_model_name

    @property
    def robot(self) -> Robot:
        return self._ikflow_solver.robot

    @property
    def ikflow_solver(self) -> IKFlowSolver:
        return self._ikflow_solver

    @property
    def network_width(self) -> int:
        return self._network_width

    @property
    def name(self) -> str:
        return str(self.__class__.__name__)

    @abstractmethod
    def generate_plan(self, problem: Problem, **kwargs) -> PlannerResult:
        raise NotImplementedError()

    # Private methods
    def _sample_latents(self, k: int, n_timesteps: int) -> torch.Tensor:
        """Returns the latent vector for the IKFlow call.

        Notes:
            1. For 'uniform', latent_vector_scale is the width of the sampling area for each dimension. The value 2.0
                is recommended. This was found in a hyperparameter search in the 'search_scatchpad.ipynb' notebook on
                March 13.

        Args:
            k: Number of paths
            n_timesteps: Number of waypoints in the target pose path
        """
        shape = (k, self._network_width)
        if self._cfg.latent_distribution == "gaussian":
            latents = torch.randn(shape, device=DEVICE) * self._cfg.latent_vector_scale  # [k x network_width]
        elif self._cfg.latent_distribution == "uniform":
            width = self._cfg.latent_vector_scale
            latents = torch.rand(shape, device=DEVICE) * width - (width / 2)
        return torch.repeat_interleave(latents, n_timesteps, dim=0)

    def _sample_latents_near(self, k: int, n_timesteps: int, center_latent: torch.Tensor) -> torch.Tensor:
        """Returns the latent vector for the IKFlow call.

        Notes:
            1. For 'uniform', latent_vector_scale is the width of the sampling area for each dimension. The value 2.0
                is recommended. This was found in a hyperparameter search in the 'search_scatchpad.ipynb' notebook on
                March 13.

        Args:
            k: Number of paths
            n_timesteps: Number of waypoints in the target pose path
        """
        assert center_latent.numel() == self._network_width, "given latent should be same dim as network width"
        shape = (k, self._network_width)
        width = self._cfg.latent_vector_scale
        latents = torch.rand(shape, device=DEVICE) * width - (width / 2) + center_latent  # [k x network_width]
        latents[0] = center_latent
        return torch.repeat_interleave(latents, n_timesteps, dim=0)

    def _get_k_ikflow_qpaths(
        self,
        ee_path: torch.Tensor,
        batched_latent: torch.Tensor,
        k: int,
        clamp_to_joint_limits: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Returns k different config space paths for the given ee_path."""
        n = ee_path.shape[0]
        with torch.inference_mode(), TimerContext("running IKFlow", enabled=self._cfg.verbosity > 0):
            ee_path_tiled = ee_path.repeat((k, 1))
            ikf_sols = self._ikflow_solver.generate_ik_solutions(
                ee_path_tiled,
                latent=batched_latent,
                clamp_to_joint_limits=clamp_to_joint_limits,
            )
        paths = [ikf_sols[i * n : (i * n) + n, :] for i in range(k)]
        return paths

    def _get_configuration_corresponding_latent(self, qs: torch.Tensor, ee_pose: torch.Tensor) -> torch.Tensor:
        """Get the latent vectors that corresponds to the given configurations"""
        with torch.inference_mode(), TimerContext(
            "running IKFlow in reverse to get latent for initial_configuration", enabled=self._cfg.verbosity > 0
        ):
            if self.robot.ndof != self._ikflow_solver.network_width:
                model_input = torch.cat(
                    [qs.view(1, self.robot.ndof), torch.zeros(1, self._ikflow_solver.network_width - self.robot.ndof)],
                    dim=1,
                )
            else:
                model_input = qs
            assert len(model_input.shape) == 2, f"Model input should be 2D tensor, is {model_input.shape}"
            conditional = torch.cat([ee_pose.view(1, 7), SINGLE_PT_ZERO.view(1, 1)], dim=1)
            output_rev, _ = self._ikflow_solver.nn_model(model_input, c=conditional, rev=False)
            return output_rev

    def _run_pipeline(self, problem: Problem, **kwargs):
        """Runs IKFlow, collision checking, and search with debug for self-collisions."""
        existing_q_data = kwargs.get("rerun_data", None)
        if "initial_q_latent" not in kwargs:
            kwargs["initial_q_latent"] = None

        # --- IKFlow ---
        t0_ikflow = time()
        k = self._cfg.k if existing_q_data is None else DEFAULT_RERUN_NEW_K

        if problem.initial_configuration is not None and kwargs["initial_q_latent"] is None:
            kwargs["initial_q_latent"] = self._get_configuration_corresponding_latent(
                problem.initial_configuration, problem.target_path[0]
            )

        if kwargs["initial_q_latent"] is not None:
            batched_latents = self._sample_latents_near(
                k, problem.n_timesteps, kwargs["initial_q_latent"]
            )
        else:
            batched_latents = self._sample_latents(k, problem.n_timesteps)

        # --- Transform target_path ---
        robot_name = problem.robot.name
        base_T_np = ROBOT_TO_BASE_TRANSFORM[robot_name]
        base_T = torch.tensor(base_T_np, dtype=problem.target_path.dtype, device=problem.target_path.device)
        base_T_inv = torch.linalg.inv(base_T)

        # ee_path wird automatisch transformiert
        ee_path_for_ikflow = torch.stack([
            mat_to_pose7(base_T_inv @ pose7_to_mat(p)) for p in problem.target_path
        ])


        # --- Run IKFlow ---
        ikflow_qpaths = self._get_k_ikflow_qpaths(ee_path_for_ikflow, batched_latents, k)
        time_ikflow = time() - t0_ikflow

        # --- DEBUG: Self-Collisions direkt nach IKFlow ---
        qs = torch.stack(ikflow_qpaths)
        from cppflow.collision_detection import qpaths_batched_self_collisions
        violations = qpaths_batched_self_collisions(problem, qs)
        pct_colliding_post_ikflow = (torch.sum(violations) / violations.numel()).item() * 100
        print(f"[DEBUG] Self-collisions after IKFlow: {pct_colliding_post_ikflow:.2f}%")

        # --- Optionally save initial solution ---
        if self._cfg.return_only_1st_plan:
            return (
                ikflow_qpaths[0],
                False,
                TimingData(-1, time_ikflow, 0.0, 0.0, 0.0, 0.0),
                {},
                (ikflow_qpaths[0], None, None),
            )

        # --- Collision checking ---
        t0_col_check = time()
        k_current = qs.shape[0]

        try:
            with TimerContext("calculating self-colliding configs", enabled=self._cfg.verbosity > 0):
                self_collision_violations = violations
                pct_colliding = pct_colliding_post_ikflow
                if pct_colliding >= 95.0:
                    print(f"[WARNING] too many self collisions: {pct_colliding:.2f}% - continuing for debug purposes")
                else:
                    print(f"  self_collision violations: {pct_colliding:.2f}%")
        except AssertionError:
            print("[DEBUG] Skipping assertion due to high self-collisions")

        with TimerContext("calculating env-colliding configs", enabled=self._cfg.verbosity > 0):
            env_collision_violations = qpaths_batched_env_collisions(problem, qs)
            pct_env_colliding = (torch.sum(env_collision_violations) / env_collision_violations.numel()).item() * 100
            print(f"  env_collision violations: {pct_env_colliding:.2f}%")

        # --- Continue rest of pipeline ---
        if existing_q_data is not None:
            qs_prev, self_collision_violations_prev, env_collision_violations_prev = existing_q_data
            qs = torch.cat([qs_prev, qs], dim=0)
            self_collision_violations = torch.cat([self_collision_violations_prev, self_collision_violations], dim=0)
            env_collision_violations = torch.cat([env_collision_violations_prev, env_collision_violations], dim=0)

        if problem.initial_configuration is not None:
            k_current = qs.shape[0]
            qs[:, 0, :] = problem.initial_configuration
            self_collision_violations[:, 0] = 0.0
            env_collision_violations[:, 0] = 0.0

        time_coll_check = time() - t0_col_check
        debug_info = {}

        # --- DP Search ---
        t0_dp_search = time()
        with TimerContext(f"running dynamic programming search with qs: {qs.shape}", enabled=self._cfg.verbosity > 0):
            qpath_search = dp_search(self.robot, qs, self_collision_violations, env_collision_violations).to(DEVICE)
        time_dp_search = time() - t0_dp_search

        q_data = (qs, self_collision_violations, env_collision_violations)

        return (
            qpath_search,
            False,
            TimingData(-1, time_ikflow, time_coll_check, 0.0, time_dp_search, 0.0),
            debug_info,
            q_data,
        )



# ----------------------------------------------------------------------------------------------------------------------
# ---
# --- Planners
# ---


class PlannerSearcher(Planner):
    """PlannerSearcher creates a finds a solution by performing a search through a graph constructed by connecting k
    ikflow generated cspace plans
    """

    def __init__(self, settings: PlannerSettings, robot: Robot):
        super().__init__(settings, robot)
        assert self._cfg.run_dp_search

    def generate_plan(self, problem: Problem, **kwargs) -> PlannerResult:
        """Runs dp_search and returns"""
        assert problem.robot.name == self.robot.name

        t0 = time()
        qpath_search, _, td, debug_info, _ = self._run_pipeline(problem, **kwargs)

        # rerun dp_search with larger k if mjac is too high
        if self._cfg.do_rerun_if_large_dp_search_mjac:
            mjac_deg, mjac_cm = get_mjacs(problem.robot, qpath_search)
            if mjac_deg > self._cfg.rerun_mjac_threshold_deg or mjac_cm > self._cfg.rerun_mjac_threshold_cm:
                print_v1(
                    f"\nRerunning dp_search with larger k b/c mjac is too high: {mjac_deg} deg, {mjac_cm} cm",
                    verbosity=self._cfg.verbosity,
                )
                qpath_search, _, td, debug_info, _ = self._run_pipeline(problem, **kwargs)
                mjac_deg, mjac_cm = get_mjacs(problem.robot, qpath_search)
                print_v1(f"new mjac after dp_search with larger k: {mjac_deg} deg,  cm", verbosity=self._cfg.verbosity)

        time_total = time() - t0
        return PlannerResult(
            plan_from_qpath(qpath_search.detach(), problem),
            TimingData(time_total, td.ikflow, td.coll_checking, td.batch_opt, td.dp_search, 0.0),
            [],
            [],
            debug_info,
        )


class CppFlowPlanner(Planner):
    # Generalized planner that runs dp_search before LM optimization. Must specify the optimization version to use

    def __init__(self, settings: PlannerSettings, robot: Robot):
        super().__init__(settings, robot)

    def generate_plan(self, problem: Problem, **kwargs) -> PlannerResult:

        t0 = time() if "t0" not in kwargs else kwargs["t0"]

        rerun_data = kwargs["rerun_data"] if "rerun_data" in kwargs else None
        results_df = kwargs["results_df"] if "results_df" in kwargs else None
        search_qpath, is_valid, td, debug_info, q_data = self._run_pipeline(problem, **kwargs)

        def time_is_exceeded():
            return time() - t0 > self._cfg.tmax_sec

        def return_(qpath):
            return PlannerResult(
                plan_from_qpath(qpath, problem),
                TimingData(time() - t0, td.ikflow, td.coll_checking, td.batch_opt, td.dp_search, 0.0),
                [],
                [],
                debug_info,
            )

        # Optionally return only the 1st plan
        if self._cfg.return_only_1st_plan:
            return PlannerResult(
                plan_from_qpath(search_qpath, problem), TimingData(time() - t0, 0, 0, 0, 0, 0), [], [], {}
            )

        # rerun dp_search with larger k if mjac is too high
        if self._cfg.do_rerun_if_large_dp_search_mjac:
            mjac_deg, mjac_cm = get_mjacs(problem.robot, search_qpath)
            if mjac_deg > self._cfg.rerun_mjac_threshold_deg or mjac_cm > self._cfg.rerun_mjac_threshold_cm:
                print_v1(
                    f"{ make_text_green_or_red('Rerunning', False)} dp_search with larger k b/c mjac is too high:"
                    f" {mjac_deg} deg, {mjac_cm} cm",
                    verbosity=self._cfg.verbosity,
                )
                kwargs["rerun_data"] = q_data
                search_qpath, is_valid, td, debug_info, q_data = self._run_pipeline(problem, **kwargs)
                mjac_deg, mjac_cm = get_mjacs(problem.robot, search_qpath)
                print_v1(f"new mjac after dp_search with larger k: {mjac_deg} deg,  cm", verbosity=self._cfg.verbosity)

        # return if not anytime mode and search path is valid, or out of time
        if time_is_exceeded():
            print_v2(
                f"Time limit exceeded after dp_search ({time() - t0:.3f} > {self._cfg.tmax_sec}), returning",
                verbosity=self._cfg.verbosity,
            )
            return return_(search_qpath)
        if (not self._cfg.anytime_mode_enabled) and is_valid:
            print_v2("dp_search path is valid and anytime mode is disabled, returning", verbosity=self._cfg.verbosity)
            return return_(search_qpath)

        # Run optimization
        # TODO(@jstmn): Handle the `initial_configuration` during optimization. This should be a fixed value that
        # impacts the gradient of the trajectory.
        with TimerContext("running run_lm_optimization()", enabled=self._cfg.verbosity > 0):
            t0_opt = time()
            if self._cfg.anytime_mode_enabled:
                optimization_result = run_lm_optimization(
                    problem,
                    search_qpath,
                    max_n_steps=75,
                    tmax_sec=self._cfg.tmax_sec - (time() - t0),
                    return_if_valid_after_n_steps=int(1e8),
                    convergence_threshold=OPTIMIZATION_CONVERGENCE_THRESHOLD,
                    results_df=results_df,
                    verbosity=self._cfg.verbosity,
                )
            else:
                optimization_result = run_lm_optimization(
                    problem,
                    search_qpath,
                    max_n_steps=20,
                    tmax_sec=self._cfg.tmax_sec - (time() - t0),
                    return_if_valid_after_n_steps=0,
                    convergence_threshold=1e6,
                    results_df=results_df,
                    verbosity=self._cfg.verbosity,
                )
            td.optimizer = time() - t0_opt
            debug_info["n_optimization_steps"] = optimization_result.n_steps_taken
        x_opt = optimization_result.x_opt.detach()

        # update convergence result
        if "results_df" in kwargs:
            problem.write_qpath_to_results_df(kwargs["results_df"], x_opt)

        if optimization_result.is_valid:
            if problem.initial_configuration is None:
                return return_(x_opt)

            # Check to see how far x_opt[0] is from the initial configuration
            initial_q_norm_dist = torch.norm(problem.initial_configuration - x_opt[0])
            if initial_q_norm_dist < SUCCESS_THRESHOLD_initial_q_norm_dist:
                return return_(x_opt)

            print_v2(
                f"'initial_configuration' is too far from x_opt[0] ({initial_q_norm_dist} <"
                f" {SUCCESS_THRESHOLD_initial_q_norm_dist})",
                verbosity=self._cfg.verbosity,
            )
            x_opt_swapped = torch.cat((problem.initial_configuration, x_opt[1:]), dim=0)
            assert torch.norm(problem.initial_configuration - x_opt_swapped[0]) < 1e-6
            plan_from_xopt_swapped = plan_from_qpath(x_opt_swapped, problem)
            if plan_from_xopt_swapped.is_valid:
                print_v2(
                    "Valid trajectory found by swapping initial_configuration and x_opt[0], returning",
                    verbosity=self._cfg.verbosity,
                )
                return return_(x_opt_swapped)

            print_v2(
                "Invalid trajectory found when swapping initial_configuration and x_opt[0], returning original trajectory",
                verbosity=self._cfg.verbosity,
            )
            return return_(x_opt)

        # Optionally rerun if optimization failed
        if self._cfg.do_rerun_if_optimization_fails and (rerun_data is None) and (not time_is_exceeded()):
            print_v1("\nRerunning dp_search because optimization failed", verbosity=self._cfg.verbosity)
            kwargs["rerun_data"] = q_data
            kwargs["t0"] = t0
            return self.generate_plan(problem, **kwargs)

        return return_(x_opt)
