from cppflow.data_type_utils import problem_from_filename
from cppflow.data_types import Constraints, PlannerSettings
from cppflow.planners import CppFlowPlanner
from cppflow.visualization import visualize_dual_plan

CONSTRAINTS = Constraints(
    max_allowed_position_error_cm=0.01,  # 0.1mm
    max_allowed_rotation_error_deg=0.1,
    max_allowed_mjac_deg=7.0,  # from the paper
    max_allowed_mjac_cm=2.0,  # from the paper
)
planner_settings_dict = {
        "CppFlow": PlannerSettings(
            verbosity=2,
            k=175,
            tmax_sec=5.0,
            anytime_mode_enabled=False,
            do_rerun_if_large_dp_search_mjac=True,
            do_rerun_if_optimization_fails=False,
            do_return_search_path_mjac=False,
        ),
        "CppFlow_fixed_q0": PlannerSettings(
            verbosity=2,
            k=175,
            tmax_sec=3.0,
            anytime_mode_enabled=False,
            latent_vector_scale=0.5,
            do_rerun_if_large_dp_search_mjac=False,
            do_rerun_if_optimization_fails=False,
            do_return_search_path_mjac=False,
        ),
        "PlannerSearcher": PlannerSettings(
            k=175,
            tmax_sec=5.0,
            anytime_mode_enabled=False,
        ),
    }

planner_settings = (
        planner_settings_dict["CppFlow"]
    )

class DualArmProblem:
    def __init__(self, problem_left, problem_right):
        self.problem_left = problem_left
        self.problem_right = problem_right
        assert problem_left.n_timesteps == problem_right.n_timesteps, "Time horizons must match"
        self.n_timesteps = problem_left.n_timesteps


class DualArmPlanner:
    def __init__(self, planner_left, planner_right):
        self.left = planner_left
        self.right = planner_right

    def plan(self, dual_problem):
        trajectory_left = self.left.generate_plan(dual_problem.problem_left)
        trajectory_right = self.right.generate_plan(dual_problem.problem_right)
        return trajectory_left, trajectory_right


def main():
    # Load individual problems
    problem_left = problem_from_filename(CONSTRAINTS, "iiwa7_L__flappy_bird")
    problem_right = problem_from_filename(CONSTRAINTS, "iiwa7_R__flappy_bird")

    # Wrap in dual-arm problem
    dual_problem = DualArmProblem(problem_left, problem_right)

    # Initialize planners
    planner_left = CppFlowPlanner(planner_settings,problem_left.robot)
    planner_right = CppFlowPlanner(planner_settings,problem_right.robot)

    dual_planner = DualArmPlanner(planner_left, planner_right)
    traj_left, traj_right = dual_planner.plan(dual_problem)

    # Visualize
    visualize_dual_plan(
        traj_left, problem_left, traj_right,
        problem_right
    ) 


if __name__ == "__main__":
    main()
