import argparse
from datetime import datetime
import pandas as pd

from cppflow.data_type_utils import problem_from_filename
from cppflow.data_types import Constraints, PlannerSettings
from cppflow.planners import CppFlowPlanner
from cppflow.visualization import visualize_dual_plan

PD_COLUMN_NAMES = [
    "Arm",
    "Problem",
    "Robot",
    "Planner",
    "Valid plan",
    "time, total (s)",
    "time, ikflow (s)",
    "time, coll_checking (s)",
    "time, batch_opt (s)",
    "time, dp_search (s)",
    "time, optimizer (s)",
    "time per opt. step (s)",
    "Max positional error (mm)",
    "Max rotational error (deg)",
    "Mean positional error (mm)",
    "Mean rotational error (deg)",
    "Mjac - prismatic (cm)",
    "Mjac - revolute (deg)",
]

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

planner_settings = planner_settings_dict["CppFlow"]

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
        result_left = self.left.generate_plan(dual_problem.problem_left)
        result_right = self.right.generate_plan(dual_problem.problem_right)
        return result_left, result_right


def row_from_result(result, problem, arm_label: str):
    # Some planners may not populate optimizer timing/debug info
    n_opt_steps = max(1, int(result.debug_info.get("n_optimization_steps", 1))) if hasattr(result, "debug_info") else 1
    time_per_opt_step = (result.timing.optimizer / n_opt_steps) if n_opt_steps > 0 else 0.0
    round_amt = 5
    return [
        arm_label,
        problem.fancy_name,
        problem.robot.name,
        result.plan.debug_info.get("planner_name", "CppFlow") if hasattr(result.plan, "debug_info") else "CppFlow",
        f"`{str(result.plan.is_valid).lower()}`",
        round(result.timing.total, 4),
        round(result.timing.ikflow, 4),
        round(result.timing.coll_checking, 4),
        round(result.timing.batch_opt, 4),
        round(result.timing.dp_search, 4),
        round(result.timing.optimizer, 4),
        round(time_per_opt_step, 4),
        round(result.plan.max_positional_error_mm, round_amt),
        round(result.plan.max_rotational_error_deg, round_amt),
        round(result.plan.mean_positional_error_mm, round_amt),
        round(result.plan.mean_rotational_error_deg, round_amt),
        round(result.plan.mjac_cm, round_amt),
        round(result.plan.mjac_deg, round_amt),
    ]


def main():
    parser = argparse.ArgumentParser(description="Dual Arm Planner")
    parser.add_argument(
        "--problem", type=str, default="flappy_bird",
        help="Problem name suffix, e.g. 'flappy_bird', 'circle', etc."
    )
    parser.add_argument("--save_csv", type=str, default=None, help="Optional: CSV-Datei für Metriken pro Arm")
    args = parser.parse_args()

    # Dynamisch Problem-Namen zusammenbauen
    problem_left_name = f"iiwa7_L__{args.problem}"
    problem_right_name = f"iiwa7_R__{args.problem}"

    # Probleme laden
    problem_left = problem_from_filename(CONSTRAINTS, problem_left_name)
    problem_right = problem_from_filename(CONSTRAINTS, problem_right_name)

    dual_problem = DualArmProblem(problem_left, problem_right)

    planner_left = CppFlowPlanner(planner_settings, problem_left.robot)
    planner_right = CppFlowPlanner(planner_settings, problem_right.robot)

    dual_planner = DualArmPlanner(planner_left, planner_right)
    result_left, result_right = dual_planner.plan(dual_problem)

    # Metriken wie in evaluate.py
    rows = [
        row_from_result(result_left, problem_left, "left"),
        row_from_result(result_right, problem_right, "right"),
    ]

    # Summierte Laufzeiten (Arme laufen nacheinander). Nicht-plan-spezifische Felder leer lassen.
    timing_fields = [
        "time, total (s)",
        "time, ikflow (s)",
        "time, coll_checking (s)",
        "time, batch_opt (s)",
        "time, dp_search (s)",
        "time, optimizer (s)",
        "time per opt. step (s)",
    ]
    sums = {f: rows[0][PD_COLUMN_NAMES.index(f)] + rows[1][PD_COLUMN_NAMES.index(f)] for f in timing_fields}

    # Gemittelte Qualitätsmetriken über beide Arme
    avg_fields = [
        "Max positional error (mm)",
        "Max rotational error (deg)",
        "Mean positional error (mm)",
        "Mean rotational error (deg)",
        "Mjac - prismatic (cm)",
        "Mjac - revolute (deg)",
    ]
    avgs = {f: (rows[0][PD_COLUMN_NAMES.index(f)] + rows[1][PD_COLUMN_NAMES.index(f)]) / 2.0 for f in avg_fields}

    valid_combined = "`true`" if (rows[0][4] == "`true`" and rows[1][4] == "`true`") else "`false`"
    combined_row = [
        "combined",
        "|".join([problem_left.fancy_name, problem_right.fancy_name]),
        "dual",
        "CppFlow",
        valid_combined,
        sums["time, total (s)"],
        sums["time, ikflow (s)"],
        sums["time, coll_checking (s)"],
        sums["time, batch_opt (s)"],
        sums["time, dp_search (s)"],
        sums["time, optimizer (s)"],
        sums["time per opt. step (s)"],
        avgs["Max positional error (mm)"],
        avgs["Max rotational error (deg)"],
        avgs["Mean positional error (mm)"],
        avgs["Mean rotational error (deg)"],
        avgs["Mjac - prismatic (cm)"],
        avgs["Mjac - revolute (deg)"],
    ]
    rows.append(combined_row)
    df = pd.DataFrame(rows, columns=PD_COLUMN_NAMES)
    print("\n=== Metrics ===")
    print(df)
    if args.save_csv:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = args.save_csv if args.save_csv.endswith(".csv") else f"{args.save_csv}_{ts}.csv"
        df.to_csv(out, index=False)
        print("Gespeichert:", out)

    visualize_dual_plan(result_left, problem_left, result_right, problem_right)


if __name__ == "__main__":
    main()
