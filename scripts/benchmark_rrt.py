#!/usr/bin/env python3
import subprocess
import pandas as pd
import argparse
import os
import re
from time import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_runs", type=int, default=10, help="Anzahl der Benchmarks")
    ap.add_argument("--script", type=str, default="scripts/rrt_ompl3.py", help="Pfad zum OMPL-Skript")
    ap.add_argument("--time_limit", type=float, default=10.0)
    ap.add_argument("--save_csv", type=str, default="benchmark_results.csv")
    ap.add_argument("--python_bin", type=str, default="python3", help="Python-Binary für Aufruf")
    # alles andere an Argumenten wird als String übergeben
    ap.add_argument("--args", type=str, default="", help="Zusätzliche Argumente für das OMPL-Skript")
    args = ap.parse_args()

    results = []

    # Regex für METRICS aus rrt_ompl3.py
    metric_re = {
        "validity_checks": re.compile(r"Validity checks:\s*(\d+)", re.I),
        "ik_calls_left": re.compile(r"IK calls left/right:\s*(\d+)\s*/\s*(\d+)", re.I),
        "ik_calls_right": re.compile(r"IK calls left/right:\s*(\d+)\s*/\s*(\d+)", re.I),
        "ik_success_left": re.compile(r"IK success left/right:\s*(\d+)\s*/\s*(\d+)", re.I),
        "ik_success_right": re.compile(r"IK success left/right:\s*(\d+)\s*/\s*(\d+)", re.I),
        "collision_checks": re.compile(r"Collision checks:\s*(\d+)", re.I),
        "collisions_found": re.compile(r"Collisions found:\s*(\d+)", re.I),
        "path_length": re.compile(r"Path length:\s*([0-9.]+)", re.I),
        "planner_time": re.compile(r"Pfad gefunden in\s*([0-9]+\.[0-9]+|[0-9]+)s", re.I),
    }

    def parse_metrics(stdout_text):
        out = {}
        for key, pattern in metric_re.items():
            m = pattern.search(stdout_text)
            if not m:
                continue
            if key in {"ik_calls_left", "ik_calls_right", "ik_success_left", "ik_success_right"}:
                # shared regex with two numbers
                out["ik_calls_left"] = int(m.group(1))
                out["ik_calls_right"] = int(m.group(2))
                continue
            if key in {"ik_success_left", "ik_success_right"}:
                continue
            if key in {"collision_checks", "collisions_found", "validity_checks"}:
                out[key] = int(m.group(1))
            elif key == "path_length":
                out[key] = float(m.group(1))
            elif key == "planner_time":
                out[key] = float(m.group(1))
        return out

    for run_idx in range(1, args.num_runs+1):
        print(f"\n=== Run {run_idx}/{args.num_runs} ===")
        save_prefix = f"run{run_idx}"
        cmd = f"{args.python_bin} {args.script} {args.args} --time_limit {args.time_limit} --save_prefix {save_prefix}"
        print("Running command:", cmd)
        t0 = time()
        try:
            # Skript ausführen
            proc = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            wall_dt = time() - t0

            # Pfad-CSV lesen, um Waypoints zu zählen
            obj_csv = f"{save_prefix}_object.csv"
            if os.path.exists(obj_csv):
                df = pd.read_csv(obj_csv)
                n_waypoints = len(df)
            else:
                n_waypoints = 0

            metrics = parse_metrics(proc.stdout)

            result_entry = {
                "run": run_idx,
                "success": n_waypoints > 0,
                "n_waypoints": n_waypoints,
                "runtime_s": wall_dt,
                "save_prefix": save_prefix,
                **metrics,
            }
            # Falls der Planer selbst die Zeit ausgibt, mitloggen
            if "planner_time" in metrics:
                result_entry["planner_time_s"] = metrics["planner_time"]
            results.append(result_entry)
        except subprocess.CalledProcessError as exc:
            wall_dt = time() - t0
            results.append({
                "run": run_idx,
                "success": False,
                "n_waypoints": 0,
                "runtime_s": wall_dt,
                "save_prefix": save_prefix,
            })
            print(f"Run {run_idx} failed. Stdout/Stderr: \n{getattr(exc, 'stdout', '')}\n{getattr(exc, 'stderr', '')}")

    # Alle Ergebnisse in CSV speichern
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.save_csv, index=False)
    print("\nBenchmark abgeschlossen. Ergebnisse gespeichert in", args.save_csv)
    print(df_results)

"""
python scripts/benchmark_rrt.py \
  --num_runs 5 \
  --time_limit 10 \
  --args "--ik_left_model iiwa7_left_arm \
          --urdf_left urdfs/iiwa7_L/iiwa7_L_updated.urdf \
          --urdf_right urdfs/iiwa7_R/iiwa7_R_updated.urdf \
          --urdf_object urdfs/object/se3_object.urdf \
          --urdf_obstacles urdfs/obstacle/obstacle.urdf \
          --start_obj 1.05,0.0,0.8,1,0,0,0 \
          --goal_obj  1.05,0.0,1.2,1,0,0,0 \
          --bounds 0.65,1.1,-0.2,0.2,0.6,1.5 \
          --max_rot_deg 35"

"""
if __name__ == "__main__":
    main()
