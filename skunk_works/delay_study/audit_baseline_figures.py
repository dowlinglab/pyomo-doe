"""Cross-check plotted arrays against independent authoritative case files.

This checks numeric content and source identity. It cannot check line styling,
visual overlap, readability, or scientific adequacy of the fitted models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--figures", type=Path, default=HERE/"manuscript/figures")
    args = parser.parse_args()
    plotted = json.loads((args.figures/"baseline_plot_data.json").read_text())
    manifest = json.loads((args.figures/"baseline_figure_manifest.json").read_text())
    for name, expected in manifest["inputs"].items():
        if sha(Path(name)) != expected:
            raise AssertionError("Figure source bytes changed: "+name)
    rmse_deltas, trajectory_deltas = [], []
    fit_identities = {}
    for calibration in ("sine", "step", "joint"):
        for initialization in ("cold", "input_equilibrated"):
            values = plotted["order_rmse__"+calibration+"__"+initialization]
            orders = [1, 2, 3, 4, 6, 8, 12, 0]
            if len(values) != len(orders):
                raise AssertionError("Unexpected curve shape")
            for value, order in zip(values, orders):
                family = "chain" if order else "pure_delay"
                path = args.results/"fits"/(f"{calibration}__{family}_n{order}_io_{initialization}_linear_exact.json")
                case = json.loads(path.read_text())
                fit_identities[path.name] = sha(path)
                rmse_deltas.append(abs(value-case["fit"]["rmse"]))
    for family, order in [("chain", 1), ("chain", 2), ("chain", 12), ("pure_delay", 0)]:
        case_path = args.results/"fits"/(f"joint__{family}_n{order}_io_cold_linear_exact.json")
        case = json.loads(case_path.read_text())
        prediction_path = args.results/case["prediction_file"]
        if sha(prediction_path) != case["prediction_sha256"]:
            raise AssertionError("Prediction source differs from fitted case")
        frame = pd.read_csv(prediction_path)
        for experiment in ("sine", "step"):
            source = frame[(frame.role == "calibration") & (frame.experiment == experiment)]
            values = plotted[family+str(order)+"__"+experiment]
            for column in ("time_s", "input_percent", "measured_C", "predicted_C"):
                actual, expected = np.asarray(values[column]), source[column].to_numpy()
                if actual.shape != expected.shape:
                    raise AssertionError("Trajectory shapes differ")
                trajectory_deltas.append(float(np.max(abs(actual-expected))))
    if max(rmse_deltas) > 1e-12 or max(trajectory_deltas) > 1e-12:
        raise AssertionError("Plotted values differ from case authorities")
    receipt = {"status": "numeric_content_and_source_hash_checks_passed",
               "producer": "audit_baseline_figures.py", "producer_sha256": sha(Path(__file__)),
               "producer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
               "rmse_points_checked": len(rmse_deltas), "trajectory_arrays_checked": len(trajectory_deltas),
               "maximum_rmse_delta_C": max(rmse_deltas), "maximum_trajectory_array_delta": max(trajectory_deltas),
               "case_sha256": fit_identities, "figure_manifest_sha256": sha(args.figures/"baseline_figure_manifest.json"),
               "figure_output_sha256": {p.name: sha(p) for stem in manifest["figures"] for p in args.figures.glob(stem+".*")},
               "limits": "Does not check typography, overlap, line style, interpretation, or statistical/model assumptions. Rendered visual review is separate."}
    (args.figures/"baseline_numeric_audit.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print("Verified 48 RMSE points and 32 trajectory arrays against saved case authorities.")


if __name__ == "__main__":
    main()
