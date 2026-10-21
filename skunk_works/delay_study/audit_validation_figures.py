"""Reconstruct validation errors and OU ACFs from saved prediction trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import load
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--figures", type=Path, default=HERE/"manuscript/figures")
    args = parser.parse_args()
    manifest = json.loads((args.figures/"validation_figure_manifest.json").read_text())
    path = args.figures/"validation_plot_data.json"
    if sha(path) != manifest["plot_data_sha256"]:
        raise RuntimeError("Validation figure data changed")
    for name, expected in manifest["input_sha256"].items():
        if sha(Path(name)) != expected:
            raise RuntimeError("Validation figure input changed")
    table_path = Path(manifest["generated_table"]["path"])
    if sha(table_path) != manifest["generated_table"]["sha256"]:
        raise RuntimeError("Validation file table changed")
    summary_path = Path(manifest["generated_summary_table"]["path"])
    if sha(summary_path) != manifest["generated_summary_table"]["sha256"]:
        raise RuntimeError("Validation summary table changed")
    summary_lines = [line for line in summary_path.read_text().splitlines() if line.startswith("$n=") or line.startswith("Pure delay & ")]
    if len(summary_lines) != 4:
        raise RuntimeError("Validation summary table lost a model")
    data = json.loads(path.read_text())
    maximum, total, inputs = 0., 0, {}

    def compare(actual, expected, tolerance=1e-9):
        nonlocal maximum, total
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape:
            raise RuntimeError("Validation audit shape mismatch")
        error = float(np.max(abs(a-b)))
        if error > tolerance:
            raise RuntimeError("Validation independent numeric difference: "+str(error))
        total += a.size
        maximum = max(maximum, error)

    for j, (case, values) in enumerate(data["validation_errors"].items()):
        record = json.loads((args.results/"fits"/(case+".json")).read_text())
        prediction = args.results/record["prediction_file"]
        if sha(prediction) != record["prediction_sha256"]:
            raise RuntimeError("Baseline prediction changed")
        inputs[str(prediction)] = sha(prediction)
        frame = pd.read_csv(prediction)
        total_sse, bias_sse, N = 0., 0., 0
        for item in values["files"]:
            group = frame[(frame.role == "validation") & (frame.experiment == item["dataset"])]
            error = (group.predicted_C-group.measured_C).to_numpy()[1:]
            compare([item["n"], item["rmse_C"], item["bias_C"], item["max_abs_error_C"]],
                    [len(error), np.sqrt(np.mean(error**2)), error.mean(), np.max(abs(error))])
            total_sse += float(error@error)
            bias_sse += len(error)*float(error.mean())**2
            N += len(error)
        compare([values["pooled_rmse_C"], values["per_file_bias_fraction_of_pooled_SSE"]], [np.sqrt(total_sse/N), bias_sse/total_sse])
        calibration = frame[frame.role == "calibration"]
        retained = pd.concat([group.iloc[1:] for _, group in calibration.groupby("experiment")])
        error = (retained.predicted_C-retained.measured_C).to_numpy()
        calibration_rmse = float(np.sqrt(np.mean(error**2)))
        compare(values["calibration_rmse_C"], calibration_rmse)
        cells = [f"{calibration_rmse:.4f}", f"{np.sqrt(total_sse/N):.4f}", f"{100*bias_sse/total_sse:.1f}"]
        actual = [cell.strip().removesuffix(r" \\").strip() for cell in summary_lines[j].split(" & ")[1:]]
        if cells != actual:
            raise RuntimeError("Printed validation summary cell changed")
    table_lines = table_path.read_text().splitlines()
    for item in data["validation_file_map"]:
        raw = load(item["file"])
        compare([item["duration_s"], item["conditioned_initial_C"], item["residual_count"]],
                [raw.time[-1]-raw.time[0], raw.temperature[0], len(raw.time)-1])
        expected_cells = [f"{raw.time[-1]-raw.time[0]:.1f}", f"{raw.temperature[0]:.2f}", str(len(raw.time)-1)]
        line = next(line for line in table_lines if line.startswith(item["code"]+" & "))
        cells = [cell.strip().removesuffix(r" \\").strip() for cell in line.split(" & ")[2:]]
        if cells != expected_cells or item["file"].removeprefix("validation_experiment_") not in line:
            raise RuntimeError("Printed validation metadata changed")
    for case, values in data["ou_correlations"].items():
        record = json.loads((args.results/"ou"/(case+".json")).read_text())
        prediction = args.results/"ou"/record["prediction_file"]
        if sha(prediction) != record["prediction_sha256"]:
            raise RuntimeError("OU prediction changed")
        inputs[str(prediction)] = sha(prediction)
        frame = pd.read_csv(prediction)
        for experiment, acfs in values.items():
            group = frame[(frame.role == "calibration") & (frame.experiment == experiment)]
            residual = (group.predicted_C-group.measured_C).to_numpy()[1:]
            time = group.time_s.to_numpy()[1:]
            phi = np.exp(-np.diff(time)/record["fit"]["correlation_length_s"])
            innovation = np.r_[residual[0], (residual[1:]-phi*residual[:-1])/np.sqrt(1-phi**2)]
            for name, vector in [("raw_C", residual), ("innovation_C", innovation)]:
                centered = vector-vector.mean()
                expected = [centered[lag:]@centered[:-lag]/(centered@centered) for lag in range(1, 31)]
                compare(acfs[name], expected)
    result = {"status": "passed", "numeric_values_checked": total, "maximum_absolute_difference": maximum,
              "input_prediction_sha256": inputs, "figure_data_sha256": manifest["plot_data_sha256"],
              "producer_sha256": sha(HERE/"audit_validation_figures.py"),
              "notice": "Validation errors, bias decomposition, printed file metadata and OU ACF arrays recomputed through independent prediction/raw-data paths. Plot and rendered-page inspection remain separate; file labels do not establish independent experiments."}
    save_json(args.figures/"validation_numeric_audit.json", result)
    print(json.dumps({key:value for key,value in result.items() if key != "input_prediction_sha256"}))


if __name__ == "__main__":
    main()
