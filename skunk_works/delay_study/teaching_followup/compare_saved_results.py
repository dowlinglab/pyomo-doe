"""Answer the teaching comparison from saved fits and bootstrap results."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def main():
    inputs = {}

    def read_fit(calibration, model):
        suffix = "original_n0_physics" if model == "original" else f"chain_n{model}_io"
        path = RESULTS / f"fits/{calibration}__{suffix}_cold_linear_exact.json"
        inputs[str(path.relative_to(HERE.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    rows, joint = [], {}
    for calibration in ["sine", "step", "joint"]:
        for model in ["original", 1, 2]:
            fit = read_fit(calibration, model)
            validation = fit["validation"]
            pooled = np.sqrt(sum(v["n"] * v["rmse_C"]**2 for v in validation) / sum(v["n"] for v in validation))
            rows.append({"calibration": calibration, "model": str(model),
                         "calibration_rmse_C": fit["fit"]["rmse"], "pooled_validation_rmse_C": float(pooled)})
            if calibration == "joint":
                joint[str(model)] = fit
    for model in ["original", "1"]:
        path = RESULTS / joint[model]["prediction_file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == joint[model]["prediction_sha256"]
        inputs[str(path.relative_to(HERE.parent))] = digest
    original = pd.read_csv(RESULTS / joint["original"]["prediction_file"])
    reduced = pd.read_csv(RESULTS / joint["1"]["prediction_file"])
    assert original[["role", "experiment", "time_s"]].equals(reduced[["role", "experiment", "time_s"]])
    prediction_difference = {}
    for role, group in original.groupby("role"):
        prediction_difference[role] = float(np.max(np.abs(group.predicted_C - reduced.loc[group.index].predicted_C)))

    path = RESULTS / "bootstrap/summary.csv"
    inputs[str(path.relative_to(HERE.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    bootstrap = pd.read_csv(path)
    intervals = []
    for order in [1, 2]:
        selected = bootstrap[(bootstrap.calibration == "joint") & (bootstrap.order == order)
                             & (bootstrap.initialization == "cold") & (bootstrap.coordinates == "io")
                             & (bootstrap.parameter == "theta")].set_index("method")
        item = {"order": order}
        for method in ["block1", "block10", "block30", "block60", "parametric"]:
            row = selected.loc[method]
            item[method] = {"lower_s": float(row.lower), "upper_s": float(row.upper),
                            "width_s": float(row.upper - row.lower), "sd_s": float(row["std"])}
        item["block60_to_block1_width_ratio"] = item["block60"]["width_s"] / item["block1"]["width_s"]
        intervals.append(item)
    out = {"calibration_and_validation": rows,
           "independently_fitted_joint_n1_vs_original_max_prediction_difference_C": prediction_difference,
           "delay_intervals": intervals, "input_sha256": inputs,
           "scope": "Cold fits. Validation is pooled over eleven duplicate-excluded files, without refitting. Bootstrap comparisons use the same centered empirical residuals for block1 and block60; block lengths are retained samples. One-stage intervals select the canonical short-delay pole label, excluding its global swapped mode. Parametric simulations instead use fixed hypothetical sigma=0.25 C, so their comparison changes marginal scale as well as dependence."}
    destination = HERE / "results/saved_comparison.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
