"""Bind the nine Tier 0 checks to saved fits and produce compact comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import Experiment
from fit_diagnostics import harmonic_residual
from run_mean import write_json
from study_cases import CASES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [("primary", args.reference)]+[(name, args.input/name) for name in CASES]
    rows = []
    bindings = {}
    nstarts = 0
    failed_starts = 0
    failed_attempts = 0
    for name, folder in paths:
        fit_path = folder/"fit.json"
        result = json.loads(fit_path.read_text())
        bindings[str(fit_path)] = hashlib.sha256(fit_path.read_bytes()).hexdigest()
        config = json.loads((folder/"effective_config.json").read_text())
        arrays = np.load(folder/"trajectories.npz")
        bindings[str(folder/"trajectories.npz")] = hashlib.sha256((folder/"trajectories.npz").read_bytes()).hexdigest()
        if json.loads((folder/"exit.json").read_text())["exit_code"] != 0:
            raise AssertionError("Nonterminal or failed sensitivity case")
        if name != "primary":
            if result["audit"]["status"] != "passed":
                raise AssertionError("Sensitivity reconstruction audit missing")
            nstarts += len(result["starts"])
            failed_starts += sum(not s["success"] for s in result["starts"])
            failed_attempts += sum(not a["success"] for s in result["starts"] for a in s.get("solver_attempts", []))
        fast = []
        for j, record in enumerate(result["record_names"]):
            if "sin_50_50" not in record:
                continue
            observed = arrays["observed_"+str(j)]
            predicted = arrays["predicted_"+str(j)]
            data = Experiment(record, arrays["time_"+str(j)], observed, arrays["input_"+str(j)], observed[0])
            harmonic = harmonic_residual(data, (observed-predicted)[1:])
            innovation = arrays["standardized_innovation_"+str(j)]
            command = arrays["input_"+str(j)][1:]
            from diagnostics import input_correlation
            correlations = input_correlation(innovation, command)
            maximum = max(abs(v["correlation"]) for v in correlations if v["correlation"] is not None)
            fast.append({"record": record, "max_abs_input_correlation": maximum,
                         "residual_harmonic_amplitude_C": harmonic["amplitude_C"]})
        best = result["best"]
        ambient = best["coefficients"][1:]
        bounds = config["ambient_offset_bounds_C"]
        row = {"case": name, "nrecords": len(result["record_names"]), "K": best["coefficients"][0],
               "tau_s": best["tau"], "theta_s": best["theta"],
               "unlabeled_slow_s": max(best["tau"], best["theta"]),
               "unlabeled_fast_s": min(best["tau"], best["theta"]),
               "pole_sum_s": best["tau"]+best["theta"], "pole_product_s2": best["tau"]*best["theta"],
               "negative_log_likelihood": best["negative_log_likelihood"],
               "ambient_range_C": [min(ambient), max(ambient)],
               "ambient_terms_at_bound": sum(abs(v-bounds[0]) < 1e-6 or abs(v-bounds[1]) < 1e-6 for v in ambient),
               "sigma_total_ML_C": best["sigma_total_ML_C"], "noise_parameters": best["noise_parameters"],
               "stationarity_warning": result["stationarity_warning"], "fast_sine": fast,
               "record_names": result["record_names"]}
        rows.append(row)
    primary = rows[0]
    for row in rows:
        same = row["record_names"] == primary["record_names"]
        row["same_observations_as_primary"] = same
        row["nll_change_from_primary_same_observations_only"] = row["negative_log_likelihood"]-primary["negative_log_likelihood"] if same else None
    labels = {"primary": "Primary", "right_hold": "Right hold", "initial_minus": "Initial -0.165 C",
              "initial_plus": "Initial +0.165 C", "preload_one_second": "1 s prehistory",
              "ambient_four_degrees": "Ambient +/-4 C", "protocol_30s": "30 s protocol only"}
    selected = [row for row in rows if row["case"] in labels]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7})
    fig, axes = plt.subplots(1, 2, figsize=(5, 3.3), sharey=True)
    colors = ["#165c85", "#b24a23", "#455b2d"]
    for i, row in enumerate(selected):
        for j, run in enumerate(row["fast_sine"]):
            axes[0].plot(run["max_abs_input_correlation"], i, "o", color=colors[j], markersize=4)
            axes[1].plot(run["residual_harmonic_amplitude_C"], i, "o", color=colors[j], markersize=4,
                         label="Run "+str(j+1) if i == 0 else None)
    axes[0].set_yticks(range(len(selected)))
    axes[0].set_yticklabels([labels[row["case"]] for row in selected])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Max |input correlation|")
    axes[1].set_xlabel("Residual harmonic, C")
    axes[0].set_xlim(0, .6)
    axes[1].set_xlim(0, .3)
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
    handles, run_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, run_labels, loc="upper center", bbox_to_anchor=(.7, 1.), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .92))
    image_path = args.output/"fast_sine_sensitivity.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    write_json(args.output/"summary.json", {"source_bindings": bindings, "cases": rows,
               "joint_start_level_solutions": nstarts, "unsuccessful_start_solutions": failed_starts,
               "unsuccessful_solver_attempts_retained": failed_attempts,
               "scope": "Nine declared Tier 0 diagnostic sensitivities; only identical-observation cases have likelihood changes. No new calibration-pair enumeration or scientific structure selection."})
    write_json(args.output/"figure_receipt.json", {"source_bindings": bindings,
               "figure_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(), "visual_review": "pending"})


if __name__ == "__main__":
    main()
