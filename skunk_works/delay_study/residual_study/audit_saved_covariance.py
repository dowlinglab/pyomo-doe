"""Audit every saved covariance fit against raw data and dense kernel likelihood."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from scipy.linalg import cholesky, solve_triangular

from audit_mean import reference
from data import load
from run_mean import write_json
from simulation import audit as audit_generation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/"generation_audit.json", audit_generation())
    receipts = []
    summary = []
    for fit_path in sorted(args.input.glob("*/fit.json")):
        folder = fit_path.parent
        fit = json.loads(fit_path.read_text())
        source = json.loads((folder/"source_snapshot.json").read_text())
        config = json.loads((folder/"effective_config.json").read_text())
        best = fit["best"]
        coefficient = np.asarray(best["coefficients"])
        width = 1 if fit["mode"] == "constant" else 2
        records = [load(name) for name in fit["record_names"]]
        arrays = np.load(folder/"trajectories.npz")
        n = 0
        quadratic = 0.
        logdet = 0.
        maximum_trajectory_error = 0.
        maximum_innovation_error = 0.
        rows = []
        for j, data in enumerate(records):
            raw_path = HERE.parents[2]/data.source
            if hashlib.sha256(raw_path.read_bytes()).hexdigest() != source["sha256"][data.source]:
                raise AssertionError("Raw-data binding mismatch")
            if not np.array_equal(data.time, arrays["time_"+str(j)]) or not np.array_equal(data.temperature, arrays["observed_"+str(j)]):
                raise AssertionError("Trajectory observations or timestamps differ")
            ambient = coefficient[1+j*width:1+(j+1)*width]
            endpoints = np.repeat(ambient, 2) if width == 1 else ambient
            if np.any(np.abs(endpoints) > 2+1e-8):
                raise AssertionError("Ambient bounds violated")
            predicted = reference(data, coefficient[0], best["tau"], best["theta"], endpoints, fit["hold"])
            trajectory_error = float(np.max(np.abs(predicted-arrays["predicted_"+str(j)])))
            maximum_trajectory_error = max(maximum_trajectory_error, trajectory_error)
            residual = data.temperature[1:]-predicted[1:]
            count = len(residual)
            time = data.time[1:]
            distance = np.abs(time[:, None]-time[None, :])
            if fit["kernel"] == "iid":
                C = np.eye(count)
            else:
                ell = best["noise_parameters"][0]
                eta = best["noise_parameters"][1] if fit["kernel"].endswith("_nugget") else 0.
                if fit["kernel"] == "matern32_nugget":
                    a = np.sqrt(3.)*distance/ell
                    C = (1+a)*np.exp(-a)
                else:
                    C = np.exp(-distance/ell)
                C = (1-eta)*C+eta*np.eye(count)
            L = cholesky(C, lower=True)
            innovation = solve_triangular(L, residual, lower=True)
            standardized = innovation/best["sigma_total_ML_C"]
            innovation_error = float(np.max(np.abs(standardized-arrays["standardized_innovation_"+str(j)])))
            maximum_innovation_error = max(maximum_innovation_error, innovation_error)
            quadratic += float(innovation@innovation)
            logdet += float(2*np.log(np.diag(L)).sum())
            n += count
            rows.append({"record": data.name, "mean_error_C": trajectory_error,
                         "standardized_innovation_error": innovation_error})
        variance = quadratic/n
        nll = .5*(n*(np.log(2*np.pi)+1+np.log(variance))+logdet)
        nll_error = abs(nll-best["negative_log_likelihood"])
        scale_error = abs(np.sqrt(variance)-best["sigma_total_ML_C"])
        if maximum_trajectory_error > 1e-7 or maximum_innovation_error > 2e-5 or nll_error > 2e-5 or scale_error > 1e-7:
            raise AssertionError((folder.name, maximum_trajectory_error, maximum_innovation_error, nll_error, scale_error))
        if n != fit["nobs"]:
            raise AssertionError("Observation count mismatch")
        p = fit["nominal_parameter_count"]
        AICc = 2*nll+2*p+2*p*(p+1)/(n-p-1)
        if abs(AICc-fit["AICc_nominal"]) > 4e-5:
            raise AssertionError("AICc arithmetic mismatch")
        alternate = json.loads((folder/"alternating.json").read_text())
        advantage = alternate["history"][-1]["negative_log_likelihood"]-nll
        if abs(advantage-fit["alternating_minus_joint_nll"]) > 2e-5:
            raise AssertionError("GLS/joint comparison arithmetic mismatch")
        receipts.append({"cell": folder.name, "fit_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
                         "trajectories_sha256": hashlib.sha256((folder/"trajectories.npz").read_bytes()).hexdigest(),
                         "producer_commit": source["commit"], "nobs": n,
                         "nll_abs_error": nll_error, "sigma_abs_error_C": scale_error,
                         "logdet_abs_error": abs(logdet-fit["logdet_unit_shape"]),
                         "max_trajectory_error_C": maximum_trajectory_error,
                         "max_standardized_innovation_error": maximum_innovation_error,
                         "per_run": rows})
        innovation_acf1 = [r["innovation_acf_1_to_30"][0] for r in fit["per_run"]]
        innovation_acf_rest = [max(abs(v) for v in r["innovation_acf_1_to_30"]) for r in fit["per_run"]]
        summary.append({"cell": folder.name, "mode": fit["mode"], "kernel": fit["kernel"],
                        "K": coefficient[0], "tau": best["tau"], "theta": best["theta"],
                        "negative_log_likelihood": nll, "AICc_nominal": AICc,
                        "raw_RMSE_C": fit["raw_RMSE_C"], "sigma_total_ML_C": best["sigma_total_ML_C"],
                        "noise_parameters": best["noise_parameters"],
                        "innovation_acf1_range": [min(innovation_acf1), max(innovation_acf1)],
                        "largest_abs_acf_1_to_30": max(innovation_acf_rest),
                        "runs_reference_p30_below_005": sum(r["ljung_box_reference_p_30"] < .05 for r in fit["per_run"]),
                        "nominal_pvalue_notice": "Descriptive fitted-parameter reference only; simulation-calibrated assessment remains pending",
                        "mean_innovation_std_range": [min(r["innovation_std"] for r in fit["per_run"]), max(r["innovation_std"] for r in fit["per_run"])],
                        "failed_starts": sum(not s["success"] for s in fit["starts"]),
                        "gradient_max_abs": best["projected_gradient_max_abs"],
                        "stationarity_warning": fit["stationarity_warning"],
                        "alternating_converged": alternate["converged"],
                        "alternating_minus_joint_nll": advantage})
    if len(receipts) != 8:
        raise AssertionError("Expected all eight pooled covariance cells, found "+str(len(receipts)))
    write_json(args.output/"audit.json", {"status": "passed", "cells": receipts,
               "scope": "All eight selected trajectories and dense-covariance likelihoods, raw-data bindings, innovations, scale/count/AICc/GLS arithmetic; no global-optimum or adequacy certificate"})
    write_json(args.output/"summary.json", {"cells": summary, "structure_selected": False,
               "remaining": "Inspect residual/innovation plots, calibrated diagnostics, active-bound/startup sensitivities and any convergence warnings before the structure decision"})
    print(json.dumps({"audit": "passed", "cells": len(receipts), "max_nll_error": max(r["nll_abs_error"] for r in receipts)}))


if __name__ == "__main__":
    main()
