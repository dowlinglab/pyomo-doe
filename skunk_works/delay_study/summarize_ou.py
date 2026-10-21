"""Audit saved OU fits using an independent scalar innovations calculation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_fits import save_json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial_summary(values):
    values = np.asarray(values)
    centered = values-values.mean()
    variance_sum = centered@centered
    acf = [float(centered[j:]@centered[:-j]/variance_sum) for j in range(1, 31)] if variance_sum > 0 else [None]*30
    return {"mean": float(values.mean()), "rms": float(np.sqrt(values@values/len(values))),
            "acf_lags_1_to_30": acf,
            "notice": "Descriptive centered sample autocorrelation; no calibrated fitted-model whiteness test is asserted."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ou", type=Path, default=HERE/"results/ou")
    parser.add_argument("--fits", type=Path, default=HERE/"results/fits")
    args = parser.parse_args()
    manifest = json.loads((args.ou/"manifest.json").read_text())
    if manifest["status"] != "complete" or len(manifest["completed"]) != manifest["expected_jobs"]:
        raise RuntimeError("OU phase not complete")
    rows, diagnostics, inputs = [], {}, {}
    maxima = {"nll_absolute_error": 0., "innovation_absolute_error_C": 0., "variance_absolute_error_C2": 0.,
              "selected_nll_minus_best_converged_abs": 0., "physical_transform_absolute_error": 0.}
    for key in sorted(manifest["completed"]):
        path = args.ou/(key+".json")
        record = json.loads(path.read_text())
        fit, u, spec = record["fit"], record["uncertainty"], record["spec"]
        fit_path = args.fits/record["fit_file"]
        prediction_path = args.ou/record["prediction_file"]
        if sha(fit_path) != record["fit_sha256"] or sha(prediction_path) != record["prediction_sha256"]:
            raise RuntimeError("OU fit/prediction input changed: "+key)
        inputs[str(path)] = sha(path)
        inputs[str(fit_path)] = sha(fit_path)
        inputs[str(prediction_path)] = sha(prediction_path)
        baseline = json.loads(fit_path.read_text())
        frame = pd.read_csv(prediction_path)
        calibration = frame[frame.role == "calibration"]
        squared, logdet, n = 0., 0., 0
        diagnostics[key] = {}
        for name in record["datasets"]:
            group = calibration[calibration.experiment == name]
            residual = (group.predicted_C-group.measured_C).to_numpy()[1:]
            time = group.time_s.to_numpy()[1:]
            rho = np.exp(-np.diff(time)/fit["correlation_length_s"])
            denominator = np.sqrt(1-rho*rho)
            innovation = np.r_[residual[0], (residual[1:]-rho*residual[:-1])/denominator]
            squared += float(innovation@innovation)
            logdet += float(np.log(1-rho*rho).sum())
            n += len(residual)
            stored = group.innovation_C.to_numpy()[1:]
            maxima["innovation_absolute_error_C"] = max(maxima["innovation_absolute_error_C"], float(np.max(abs(stored-innovation))))
            diagnostics[key][name] = {"raw_C": serial_summary(residual), "innovation_C": serial_summary(innovation),
                                      "standardized_innovation": serial_summary(innovation/fit["sigma_stationary_ML_C"])}
        variance = squared/n
        nll = .5*(n*(np.log(2*np.pi)+1+np.log(variance))+logdet)
        if n != fit["n_conditional"]:
            raise RuntimeError("Conditional sample count changed")
        maxima["nll_absolute_error"] = max(maxima["nll_absolute_error"], abs(nll-fit["negative_log_likelihood"]))
        maxima["variance_absolute_error_C2"] = max(maxima["variance_absolute_error_C2"], abs(variance-fit["sigma_stationary_ML_C"]**2))
        converged = [attempt["negative_log_likelihood"] for attempt in fit["attempts"] if attempt["success"]]
        if not converged:
            raise RuntimeError("No converged OU attempt: "+key)
        maxima["selected_nll_minus_best_converged_abs"] = max(maxima["selected_nll_minus_best_converged_abs"],
                                                               abs(min(converged)-fit["negative_log_likelihood"]))
        p = fit["parameters"]
        physical = np.array([.032/p["K"], .032*p["tau"]/p["K"], p["theta"]])
        stored_physical = np.array([fit["physics_parameters"][name] for name in ("Ua", "Cp", "theta")])
        maxima["physical_transform_absolute_error"] = max(maxima["physical_transform_absolute_error"], float(np.max(abs(physical-stored_physical))))
        C = u["observed_cov_log_mean_and_length"]
        full_sd = np.sqrt(np.maximum(np.diag(C), 0)) if C is not None else None
        mean_rank = u["mean_rank_relative_1e-8"]
        row = {"case": key, "calibration": record["calibration"], "family": spec["family"], "order": spec["order"],
               "initialization": spec["initialization"], **p, "Ua": physical[0], "Cp": physical[1],
               "correlation_length_s": fit["correlation_length_s"], "sigma_stationary_ML_C": fit["sigma_stationary_ML_C"],
               "nll": nll, "aicc_nominal": fit["aicc_nominal"], "rmse_raw_C": fit["rmse_raw_C"],
               "baseline_rmse_C": baseline["fit"]["rmse"],
               "twice_nll_improvement_over_iid": fit["twice_nll_improvement_over_iid"],
               "mean_rank": mean_rank, "full_observed_covariance_defined": C is not None,
               "theta_std_observed_native": p["theta"]*full_sd[2] if full_sd is not None and mean_rank == 3 else None,
               "length_std_observed_native": fit["correlation_length_s"]*full_sd[3] if full_sd is not None else None,
               "hessian_relative_step_disagreement": u["hessian_relative_step_disagreement"],
               "active_bound_count": sum(value != 0 for value in fit["active_bounds_log"]),
               "stationarity_warning": fit["stationarity_warning"], "requires_revisit": fit["requires_revisit"],
               "validation_mean_rmse_C": np.mean([item["rmse_C"] for item in record["validation"]]),
               "baseline_validation_mean_rmse_C": np.mean([item["rmse_C"] for item in baseline["validation"]]),
               "maximum_abs_innovation_acf_lag1": max(abs(value["innovation_C"]["acf_lags_1_to_30"][0]) for value in diagnostics[key].values())}
        rows.append(row)
    limits = {"nll_absolute_error": 1e-7, "innovation_absolute_error_C": 1e-9, "variance_absolute_error_C2": 1e-9,
              "selected_nll_minus_best_converged_abs": 1e-9, "physical_transform_absolute_error": 1e-12}
    passed = all(maxima[name] <= bound for name, bound in limits.items())
    pd.DataFrame(rows).to_csv(args.ou/"summary.csv", index=False)
    save_json(args.ou/"residual_diagnostics.json", diagnostics)
    save_json(args.ou/"summary_audit.json", {"status": "passed" if passed else "failed", "jobs": len(rows),
                                             "input_sha256": inputs, "maximum_errors": maxima, "tolerances": limits,
                                             "producer_sha256": sha(HERE/"summarize_ou.py"),
                                             "notice": "Independent scalar OU innovations validate saved likelihood arithmetic and serialization, not model adequacy or coverage. Large remaining innovation correlations are evidence against the single-component working covariance."})
    print(json.dumps({"status": "passed" if passed else "failed", "jobs": len(rows), "maximum_errors": maxima}))
    return not passed


if __name__ == "__main__":
    raise SystemExit(main())
