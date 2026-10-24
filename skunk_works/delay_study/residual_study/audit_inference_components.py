"""Independent fixed-gain projection and numerical curvature preflight."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from scipy.linalg import cholesky, solve_triangular

from audit_mean import reference
from covariance import Whitening, dense_covariance
from covariance_fit import projected_state
from data import load
from mean import fitted_records, project
from run_mean import write_json
from teaching_inference import local_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ordinary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fit_path = args.reference/"fit.json"
    result = json.loads(fit_path.read_text())
    config = json.loads((args.reference/"effective_config.json").read_text())
    records = [load(n) for n in result["record_names"]]
    best = result["best"]
    subset = [records[0], records[-1]]
    checks = []
    for kernel, parameters in [("iid", []), ("ou_nugget", best["noise_parameters"])]:
        for fixed in (.65, .8):
            settings = copy.deepcopy(config); settings["fixed_K"] = fixed
            W = Whitening([d.time[1:] for d in subset], kernel, parameters)
            coefficients, residual = project(subset, best["tau"], best["theta"], "constant", "left", settings, W)
            expected = [fixed]
            expected_residuals = []
            for d in subset:
                mean = reference(d, fixed, best["tau"], best["theta"], [0., 0.], "left")
                a = -np.expm1(-(d.time[1:]-d.time[0])/best["tau"])
                y = d.temperature[1:]-mean[1:]
                L = cholesky(dense_covariance(d.time[1:], kernel, parameters), lower=True)
                a, y = solve_triangular(L, a, lower=True), solve_triangular(L, y, lower=True)
                delta = float(np.clip(a@y/(a@a), -2., 2.))
                expected.append(delta)
                expected_residuals.append(a*delta-y)
            beta_error = float(np.max(np.abs(coefficients-expected)))
            residual_error = float(np.max(np.abs(residual-np.concatenate(expected_residuals))))
            if beta_error > 1e-7 or residual_error > 1e-7:
                raise AssertionError(("Fixed-gain projection", kernel, fixed, beta_error, residual_error))
            checks.append({"kernel": kernel, "K_fixed": fixed, "coefficient_max_error": beta_error,
                           "whitened_residual_max_error": residual_error})
    original = np.load(args.reference/"trajectories.npz")
    default_error = max(float(np.max(np.abs(p-original["predicted_"+str(j)]))) for j, p in enumerate(fitted_records(records, result)))
    nll = projected_state(records, config, "constant", "left", "ou_nugget", np.asarray(best["coordinates"]))[0]
    if default_error > 1e-10 or abs(nll-best["negative_log_likelihood"]) > 1e-7:
        raise AssertionError("Default behavior changed when fixed-gain profiling was added")
    ordinary = json.loads(args.ordinary.read_text())
    information = local_analysis(records, result, ordinary)
    write_json(args.output/"information_primary.json", information)
    write_json(args.output/"audit.json", {"status": "passed", "fixed_gain_checks": checks,
               "default_prediction_max_error_C": default_error, "default_nll_error": nll-best["negative_log_likelihood"],
               "noise_scale_hessian_identity_error": information["observed_curvature"]["common_scale_relative_identity_error"],
               "reference_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
               "scope": "Analytical bounded ambient estimates with independently propagated means and dense covariance; default reproduction and observed noise-scale Hessian identity. Not confidence coverage or global profile verification."})


if __name__ == "__main__":
    main()
