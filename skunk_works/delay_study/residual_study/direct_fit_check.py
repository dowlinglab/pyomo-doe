"""Check selected variable projection against direct bounded mean/noise fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from scipy.optimize import minimize

from covariance import Whitening, profiled_likelihood
from covariance_fit import numerical_gradient, projected_gradient
from data import load
from mean import fitted_records
from run_mean import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fit_path = args.reference/"fit.json"
    fit = json.loads(fit_path.read_text())
    config = json.loads((args.reference/"effective_config.json").read_text())
    if fit["mode"] != "constant" or fit["kernel"] != "ou_nugget":
        raise ValueError("This scoped check targets the leading constant/OU-plus-measurement description")
    records = [load(n) for n in fit["record_names"]]
    best = fit["best"]
    bounds = [tuple(np.log(config["thermal_bounds"][k])) for k in ("K", "tau", "theta")]
    bounds += [tuple(config["ambient_offset_bounds_C"])]*len(records)
    bounds += [(np.log(.05), np.log(3000.)), (0., 1.)]
    point = np.r_[np.log([best["coefficients"][0], best["tau"], best["theta"]]),
                  best["coefficients"][1:], np.log(best["noise_parameters"][0]), best["noise_parameters"][1]]
    def objective(x):
        candidate = dict(fit, best={"tau": np.exp(x[1]), "theta": np.exp(x[2]),
                                    "coefficients": np.r_[np.exp(x[0]), x[3:-2]]})
        predicted = fitted_records(records, candidate)
        residual = np.concatenate([(d.temperature-p)[1:] for d, p in zip(records, predicted)])
        W = Whitening([d.time[1:] for d in records], "ou_nugget", [np.exp(x[-2]), x[-1]])
        return profiled_likelihood(residual, W)[0]
    initial_error = objective(point)-best["negative_log_likelihood"]
    if abs(initial_error) > 1e-7:
        raise AssertionError("Direct objective differs at reference point")
    derivative = lambda x: numerical_gradient(objective, x, bounds, step=2e-6)
    perturbation = np.r_[.005, -.005, .005, [.01*(-1.)**j for j in range(len(records))], .02, -.0005]
    starts = [point, np.clip(point+perturbation, [b[0] for b in bounds], [b[1] for b in bounds])]
    attempts = []
    for index, first in enumerate(starts):
        solution = minimize(objective, first, jac=derivative, method="SLSQP", bounds=bounds,
                            options={"maxiter": 600, "ftol": 1e-10})
        gradient = derivative(solution.x)
        pg = float(np.max(np.abs(projected_gradient(solution.x, gradient, bounds))))
        attempts.append({"start": index, "initial_coordinates": first.tolist(),
                         "coordinates": solution.x.tolist(), "success": bool(solution.success),
                         "message": str(solution.message), "nfev": int(solution.nfev),
                         "negative_log_likelihood": float(solution.fun),
                         "projected_gradient_max_abs": pg})
        print("direct", index, solution.fun, solution.success, pg, flush=True)
    converged = [a for a in attempts if a["success"] and a["projected_gradient_max_abs"] <= .05]
    selected = min(converged, key=lambda a: a["negative_log_likelihood"]) if converged else None
    difference = None if selected is None else selected["negative_log_likelihood"]-best["negative_log_likelihood"]
    passed = selected is not None and abs(difference) <= 1e-5
    write_json(args.output/"audit.json", {"status": "passed" if passed else "requires_revisit",
               "reference_fit_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
               "initial_objective_difference": float(initial_error), "selected_nll_difference": difference,
               "coordinate_order": ["logK", "logtau", "logtheta"]+["ambient_C_"+str(j) for j in range(len(records))]+["logell", "nugget_fraction"],
               "attempts": attempts,
               "scope": "Two direct 17-coordinate optimizations with variance profiled; ambient/gain not variable-projected. Shared independently audited forward/covariance functions. Not global optimality or scientific adequacy."})
    if not passed:
        raise RuntimeError("Direct-fit cross-check needs investigation")


if __name__ == "__main__":
    main()
