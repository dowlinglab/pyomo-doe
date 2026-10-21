"""Multistart mean/noise estimation and local likelihood curvature for OU errors."""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from inference import ModelSpec, parameter_box, predict, sensitivities
from noise import objective, profile_ou, whiten_ou


def projected_gradient(point, gradient, lower, upper):
    result = np.array(gradient, copy=True)
    result[(point <= lower+1e-6) & (result > 0)] = 0
    result[(point >= upper-1e-6) & (result < 0)] = 0
    return result


def fit_ou(datasets, spec, starting, n_starts=24, seed=532, maxiter=600):
    if spec.coordinates != "io" or spec.family == "original":
        raise ValueError("Use identified IO mean parameters; transform physical coordinates explicitly")
    lower, upper = parameter_box(spec)
    lo, hi = np.log(np.r_[lower, .05]), np.log(np.r_[upper, 3000.])
    lengths = [.05, .2, 1., 3., 10., 30., 100., 500.]
    starts = [np.log(np.r_[starting, length]) for length in lengths[:n_starts]]
    if n_starts > len(starts):
        starts.extend(qmc.scale(qmc.LatinHypercube(d=4, seed=seed).random(n_starts-len(starts)), lo, hi))
    attempts, best = [], None
    begun = time.monotonic()
    for index, start in enumerate(starts):
        first = np.clip(start, lo+1e-10, hi-1e-10)
        solution = minimize(objective, first, args=(datasets, spec), jac=True,
                            method="L-BFGS-B", bounds=list(zip(lo, hi)),
                            options={"ftol": 1e-12, "gtol": 1e-5, "maxiter": maxiter,
                                     "maxfun": 6*maxiter, "maxls": 40})
        gradient = projected_gradient(solution.x, solution.jac, lo, hi)
        row = {"start": index, "initial_parameters_and_length": np.exp(first).tolist(),
               "parameters_and_length": np.exp(solution.x).tolist(),
               "negative_log_likelihood": float(solution.fun), "success": bool(solution.success),
               "solver": "L-BFGS-B", "solver_status": int(solution.status), "message": str(solution.message),
               "iterations": int(solution.nit), "function_evaluations": int(solution.nfev),
               "projected_gradient_max_abs": float(np.max(abs(gradient)))}
        attempts.append(row)
        if solution.success and np.isfinite(solution.fun) and (best is None or solution.fun < best["negative_log_likelihood"]):
            best = row
        if not solution.success:
            fallback = minimize(objective, solution.x, args=(datasets, spec), jac=True,
                                method="SLSQP", bounds=list(zip(lo, hi)),
                                options={"ftol": 1e-10, "maxiter": 3*maxiter})
            gradient = projected_gradient(fallback.x, fallback.jac, lo, hi)
            row = {"start": index, "initial_parameters_and_length": np.exp(solution.x).tolist(),
                   "parameters_and_length": np.exp(fallback.x).tolist(),
                   "negative_log_likelihood": float(fallback.fun), "success": bool(fallback.success),
                   "solver": "SLSQP recovery", "solver_status": int(fallback.status), "message": str(fallback.message),
                   "iterations": int(fallback.nit), "function_evaluations": int(fallback.nfev),
                   "projected_gradient_max_abs": float(np.max(abs(gradient)))}
            attempts.append(row)
            if fallback.success and np.isfinite(fallback.fun) and (best is None or fallback.fun < best["negative_log_likelihood"]):
                best = row
    if best is None:
        raise RuntimeError("No converged OU mean/noise solution")
    parameters = np.asarray(best["parameters_and_length"][:3])
    length = best["parameters_and_length"][-1]
    # Use a common-feasible pole-label convention, matching the baseline fit.
    if spec.family == "chain" and spec.order == 1 and spec.initialization == "cold" and parameters[1] < parameters[2]:
        candidate = parameters[[0, 2, 1]]
        if np.all(candidate >= lower) and np.all(candidate <= upper):
            parameters = candidate
    residuals = [(predict(data, parameters, spec)-data.temperature)[1:] for data in datasets]
    noise = profile_ou(residuals, [data.time[1:] for data in datasets], length)
    values = np.r_[parameters, length]
    active = np.where(np.log(values) <= lo+1e-5, -1, np.where(np.log(values) >= hi-1e-5, 1, 0))
    iid_candidate = objective(np.log(np.r_[starting, .05]), datasets, spec)[0]
    return {"parameters": dict(zip(spec.names, parameters.tolist())), "correlation_length_s": length,
            "sigma_stationary_ML_C": noise["sigma_ML_C"], "negative_log_likelihood": noise["negative_log_likelihood"],
            "n_conditional": noise["n_conditional"], "raw_sse_C2": float(sum(r@r for r in residuals)),
            "weighted_sse_C2": noise["weighted_sse_C2"], "logdet_correlation": noise["logdet_correlation"],
            "active_bounds_log": active.tolist(), "noise_length_box_s": [.05, 3000.],
            "IID_lower_length_candidate_nll": float(iid_candidate),
            "requires_revisit": bool(noise["negative_log_likelihood"] > iid_candidate+1e-5),
            "selected_projected_gradient_max_abs": best["projected_gradient_max_abs"],
            "stationarity_warning": bool(best["projected_gradient_max_abs"] > .05),
            "attempts": attempts, "seed": seed, "elapsed_seconds": time.monotonic()-begun,
            "notice": "Working stationary Gaussian OU residual model, independent across experiments. First temperatures, ambient and heater calibration are fixed. Better likelihood does not establish thermal model adequacy. All solver outcomes retained; multistart supplies no global certificate."}


def observed_hessian(point, datasets, spec, step):
    columns = []
    for j in range(len(point)):
        delta = np.zeros(len(point))
        delta[j] = step
        columns.append((objective(point+delta, datasets, spec)[1]-objective(point-delta, datasets, spec)[1])/(2*step))
    raw = np.column_stack(columns)
    return (raw+raw.T)/2, float(np.linalg.norm(raw-raw.T)/max(np.linalg.norm(raw), 1e-30))


def local_uncertainty(result, datasets, spec):
    parameters = np.array([result["parameters"][name] for name in spec.names])
    length = result["correlation_length_s"]
    sigma = result["sigma_stationary_ML_C"]
    whitened = [whiten_ou(sensitivities(parameters, [data], spec), data.time[1:], length)[0] for data in datasets]
    J = np.vstack(whitened)/sigma
    _, singular, Vh = np.linalg.svd(J, full_matrices=False)
    keep = singular > singular[0]*1e-8
    rank = int(keep.sum())
    basis = Vh[keep]
    inverse = (basis.T/(singular[keep]**2))@basis
    nonestimable = np.sum(Vh[~keep]**2, axis=0) > 1e-10
    M = np.array([[-1., 0., 0.], [-1., 1., 0.], [0., 0., 1.]])
    point = np.log(np.r_[parameters, length])
    hessian, asymmetry = observed_hessian(point, datasets, spec, 1e-4)
    finer, fine_asymmetry = observed_hessian(point, datasets, spec, 5e-5)
    values, vectors = np.linalg.eigh(finer)
    covariance = None
    if values[0] > max(abs(values))*1e-8:
        covariance = (vectors/values)@vectors.T
    N = np.eye(4)
    N[:3, :3] = M
    return {"parameters_log_order": spec.names+["correlation_length_s"],
            "mean_rank_relative_1e-8": rank, "mean_singular_values_noise_scaled": singular.tolist(),
            "mean_cov_log_conditional_noise": inverse.tolist() if rank == 3 else None,
            "physics_mean_cov_log_conditional_noise": (M@inverse@M.T).tolist() if rank == 3 else None,
            "mean_nonestimable_native_parameters": [name for name, flag in zip(spec.names, nonestimable) if flag],
            "mean_std_native_conditional_noise": [None if flag else float(sd) for flag, sd in zip(nonestimable, parameters*np.sqrt(np.maximum(np.diag(inverse), 0)))],
            "observed_profile_variance_hessian_log": finer.tolist(), "observed_hessian_eigenvalues": values.tolist(),
            "observed_hessian_eigenvectors_columns": vectors.tolist(),
            "observed_cov_log_mean_and_length": covariance.tolist() if covariance is not None else None,
            "physics_observed_cov_log_mean_and_length": (N@covariance@N.T).tolist() if covariance is not None else None,
            "hessian_step_sizes_log": [1e-4, 5e-5],
            "hessian_relative_step_disagreement": float(np.linalg.norm(hessian-finer)/max(np.linalg.norm(finer), 1e-30)),
            "hessian_raw_relative_asymmetry": [asymmetry, fine_asymmetry],
            "notice": "Conditional mean information holds fitted OU scale/length fixed. Observed curvature profiles variance and includes mean/length coupling; it is local and can be nonpositive or step sensitive. Active bounds, pole-label modes and mean misspecification invalidate routine symmetric confidence interpretations."}
