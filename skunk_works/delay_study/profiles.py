"""Likelihood profiles with nuisance reoptimization and disconnected intervals."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc, chi2

from inference import ModelSpec, parameter_box, residuals
from models import to_io, to_physics


def profile_grid(record, parameter_index, target_points=61):
    result, uncertainty = record["fit"], record["uncertainty"]
    spec = ModelSpec(**result["spec"])
    parameters = np.array([result["parameters"][n] for n in spec.names])
    lower, upper = parameter_box(spec)
    lo, hi, center = np.log(lower), np.log(upper), np.log(parameters)
    j = parameter_index
    global_grid = np.linspace(lo[j], hi[j], max(20, target_points//3))
    std_native = uncertainty["std_native_iid"][j]
    width = 6*std_native/parameters[j] if std_native is not None else .5*(hi[j]-lo[j])
    width = max(width, 1e-5)
    local_grid = np.linspace(max(lo[j], center[j]-width), min(hi[j], center[j]+width), max(41, target_points*2//3))
    grids = [global_grid, local_grid, [center[j]]]
    if spec.family == "chain" and spec.order == 1 and spec.initialization == "cold":
        canonical = parameters if spec.coordinates == "io" else to_io(parameters)
        swapped = canonical[[0, 2, 1]]
        if spec.coordinates == "physics":
            swapped = to_physics(swapped)
        second_center = np.log(swapped[j])
        if lo[j] <= second_center <= hi[j] and abs(second_center-center[j]) > width:
            grids.append(np.linspace(max(lo[j], second_center-width), min(hi[j], second_center+width), 41))
            grids.append([second_center])
    return np.unique(np.concatenate(grids))


def interval_components(log_values, statistic, threshold, lower_bound, upper_bound):
    inside = np.asarray(statistic) <= threshold
    groups = np.split(np.flatnonzero(inside), np.flatnonzero(np.diff(np.flatnonzero(inside)) > 1)+1)
    components = []
    for indices in groups:
        if not len(indices):
            continue
        left, right = indices[0], indices[-1]
        if left > 0:
            qleft = np.interp(threshold, [statistic[left], statistic[left-1]], [log_values[left], log_values[left-1]])
        else:
            qleft = log_values[left]
        if right < len(log_values)-1:
            qright = np.interp(threshold, [statistic[right], statistic[right+1]], [log_values[right], log_values[right+1]])
        else:
            qright = log_values[right]
        components.append({"lower": float(np.exp(qleft)), "upper": float(np.exp(qright)),
                           "lower_status": "open_at_exploration_bound" if left == 0 else "interpolated_grid_crossing",
                           "upper_status": "open_at_exploration_bound" if right == len(log_values)-1 else "interpolated_grid_crossing"})
    return components


def profile(record, datasets, parameter_index, grid_points=61, n_starts=4, seed=532, max_nfev=600,
            recover_original=False):
    result = record["fit"]
    spec = ModelSpec(**result["spec"])
    p = np.array([result["parameters"][name] for name in spec.names])
    center = np.log(p)
    lower, upper = parameter_box(spec)
    lo, hi = np.log(lower), np.log(upper)
    grid = profile_grid(record, parameter_index, grid_points)
    j = parameter_index
    nuisance = np.arange(len(p)) != j
    solved, rows = {}, {}
    # Solve the center first and then move outward, retaining both sides.
    order = np.argsort(np.abs(grid-center[j]))
    for index in order:
        value = grid[index]
        nearest = min(solved, key=lambda other: abs(grid[other]-value)) if solved else None
        warm = solved[nearest] if nearest is not None else center
        starts = [center[nuisance], warm[nuisance]][:n_starts]
        if n_starts > len(starts):
            sampler = qmc.LatinHypercube(d=int(nuisance.sum()), seed=seed+int(index))
            starts.extend(qmc.scale(sampler.random(n_starts-len(starts)), lo[nuisance], hi[nuisance]))
        attempts = []
        best = None

        def assemble(other):
            q = center.astype(np.result_type(center, other), copy=True)
            q[j] = value
            q[nuisance] = other
            return q

        def objective(other):
            return residuals(assemble(other), datasets, spec)

        for start in starts:
            fitted = least_squares(objective, np.clip(start, lo[nuisance]+1e-10, hi[nuisance]-1e-10),
                                   bounds=(lo[nuisance], hi[nuisance]), jac="3-point",
                                   ftol=1e-10, xtol=1e-10, gtol=1e-7, max_nfev=max_nfev)
            sse = float(fitted.fun@fitted.fun)
            attempt = {"success": bool(fitted.success), "sse": sse, "nfev": int(fitted.nfev),
                       "status": int(fitted.status), "optimality": float(fitted.optimality),
                       "parameters": np.exp(assemble(fitted.x)).tolist(),
                       "phase": "initial", "jacobian": "3-point", "max_nfev": max_nfev}
            attempts.append(attempt)
            if fitted.success and (best is None or sse < best["sse"]):
                best = attempt
        if best is None and recover_original and spec.family == "original":
            # Keep failed initial attempts. The original parameterization has an
            # exact redundant direction under these initial conditions; avoid
            # differencing cancellation and scale by the actual Jacobian.
            recovery_starts = [np.log(a["parameters"])[nuisance] for a in sorted(attempts, key=lambda a: a["sse"])]
            for start in recovery_starts:
                fitted = least_squares(objective, np.clip(start, lo[nuisance]+1e-10, hi[nuisance]-1e-10),
                                       bounds=(lo[nuisance], hi[nuisance]), jac="cs", x_scale="jac",
                                       ftol=1e-10, xtol=1e-10, gtol=1e-7, max_nfev=10*max_nfev)
                sse = float(fitted.fun@fitted.fun)
                attempt = {"success": bool(fitted.success), "sse": sse, "nfev": int(fitted.nfev),
                           "status": int(fitted.status), "optimality": float(fitted.optimality),
                           "parameters": np.exp(assemble(fitted.x)).tolist(),
                           "phase": "recovery", "jacobian": "complex-step", "x_scale": "jac",
                           "max_nfev": 10*max_nfev}
                attempts.append(attempt)
                if fitted.success and (best is None or sse < best["sse"]):
                    best = attempt
        if best is None:
            raise RuntimeError("Profile nuisance fits failed at "+str(np.exp(value)))
        solved[index] = np.log(best["parameters"])
        rows[index] = {"parameter_value": float(np.exp(value)), "log_value": float(value),
                       "sse": best["sse"], "parameters": best["parameters"], "attempts": attempts}
    points = [rows[index] for index in range(len(grid))]
    baseline = result["sse"]
    statistics_unknown = [result["n"]*np.log(point["sse"]/baseline) for point in points]
    known_sigma = record["uncertainty"]["known_sigma_C"]
    statistics_known = [(point["sse"]-baseline)/known_sigma**2 for point in points]
    threshold = float(chi2.ppf(.95, 1))
    for point, unknown, known in zip(points, statistics_unknown, statistics_known):
        point["lr_unknown_sigma_iid"] = float(unknown)
        point["lr_known_sigma_iid"] = float(known)
    return {"spec": asdict(spec), "profiled_parameter": spec.names[j], "parameter_index": j,
            "baseline_sse": baseline, "n_conditional": result["n"], "point_count": len(points),
            "starts_per_point": n_starts, "seed": seed, "points": points, "chi2_95_df1": threshold,
            "original_recovery_enabled": recover_original,
            "intervals_unknown_sigma_iid": interval_components(grid, statistics_unknown, threshold, lo[j], hi[j]),
            "intervals_known_sigma_iid": interval_components(grid, statistics_known, threshold, lo[j], hi[j]),
            "minimum_profile_minus_baseline_sse": float(min(point["sse"] for point in points)-baseline),
            "baseline_requires_revisit": bool(min(point["sse"] for point in points) < baseline*(1-1e-6)),
            "notice": "Pointwise asymptotic likelihood-ratio thresholds conditional on IID Gaussian noise and fixed initialization. Grid crossing interpolation; open boundaries do not prove infinite intervals. Serial residuals and rank loss invalidate routine nominal coverage."}
