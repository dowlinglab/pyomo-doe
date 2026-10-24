"""Nuisance/noise-refitted profiles for a restricted n=1 working model."""
import copy

import numpy as np
from scipy.optimize import minimize

from covariance_fit import covariance_bounds, numerical_gradient, projected_gradient, projected_state

CUTOFF_95_NOMINAL = 3.841458820694124


def profile_point(records, config, reference, parameter, value, neighbor=None):
    if reference["kernel"] != "ou_nugget":
        raise ValueError("Profile implementation is scoped to the selected OU-plus-measurement kernel")
    settings = copy.deepcopy(config)
    fixed_index = {"tau": 0, "theta": 1}.get(parameter)
    if parameter == "K":
        settings["fixed_K"] = value
    elif parameter not in {"tau", "theta"}:
        raise ValueError("Unknown profile parameter")
    bounds = [tuple(np.log(settings["thermal_bounds"][k])) for k in ("tau", "theta")]+covariance_bounds("ou_nugget")
    free = [i for i in range(4) if i != fixed_index]
    free_bounds = [bounds[i] for i in free]
    baseline = reference["best"]
    seeds = [np.asarray(baseline["coordinates"])]
    opposite = [s for s in reference["starts"] if s.get("success") and
                (s["tau"] > s["theta"]) != (baseline["tau"] > baseline["theta"])]
    if opposite:
        seeds.append(np.asarray(min(opposite, key=lambda s: s["negative_log_likelihood"])["coordinates"]))
    if neighbor is not None:
        seeds.insert(0, np.asarray(neighbor["coordinates"]))
    alternative_noise = seeds[-1].copy()
    alternative_noise[-2:] = [np.log(10.), .5]
    seeds.append(alternative_noise)
    def full_point(x):
        point = np.empty(4)
        point[free] = x
        if fixed_index is not None:
            point[fixed_index] = np.log(value)
        return point
    objective = lambda x: projected_state(records, settings, reference["mode"], reference["hold"], "ou_nugget", full_point(x))[0]
    derivative = lambda x: numerical_gradient(objective, x, free_bounds, step=2e-6)
    attempts = []
    best = None
    unique = set()
    for seed in seeds:
        start = np.clip(seed[free], [b[0] for b in free_bounds], [b[1] for b in free_bounds])
        key = tuple(np.round(start, 10))
        if key in unique:
            continue
        unique.add(key)
        try:
            solution = minimize(objective, start, jac=derivative, method="L-BFGS-B", bounds=free_bounds,
                                options={"ftol": 1e-12, "gtol": 1e-5, "maxiter": 600, "maxls": 35})
            pg = float(np.max(np.abs(projected_gradient(solution.x, derivative(solution.x), free_bounds))))
            candidates = [(solution, "L-BFGS-B", pg)]
            if not solution.success or pg > .05:
                recovery = minimize(objective, solution.x, jac=derivative, method="SLSQP", bounds=free_bounds,
                                    options={"ftol": 1e-10, "maxiter": 600})
                pg2 = float(np.max(np.abs(projected_gradient(recovery.x, derivative(recovery.x), free_bounds))))
                candidates.append((recovery, "SLSQP refinement", pg2))
            for sol, solver, gradient in candidates:
                point = full_point(sol.x)
                nll, coefficients, sigma, _ = projected_state(records, settings, reference["mode"], reference["hold"], "ou_nugget", point)
                row = {"solver": solver, "success": bool(sol.success), "message": str(sol.message),
                       "negative_log_likelihood": nll, "coordinates": point.tolist(),
                       "coefficients": coefficients.tolist(), "sigma_total_ML_C": sigma,
                       "projected_gradient_max_abs": gradient, "nfev": int(sol.nfev)}
                attempts.append(row)
                if row["success"] and (best is None or nll < best["negative_log_likelihood"]):
                    best = row
        except Exception as exc:
            attempts.append({"success": False, "exception": repr(exc), "initial_free_coordinates": start.tolist()})
    return {"parameter": parameter, "value": float(value), "best": best, "attempts": attempts,
            "status": "failed" if best is None else ("requires_revisit" if best["projected_gradient_max_abs"] > .05 else "complete")}


def curve(records, config, reference, parameter, callback=None):
    lower, upper = config["thermal_bounds"][parameter]
    primary = reference["best"]
    mle = primary["coefficients"][0] if parameter == "K" else primary[parameter]
    central = mle*np.exp(np.linspace(-.2, .2, 17))
    seeds = [np.geomspace(lower, upper, 25), central, [mle]]
    if parameter != "K":
        opposite = [s for s in reference["starts"] if s.get("success") and
                    (s["tau"] > s["theta"]) != (primary["tau"] > primary["theta"])]
        if opposite:
            other = min(opposite, key=lambda s: s["negative_log_likelihood"])[parameter]
            seeds.append(other*np.exp(np.linspace(-.2, .2, 17)))
    points = np.unique(np.clip(np.concatenate(seeds), lower, upper))
    evaluated = {}
    baseline = primary["negative_log_likelihood"]
    def evaluate(value, role):
        key = float(value)
        if key in evaluated:
            return evaluated[key]
        good = [row for row in evaluated.values() if row["best"] is not None]
        neighbor = min(good, key=lambda r: abs(np.log(r["value"]/key)))["best"] if good else None
        row = profile_point(records, config, reference, parameter, key, neighbor)
        row["role"] = role
        row["twice_nll_difference"] = None if row["best"] is None else 2*(row["best"]["negative_log_likelihood"]-baseline)
        evaluated[key] = row
        if callback:
            callback(row, list(evaluated.values()))
        return row
    for value in sorted(points, key=lambda v: abs(np.log(v/mle))):
        evaluate(value, "declared grid")
    base = sorted(evaluated.values(), key=lambda r: r["value"])
    crossings = []
    for left, right in zip(base[:-1], base[1:]):
        if left["status"] != "complete" or right["status"] != "complete":
            continue
        a = left["twice_nll_difference"]-CUTOFF_95_NOMINAL
        b = right["twice_nll_difference"]-CUTOFF_95_NOMINAL
        if a*b >= 0:
            continue
        lo, hi = left["value"], right["value"]
        for _ in range(8):
            midpoint = float(np.sqrt(lo*hi))
            row = evaluate(midpoint, "crossing refinement")
            if row["status"] != "complete":
                break
            c = row["twice_nll_difference"]-CUTOFF_95_NOMINAL
            if a*c <= 0:
                hi = midpoint
            else:
                lo, a = midpoint, c
        crossings.append({"log_bisection_bracket": [float(lo), float(hi)],
                          "initial_bracket": [left["value"], right["value"]]})
    rows = sorted(evaluated.values(), key=lambda r: r["value"])
    lower_minimum = any(r["twice_nll_difference"] is not None and r["twice_nll_difference"] < -2e-5 for r in rows)
    # Keep components explicit; missing points break a component rather than
    # manufacturing an excluded or included interval through a failed region.
    components = []
    current = []
    for row in rows:
        included = row["status"] == "complete" and row["twice_nll_difference"] <= CUTOFF_95_NOMINAL
        if included:
            current.append(row["value"])
        elif current:
            components.append([min(current), max(current)]); current = []
    if current:
        components.append([min(current), max(current)])
    return {"parameter": parameter, "mode": reference["mode"], "kernel": reference["kernel"],
            "baseline_nll": baseline, "mle_value": mle, "nominal_twice_nll_cutoff": CUTOFF_95_NOMINAL,
            "declared_bounds": [lower, upper], "points": rows, "refined_crossings": crossings,
            "included_grid_components": components, "lower_minimum_found": lower_minimum,
            "numerical_status": "requires_revisit" if lower_minimum or any(r["status"] != "complete" for r in rows) else "complete",
            "touches_lower_bound": bool(rows[0]["status"] == "complete" and rows[0]["twice_nll_difference"] <= CUTOFF_95_NOMINAL),
            "touches_upper_bound": bool(rows[-1]["status"] == "complete" and rows[-1]["twice_nll_difference"] <= CUTOFF_95_NOMINAL),
            "notice": "Finite-grid nuisance/noise-refitted likelihood support; cutoff is nominal, with active bounds, pole modes and mean/error misspecification. Crossings are brackets, unresolved points stay unresolved, and finite multistart scanning is not a global certificate."}
