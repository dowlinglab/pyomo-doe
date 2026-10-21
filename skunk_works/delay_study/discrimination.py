"""Finite-horizon mean separation, distinct from parameter-information design.

Fixed means share an assumed known Gaussian covariance. Refit distances allow
the alternative mean parameters to vary inside their declared boxes; neither
quantity establishes real-data testing power with unknown model discrepancy.
"""
import hashlib

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.stats import norm, qmc

from data import Experiment
from design import energy_weights
from inference import parameter_box, predict, sensitivities
from noise import whiten_ou


def separation_map(first, second, time):
    if first.sigma != second.sigma or first.noise_length_s != second.noise_length_s:
        raise ValueError("Mean separation requires a common assumed covariance")
    delta = (first.response_map-second.response_map)[1:]
    if first.noise_length_s is not None:
        delta, _ = whiten_ou(delta, np.asarray(time)[1:], first.noise_length_s)
    return delta/first.sigma


def squared_distance_and_gradient(matrix, amplitudes):
    delta = matrix@amplitudes
    return float(delta@delta), 2*matrix.T@delta


def fixed_gaussian_error(distance_squared):
    """Equal-prior minimum classification error for two fixed Gaussian means.

Projecting onto R^-1 times the mean difference gives a univariate normal
log-likelihood ratio with mean +/-D^2/2 and variance D^2. Threshold zero
therefore gives Phi(-D/2). This ideal simple-hypothesis result does not apply
after estimating alternative parameters from the same future observations.
"""
    if distance_squared < 0:
        raise ValueError("A squared distance cannot be negative")
    return float(norm.sf(np.sqrt(distance_squared)/2))


def optimize_separation(maps, knots, starts, safety, reference, aggregation="min",
                        mean_input=50., maximum_temperature=85., maximum_slew=1., maxiter=400):
    reference_values = np.array([squared_distance_and_gradient(matrix, reference)[0] for matrix in maps])
    if np.any(reference_values <= 0):
        raise ValueError("Reference must separate each fixed pair")
    weights = energy_weights(knots)
    difference = np.diff(np.eye(len(knots)), axis=0)
    constraints_map = np.vstack([*[s.response_map for s in safety], difference, -difference])
    limits = np.r_[np.concatenate([np.full(len(s.response_map), maximum_temperature-s.ambient) for s in safety]),
                   maximum_slew*np.diff(knots), maximum_slew*np.diff(knots)]

    def objective(amplitudes):
        rows = [squared_distance_and_gradient(matrix, amplitudes) for matrix in maps]
        values = np.array([row[0] for row in rows])
        if np.any(values <= 0):
            return np.inf, np.zeros(len(knots))
        log_values = np.log(values/reference_values)
        gradients = np.array([row[1] for row in rows])/values[:, None]
        if aggregation == "mean":
            return -float(log_values.mean()), -gradients.mean(axis=0)
        if aggregation == "min":
            j = int(np.argmin(log_values))
            return -float(log_values[j]), -gradients[j]
        raise ValueError("Unknown aggregation")

    constraints = [{"type": "eq", "fun": lambda v: weights@v-mean_input, "jac": lambda v: weights},
                   {"type": "ineq", "fun": lambda v: limits-constraints_map@v, "jac": lambda v: -constraints_map}]
    attempts = []
    for name, start in starts.items():
        if not np.isfinite(objective(start)[0]):
            attempts.append({"start": name, "success": False, "status": "zero_separation_start"})
            continue
        result = minimize(objective, start, jac=True, method="SLSQP", bounds=[(0., 100.)]*len(knots),
                          constraints=constraints, options={"ftol": 1e-9, "maxiter": maxiter})
        v = result.x
        violation = max(0., float(np.max(constraints_map@v-limits)), abs(float(weights@v-mean_input)),
                        float(np.max(-v)), float(np.max(v-100)))
        attempts.append({"start": name, "success": bool(result.success), "status": int(result.status),
                         "message": str(result.message), "iterations": int(result.nit),
                         "objective": float(result.fun) if np.isfinite(result.fun) else None,
                         "maximum_constraint_violation": violation, "amplitudes_percent": v.tolist()})
    accepted = [row for row in attempts if row["success"] and row.get("objective") is not None
                and row["maximum_constraint_violation"] < 1e-6]
    return {"best": min(accepted, key=lambda row: row["objective"]) if accepted else None,
            "attempts": attempts, "aggregation": aggregation, "reference_distance_squared": reference_values.tolist(),
            "notice": "Local multistart optimization of scenario-relative log fixed-mean squared separation. Min aggregation is nonsmooth at pair ties. No global certificate, nuisance robustness or real-data testing-power guarantee."}


def refitted_separation(true_scenario, alternative, amplitudes, time, basis, starts=24, seed=532, max_nfev=600):
    """Closest found alternative mean for one fixed generating scenario/input."""
    if true_scenario.noise_length_s != alternative.noise_length_s or true_scenario.sigma != alternative.sigma:
        raise ValueError("Refit diagnostic requires a common working covariance")
    temperature = true_scenario.ambient+true_scenario.response_map@amplitudes
    data = Experiment("noise_free_discrimination_target", time, temperature, basis@amplitudes, true_scenario.ambient)
    lower, upper = parameter_box(alternative.spec)
    lo, hi = np.log(lower), np.log(upper)
    # Both fitted baseline parameters and generating K,tau,theta are feasible
    # starts. The latter gives the exact zero-distance constant/equilibrated
    # counterexample, rather than depending on a random start to find it.
    initial = [np.log(alternative.parameters), np.log(true_scenario.parameters)]
    hashed_seed = seed ^ int(hashlib.sha256((true_scenario.name+alternative.name).encode()).hexdigest()[:8], 16)
    if starts > 2:
        initial.extend(qmc.scale(qmc.LatinHypercube(3, seed=hashed_seed).random(starts-2), lo, hi))
    initial = initial[:starts]
    cached_q, cached_r, cached_J = None, None, None

    def evaluate(q):
        nonlocal cached_q, cached_r, cached_J
        if cached_q is None or not np.array_equal(q, cached_q):
            parameters = np.exp(q)
            r = (predict(data, parameters, alternative.spec)-temperature)[1:]
            J = sensitivities(parameters, [data], alternative.spec)
            if alternative.noise_length_s is not None:
                r, _ = whiten_ou(r, time[1:], alternative.noise_length_s)
                J, _ = whiten_ou(J, time[1:], alternative.noise_length_s)
            cached_q, cached_r, cached_J = q.copy(), r/alternative.sigma, J/alternative.sigma
        return cached_r, cached_J

    attempts, best = [], None
    for start in initial:
        result = least_squares(lambda q: evaluate(q)[0], np.clip(start, lo+1e-10, hi-1e-10),
                               jac=lambda q: evaluate(q)[1], bounds=(lo, hi), max_nfev=max_nfev,
                               ftol=1e-10, xtol=1e-10, gtol=1e-7)
        row = {"success": bool(result.success), "status": int(result.status), "nfev": int(result.nfev),
               "optimality": float(result.optimality), "distance_squared": float(result.fun@result.fun),
               "parameters": np.exp(result.x).tolist(),
               "active_bounds": (np.minimum(result.x-lo, hi-result.x) < 1e-5).tolist()}
        attempts.append(row)
        if result.success and (best is None or row["distance_squared"] < best["distance_squared"]):
            best = row
    matrix = separation_map(true_scenario, alternative, time)
    fixed = squared_distance_and_gradient(matrix, amplitudes)[0]
    return {"best": best, "attempts": attempts, "fixed_parameter_distance_squared": fixed,
            "requires_revisit": best is None or best["distance_squared"] > fixed+1e-6*max(1., fixed),
            "notice": "Noise-free alternative-parameter refit, directional and bounded. The smallest converged distance is an upper bound on the globally closest alternative mean; it can overstate discrimination if a lower minimum was missed. No chi-squared calibration or simple-hypothesis error probability is assigned to the refitted distance."}
