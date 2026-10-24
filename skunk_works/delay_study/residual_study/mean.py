"""Exact n=1 command response and bounded ambient variable projection."""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import least_squares, lsq_linear


@njit(cache=True)
def _unit_response(time, command, tau, theta, right_hold):
    output = np.zeros(len(time))
    lag = 0.0
    for k in range(len(time)-1):
        h = time[k+1]-time[k]
        a = h*(1.0/tau-1.0/theta)
        et = np.exp(-h/tau)
        eh = np.exp(-h/theta)
        # Stable repeated-pole limit; the other branch avoids overflow in exprel.
        if abs(a) < 1e-5:
            cross = et*h/tau*(1+a/2+a*a/6+a*a*a/24+a**4/120)
        else:
            cross = theta/(theta-tau)*(eh-et)
        u = command[k+int(right_hold)]
        output[k+1] = et*output[k]+u*(-np.expm1(-h/tau))+(lag-u)*cross
        lag = eh*lag+u*(-np.expm1(-h/theta))
    return output


def unit_response(data, tau, theta, hold="left"):
    if hold not in {"left", "right"} or min(tau, theta) <= 0:
        raise ValueError("Positive time scales and a constant hold required")
    return _unit_response(data.time, data.input, tau, theta, hold == "right")


def ambient_basis(time, tau, mode):
    elapsed = time-time[0]
    a = -np.expm1(-elapsed/tau)
    if mode == "constant":
        return a[:, None]
    if mode != "linear":
        raise ValueError("Unknown ambient mode")
    # Coefficients multiply the two endpoint offsets, each bounded independently.
    b = (elapsed-tau*a)/elapsed[-1]
    return np.column_stack((a-b, b))


def design(records, tau, theta, mode, hold):
    nuisance = 1 if mode == "constant" else 2
    nobs = sum(len(d.time)-1 for d in records)
    X = np.zeros((nobs, 1+nuisance*len(records)))
    y = np.empty(nobs)
    offset = 0
    for j, data in enumerate(records):
        count = len(data.time)-1
        section = slice(offset, offset+count)
        X[section, 0] = unit_response(data, tau, theta, hold)[1:]
        X[section, 1+j*nuisance:1+(j+1)*nuisance] = ambient_basis(data.time, tau, mode)[1:]
        y[section] = data.temperature[1:]-data.temperature[0]
        offset += count
    return X, y


def project(records, tau, theta, mode, hold, config, whitening=None):
    X, y = design(records, tau, theta, mode, hold)
    if whitening is not None:
        X, y = whitening(X), whitening(y)
    lo, hi = config["ambient_offset_bounds_C"]
    lower = np.full(X.shape[1], lo)
    upper = np.full(X.shape[1], hi)
    lower[0], upper[0] = config["thermal_bounds"]["K"]
    fit = lsq_linear(X, y, bounds=(lower, upper), tol=1e-11, lsmr_tol=1e-12)
    if not fit.success:
        raise RuntimeError("Bounded linear projection failed: "+fit.message)
    return fit.x, X@fit.x-y


def starting_points(config):
    rng = np.random.RandomState(config["seed"])
    bounds = np.array([config["thermal_bounds"][k] for k in ("tau", "theta")])
    seeds = [[150, 16], [16, 150], [80, 80], [300, 4]]
    while len(seeds) < config["multistarts"]:
        seeds.append(np.exp(rng.uniform(np.log(bounds[:, 0]), np.log(bounds[:, 1]))).tolist())
    return seeds[:config["multistarts"]]


def fit_mean(records, config, mode="constant", hold="left", callback=None):
    bounds = np.array([config["thermal_bounds"][k] for k in ("tau", "theta")])
    starts = []
    best = None
    for index, seed in enumerate(starting_points(config)):
        try:
            def residual(logtimes):
                return project(records, *np.exp(logtimes), mode, hold, config)[1]
            fit = least_squares(residual, np.log(seed), bounds=np.log(bounds).T,
                                ftol=1e-10, xtol=1e-10, gtol=1e-8,
                                max_nfev=600, diff_step=1e-5)
            tau, theta = np.exp(fit.x)
            coefficients, r = project(records, tau, theta, mode, hold, config)
            row = {"start": index, "seed_tau_theta": seed, "success": bool(fit.success),
                   "message": fit.message, "nfev": fit.nfev, "SSE": float(r@r),
                   "tau": float(tau), "theta": float(theta),
                   "coefficients": coefficients.tolist(), "optimality": float(fit.optimality),
                   "time_bound_mask": fit.active_mask.tolist()}
            if best is None or row["SSE"] < best["SSE"]:
                best = row
        except Exception as exc:
            row = {"start": index, "seed_tau_theta": seed, "success": False,
                   "exception": type(exc).__name__+": "+str(exc)}
        starts.append(row)
        if callback:
            callback(row)
    if best is None:
        raise RuntimeError("Every mean-fit start failed")
    return {"mode": mode, "hold": hold, "nobs": sum(len(d.time)-1 for d in records),
            "record_names": [d.name for d in records], "best": best, "starts": starts,
            "all_starts_successful": all(s["success"] for s in starts)}


def fitted_records(records, result):
    best = result["best"]
    c = np.asarray(best["coefficients"])
    width = 1 if result["mode"] == "constant" else 2
    fitted = []
    for j, data in enumerate(records):
        pred = data.temperature[0]+c[0]*unit_response(data, best["tau"], best["theta"], result["hold"])
        pred += ambient_basis(data.time, best["tau"], result["mode"])@c[1+j*width:1+(j+1)*width]
        fitted.append(pred)
    return fitted
