"""Variable-projection likelihood and alternating GLS for the teaching study."""
import time

import numpy as np
from scipy.optimize import least_squares, minimize

from covariance import Whitening, profiled_likelihood
from mean import design, project, starting_points


def covariance_coordinates(kernel, parameters):
    if kernel == "iid":
        return np.empty(0)
    return np.r_[np.log(parameters[0]), parameters[1:]]


def covariance_parameters(kernel, point):
    if kernel == "iid":
        return []
    return np.r_[np.exp(point[0]), point[1:]].tolist()


def covariance_bounds(kernel):
    if kernel == "iid":
        return []
    return [(np.log(.05), np.log(3000.))]+([(0., 1.)] if kernel.endswith("_nugget") else [])


def projected_state(records, config, mode, hold, kernel, point):
    tau, theta = np.exp(point[:2])
    noise = covariance_parameters(kernel, point[2:])
    W = Whitening([d.time[1:] for d in records], kernel, noise)
    coefficients, white_residual = project(records, tau, theta, mode, hold, config, W)
    n = len(white_residual)
    scale2 = float(white_residual@white_residual/n)
    if not np.isfinite(scale2) or scale2 <= 0:
        raise ValueError("Invalid projected covariance scale")
    nll = .5*(n*(np.log(2*np.pi)+1+np.log(scale2))+W.logdet)
    return float(nll), coefficients, float(np.sqrt(scale2)), W


def numerical_gradient(function, point, bounds, step=2e-5):
    gradient = []
    for j in range(len(point)):
        lo, hi = bounds[j]
        hplus = min(step, hi-point[j])
        hminus = min(step, point[j]-lo)
        if hplus > 0 and hminus > 0:
            p, m = point.copy(), point.copy()
            p[j] += min(hplus, hminus)
            m[j] -= min(hplus, hminus)
            gradient.append((function(p)-function(m))/(2*min(hplus, hminus)))
        elif hplus > 0:
            p = point.copy(); p[j] += hplus
            gradient.append((function(p)-function(point))/hplus)
        else:
            m = point.copy(); m[j] -= hminus
            gradient.append((function(point)-function(m))/hminus)
    return np.asarray(gradient)


def projected_gradient(point, gradient, bounds):
    projected = gradient.copy()
    for j, (lo, hi) in enumerate(bounds):
        if (point[j] <= lo+1e-6 and gradient[j] > 0) or (point[j] >= hi-1e-6 and gradient[j] < 0):
            projected[j] = 0
    return projected


def joint_fit(records, config, mode, hold, kernel, baseline, callback=None, alternating=None):
    bounds = [tuple(np.log(config["thermal_bounds"][k])) for k in ("tau", "theta")]+covariance_bounds(kernel)
    baseline_best = baseline["best"]
    base_times = [baseline_best["tau"], baseline_best["theta"]]
    pole_seeds = starting_points(config)
    pole_seeds[0] = base_times
    # Preserve the other OLS basin when it is available, without relabeling it.
    alternatives = [s for s in baseline["starts"] if s.get("success") and
                    (s["tau"] > s["theta"]) != (base_times[0] > base_times[1])]
    if alternatives and len(pole_seeds) > 1:
        alt = min(alternatives, key=lambda r: r["SSE"])
        pole_seeds[1] = [alt["tau"], alt["theta"]]
    noise_seeds = [[3., .2], [30., .05], [100., .5], [300., .9]]
    rows = []
    best = None
    begun = time.monotonic()
    objective = lambda x: projected_state(records, config, mode, hold, kernel, x)[0]
    for index, thermal in enumerate(pole_seeds):
        noise = noise_seeds[index % len(noise_seeds)]
        coords = [] if kernel == "iid" else [np.log(noise[0])]+([noise[1]] if kernel.endswith("_nugget") else [])
        point = np.r_[np.log(thermal), coords]
        if index == 2 and alternating is not None:
            last = alternating["history"][-1]
            point = np.r_[np.log([last["tau"], last["theta"]]),
                          covariance_coordinates(kernel, last["noise_parameters"])]
        try:
            fit = minimize(objective, point, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 600, "maxfun": 5000, "ftol": 1e-12,
                                    "gtol": 1e-5, "maxls": 35, "eps": 2e-6})
            nll, coefficients, sigma, _ = projected_state(records, config, mode, hold, kernel, fit.x)
            gradient = numerical_gradient(objective, fit.x, bounds)
            pg = float(np.max(np.abs(projected_gradient(fit.x, gradient, bounds))))
            row = {"start": index, "initial_coordinates": point.tolist(), "coordinates": fit.x.tolist(),
                   "success": bool(fit.success), "message": str(fit.message), "nfev": int(fit.nfev),
                   "nit": int(fit.nit), "negative_log_likelihood": nll,
                   "tau": float(np.exp(fit.x[0])), "theta": float(np.exp(fit.x[1])),
                   "noise_parameters": covariance_parameters(kernel, fit.x[2:]),
                   "sigma_total_ML_C": sigma, "coefficients": coefficients.tolist(),
                   "projected_gradient_max_abs": pg,
                   "active_bounds": [(-1 if x <= b[0]+1e-6 else (1 if x >= b[1]-1e-6 else 0)) for x, b in zip(fit.x, bounds)]}
            if row["success"] and (best is None or nll < best["negative_log_likelihood"]):
                best = row
        except Exception as exc:
            row = {"start": index, "initial_coordinates": point.tolist(), "success": False,
                   "exception": type(exc).__name__+": "+str(exc)}
        rows.append(row)
        if callback:
            callback(row)
    if best is None:
        raise RuntimeError("No converged joint-likelihood start")
    return {"mode": mode, "hold": hold, "kernel": kernel, "best": best,
            "starts": rows, "elapsed_s": time.monotonic()-begun,
            "record_names": [d.name for d in records],
            "nobs": sum(len(d.time)-1 for d in records),
            "coordinate_order": ["log_tau", "log_theta"]+([] if kernel == "iid" else ["log_ell"]+(["nugget_variance_fraction"] if kernel.endswith("_nugget") else [])),
            "noise_parameter_order": [] if kernel == "iid" else ["length_s"]+(["nugget_variance_fraction"] if kernel.endswith("_nugget") else []),
            "noise_bounds": covariance_bounds(kernel),
            "stationarity_warning": best["projected_gradient_max_abs"] > .05}


def alternating_fit(records, config, mode, hold, kernel, baseline, noise_start):
    times = np.log([baseline["best"]["tau"], baseline["best"]["theta"]])
    coefficients = np.asarray(baseline["best"]["coefficients"])
    noise = covariance_coordinates(kernel, noise_start)
    times_bounds = np.log(np.array([config["thermal_bounds"][k] for k in ("tau", "theta")])).T
    noise_bounds = covariance_bounds(kernel)
    history = []
    stable = 0
    previous = None
    for iteration in range(50):
        X, y = design(records, *np.exp(times), mode, hold)
        raw = X@coefficients-y
        def noise_objective(x):
            W = Whitening([d.time[1:] for d in records], kernel, covariance_parameters(kernel, x))
            return profiled_likelihood(raw, W)[0]
        if kernel != "iid":
            nf = minimize(noise_objective, noise, method="L-BFGS-B", bounds=noise_bounds,
                          options={"ftol": 1e-12, "gtol": 1e-5, "maxiter": 400, "maxls": 35, "eps": 2e-6})
            noise = nf.x
            noise_success = bool(nf.success)
        else:
            noise_success = True
        W = Whitening([d.time[1:] for d in records], kernel, covariance_parameters(kernel, noise))
        residual = lambda x: project(records, *np.exp(x), mode, hold, config, W)[1]
        mf = least_squares(residual, times, bounds=times_bounds, max_nfev=600,
                           ftol=1e-11, xtol=1e-11, gtol=1e-8, diff_step=2e-6)
        times = mf.x
        coefficients, _ = project(records, *np.exp(times), mode, hold, config, W)
        X, y = design(records, *np.exp(times), mode, hold)
        nll, sigma, _ = profiled_likelihood(X@coefficients-y, W)
        scaled = np.r_[times, noise, coefficients/np.r_[1., np.full(len(coefficients)-1, 2.)], np.log(sigma)]
        delta = None if previous is None else float(np.max(np.abs(scaled-previous["scaled"])))
        likelihood_delta = None if previous is None else abs(nll-previous["nll"])/max(1., abs(nll))
        if delta is not None and delta < 1e-4 and likelihood_delta < 1e-7 and noise_success and mf.success:
            stable += 1
        else:
            stable = 0
        history.append({"iteration": iteration, "negative_log_likelihood": nll,
                        "tau": float(np.exp(times[0])), "theta": float(np.exp(times[1])),
                        "coefficients": coefficients.tolist(), "noise_parameters": covariance_parameters(kernel, noise),
                        "sigma_total_ML_C": sigma, "noise_solver_success": noise_success,
                        "mean_solver_success": bool(mf.success), "max_scaled_change": delta,
                        "relative_likelihood_change": likelihood_delta})
        previous = {"scaled": scaled, "nll": nll}
        if stable >= 2:
            break
    return {"history": history, "converged": stable >= 2, "iteration_cap": 50,
            "scope": "Block coordinate likelihood fitting with full temporal whitening; numerical stability does not establish residual adequacy"}
