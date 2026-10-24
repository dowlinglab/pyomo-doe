"""Independent selected-fit checks and descriptive innovation/input summaries."""
import numpy as np
from scipy.linalg import cholesky, solve_triangular

from audit_mean import reference
from audit_saved_mean import acf
from covariance import dense_covariance
from diagnostics import input_correlation
from mean import fitted_records


def harmonic_residual(data, residual):
    period = 30. if "sin_50_50" in data.name else (120. if "sin_2_50" in data.name else (300. if "sin_5_50" in data.name else None))
    if period is None:
        return None
    time = data.time[1:]-data.time[0]
    keep = time >= 300.
    t = time[keep]
    centered = (t-t.mean())/max(1., np.ptp(t))
    X = np.column_stack((np.ones(len(t)), centered, centered**2,
                         np.sin(2*np.pi*t/period), np.cos(2*np.pi*t/period)))
    coefficients = np.linalg.lstsq(X, residual[keep], rcond=None)[0]
    return {"period_s": period, "start_time_s": 300., "samples": len(t),
            "amplitude_C": float(np.linalg.norm(coefficients[-2:])),
            "sin_cos_coefficients_C": coefficients[-2:].tolist(),
            "design_condition_number": float(np.linalg.cond(X)),
            "notice": "Descriptive residual harmonic after quadratic detrending, no IID standard-error claim; only two late cycles at five-minute period"}


def evaluate(records, result, config):
    best = result["best"]
    predictions = fitted_records(records, result)
    width = 1 if result["mode"] == "constant" else 2
    coefficients = np.asarray(best["coefficients"])
    scale = best["sigma_total_ML_C"]
    quadratic = 0.
    logdet = 0.
    n = 0
    metrics = []
    arrays = {}
    forward_error = 0.
    bound = config["ambient_offset_bounds_C"]
    if np.any(coefficients[1:] < bound[0]-1e-7) or np.any(coefficients[1:] > bound[1]+1e-7):
        raise AssertionError("Ambient bound failure")
    for j, (data, predicted) in enumerate(zip(records, predictions)):
        nuisance = coefficients[1+j*width:1+(j+1)*width]
        endpoints = np.repeat(nuisance, 2) if width == 1 else nuisance
        independent = reference(data, coefficients[0], best["tau"], best["theta"], endpoints,
                                result["hold"], initial_delta=result.get("initial_temperature_offset_C", 0.),
                                preload_seconds=result.get("preload_seconds", 0.))
        forward_error = max(forward_error, float(np.max(np.abs(independent-predicted))))
        residual = data.temperature[1:]-independent[1:]
        C = dense_covariance(data.time[1:], result["kernel"], best["noise_parameters"])
        L = cholesky(C, lower=True)
        innovation = solve_triangular(L, residual, lower=True)
        standardized = innovation/scale
        quadratic += float(innovation@innovation)
        logdet += float(2*np.log(np.diag(L)).sum())
        n += len(residual)
        cross = input_correlation(standardized, data.input[1:])
        max_input = max((abs(row["correlation"]) for row in cross if row["correlation"] is not None), default=None)
        metrics.append({"name": data.name, "RMSE_C": float(np.sqrt(np.mean(residual**2))),
                        "bias_C": float(residual.mean()), "innovation_acf_1_to_30": acf(standardized),
                        "innovation_max_abs_input_correlation": max_input,
                        "innovation_mean": float(standardized.mean()), "innovation_std": float(standardized.std()),
                        "harmonic_residual": harmonic_residual(data, residual)})
        arrays.update({"time_"+str(j): data.time, "observed_"+str(j): data.temperature,
                       "predicted_"+str(j): predicted, "input_"+str(j): data.input,
                       "standardized_innovation_"+str(j): standardized})
    sigma = float(np.sqrt(quadratic/n))
    nll = .5*(n*(np.log(2*np.pi)+1+np.log(quadratic/n))+logdet)
    error = abs(nll-best["negative_log_likelihood"])
    if forward_error > 1e-7 or error > 2e-5 or abs(sigma-scale) > 1e-7:
        raise AssertionError(("Selected fit reconstruction", forward_error, error, sigma-scale))
    return {"audit": {"status": "passed", "nll_abs_error": error, "max_forward_error_C": forward_error,
                       "scope": "Independent augmented-matrix mean propagation and dense kernel likelihood; not global optimality or adequacy"},
            "per_run": metrics, "nobs": n, "logdet_unit_shape": logdet,
            "raw_RMSE_C": float(np.sqrt(np.mean(np.concatenate([(d.temperature-p)[1:] for d, p in zip(records, predictions)])**2)))}, arrays
