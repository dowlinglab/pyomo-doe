"""Exact Gaussian OU residual likelihood at the actual observation times.

This is a working residual model, not a claim that thermal model discrepancy
is a stationary Ornstein-Uhlenbeck process. Separate experiments have independent
initial residuals. The first temperature that initializes the mean is excluded.
"""
from __future__ import annotations

import numpy as np

from inference import predict, sensitivities


def whiten_ou(values, time, length):
    """Whiten vectors or sensitivity columns for unit stationary OU variance."""
    values, time = np.asarray(values, dtype=float), np.asarray(time, dtype=float)
    if len(values) != len(time) or not len(time) or np.any(np.diff(time) <= 0):
        raise ValueError("Matching values and increasing times required")
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Positive finite correlation length required")
    delta = np.diff(time)/length
    phi = np.exp(-delta)
    variance = -np.expm1(-2*delta)
    result = values.copy()
    if values.ndim == 1:
        result[1:] = (values[1:]-phi*values[:-1])/np.sqrt(variance)
    elif values.ndim == 2:
        result[1:] = (values[1:]-phi[:, None]*values[:-1])/np.sqrt(variance[:, None])
    else:
        raise ValueError("A vector or matrix is required")
    return result, float(np.log(variance).sum())


def profile_ou(residual_groups, time_groups, length, known_sigma=None):
    """Profile a shared stationary variance; retain the determinant term.

    Returns the likelihood and its derivative in log correlation length, along
    with the whitened residuals. Omitting the determinant would favor arbitrary
    correlation scales and would not define a Gaussian likelihood.
    """
    if len(residual_groups) != len(time_groups) or not residual_groups:
        raise ValueError("Nonempty matched experiment groups required")
    if known_sigma is not None and (not np.isfinite(known_sigma) or known_sigma <= 0):
        raise ValueError("Known standard deviation must be positive and finite")
    whitened, derivatives = [], []
    logdet, dlogdet = 0., 0.
    for residual, time in zip(residual_groups, time_groups):
        residual, time = np.asarray(residual), np.asarray(time)
        if residual.ndim != 1:
            raise ValueError("Each residual group must be a vector")
        white, determinant = whiten_ou(residual, time, length)
        delta = np.diff(time)/length
        phi = np.exp(-delta)
        dphi = phi*delta
        variance = -np.expm1(-2*delta)
        derivative = np.zeros(len(residual))
        derivative[1:] = -dphi*residual[:-1]/np.sqrt(variance)+white[1:]*phi*dphi/variance
        whitened.append(white)
        derivatives.append(derivative)
        logdet += determinant
        dlogdet += float(np.sum(-2*phi*dphi/variance))
    n = sum(map(len, whitened))
    rss = float(sum(w@w for w in whitened))
    sigma2 = rss/n if known_sigma is None else known_sigma**2
    if sigma2 <= 0 or not np.isfinite(sigma2):
        raise ValueError("Positive finite residual variance required")
    nll = .5*(n*np.log(2*np.pi*sigma2)+rss/sigma2+logdet)
    derivative = sum(w@dw for w, dw in zip(whitened, derivatives))/sigma2+.5*dlogdet
    return {"negative_log_likelihood": float(nll), "sigma_ML_C": float(np.sqrt(sigma2)),
            "weighted_sse_C2": rss, "n_conditional": n, "logdet_correlation": logdet,
            "derivative_log_length": float(derivative), "whitened_residuals": whitened}


def objective(log_parameters_and_length, datasets, spec, known_sigma=None):
    parameters = np.exp(np.asarray(log_parameters_and_length[:-1]))
    length = float(np.exp(log_parameters_and_length[-1]))
    residuals = [(predict(data, parameters, spec)-data.temperature)[1:] for data in datasets]
    times = [data.time[1:] for data in datasets]
    result = profile_ou(residuals, times, length, known_sigma)
    variance = result["sigma_ML_C"]**2
    gradient = np.zeros(len(parameters)+1)
    for data, white in zip(datasets, result["whitened_residuals"]):
        J = sensitivities(parameters, [data], spec, log=True)
        whitened_J, _ = whiten_ou(J, data.time[1:], length)
        gradient[:-1] += whitened_J.T@white/variance
    gradient[-1] = result["derivative_log_length"]
    return result["negative_log_likelihood"], gradient
