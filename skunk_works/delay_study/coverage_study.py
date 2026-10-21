"""Controlled coverage experiments; this does not validate the real-data mean model."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from data import Experiment
from inference import predict


def sample_errors(time, noise, rng):
    sigma = noise["sigma_C"]
    if noise["kind"] == "iid":
        return sigma*rng.normal(size=len(time))
    if noise["kind"] != "ou":
        raise ValueError("Unknown simulation noise")
    length = noise["length_s"]
    errors = np.empty(len(time))
    errors[0] = sigma*rng.normal()
    for i, delta in enumerate(np.diff(time), 1):
        phi = np.exp(-delta/length)
        errors[i] = phi*errors[i-1]+sigma*np.sqrt(-np.expm1(-2*delta/length))*rng.normal()
    return errors


def simulate_experiments(datasets, truth, spec, noise, seed):
    rng = np.random.default_rng(seed)
    result = []
    for data in datasets:
        mean = predict(data, truth, spec)
        errors = sample_errors(data.time[1:], noise, rng)
        observed = mean.copy()
        observed[1:] += errors
        # The conditioned initial temperature is held exactly at the known mean.
        result.append(Experiment(data.name, data.time.copy(), observed, data.input.copy(), data.ambient,
                                 data.power, source="Synthetic observations based on "+data.source))
    return result


def local_intervals(parameters, truth, covariance):
    if covariance is None:
        return {"status": "undefined_local_covariance", "covered": None, "lower": None, "upper": None}
    covariance = np.asarray(covariance)
    if np.any(np.diag(covariance) < 0):
        raise ValueError("Negative local variance")
    halfwidth = norm.ppf(.975)*np.sqrt(np.diag(covariance))
    lower, upper = np.exp(np.log(parameters)-halfwidth), np.exp(np.log(parameters)+halfwidth)
    return {"status": "conditional_log_wald_interval", "lower": lower.tolist(), "upper": upper.tolist(),
            "covered": ((truth >= lower) & (truth <= upper)).tolist(),
            "width": (upper-lower).tolist()}


def profile_interval_summary(components, truth):
    if not components:
        return {"covered": False, "component_count": 0, "total_component_width": 0., "envelope_width": 0.,
                "open_at_exploration_bound": False}
    return {"covered": any(c["lower"] <= truth <= c["upper"] for c in components),
            "component_count": len(components),
            "total_component_width": sum(c["upper"]-c["lower"] for c in components),
            "envelope_width": max(c["upper"] for c in components)-min(c["lower"] for c in components),
            "open_at_exploration_bound": any("open_at_exploration_bound" in [c["lower_status"], c["upper_status"]] for c in components)}


def wilson_interval(successes, trials, confidence=.95):
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Invalid binomial counts")
    if trials == 0:
        return None
    z = norm.ppf((1+confidence)/2)
    p = successes/trials
    center = (p+z*z/(2*trials))/(1+z*z/trials)
    radius = z*np.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/(1+z*z/trials)
    lower = 0. if successes == 0 else float(max(0., center-radius))
    upper = 1. if successes == trials else float(min(1., center+radius))
    return [lower, upper]
