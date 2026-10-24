"""Conditional mean/future-observation bands with explicit ambient scenarios."""
import numpy as np
from scipy.special import ndtr

from covariance import Whitening
from mean import fitted_records, unit_response
from refit_bootstrap import moving_blocks


def future_mean(data, fit, best, ambient=0.):
    tau, theta = best["tau"], best["theta"]
    elapsed = data.time-data.time[0]
    response = data.temperature[0]+best["coefficients"][0]*unit_response(data, tau, theta, fit["hold"], fit.get("preload_seconds", 0.))
    response += fit.get("initial_temperature_offset_C", 0.)*np.exp(-elapsed/tau)
    return response+ambient*(-np.expm1(-elapsed/tau))


def gaussian_mixture_quantile(means, scales, probability):
    means, scales = np.asarray(means), np.asarray(scales)
    if np.any(scales <= 0):
        raise ValueError("Nonpositive future-observation scale")
    lower = means.min(axis=0)-8*scales.max()
    upper = means.max(axis=0)+8*scales.max()
    for _ in range(34):
        middle = (lower+upper)/2
        cdf = ndtr((middle[None, :]-means)/scales[:, None]).mean(axis=0)
        lower = np.where(cdf < probability, middle, lower)
        upper = np.where(cdf >= probability, middle, upper)
    quantile = (lower+upper)/2
    error = float(np.max(np.abs(ndtr((quantile[None, :]-means)/scales[:, None]).mean(axis=0)-probability)))
    if error > 2e-6:
        raise AssertionError("Mixture quantile did not converge")
    return quantile, error


def empirical_innovations(records, fit):
    predictions = fitted_records(records, fit)
    raw = np.concatenate([(d.temperature-p)[1:] for d, p in zip(records, predictions)])
    W = Whitening([d.time[1:] for d in records], fit["kernel"], fit["best"]["noise_parameters"])
    e = W(raw)/fit["best"]["sigma_total_ML_C"]
    result = {}
    cursor = 0
    for d in records:
        count = len(d.time)-1
        values = e[cursor:cursor+count]
        role = "step" if "step_50" in d.name else "sine"
        if role in result:
            raise ValueError("Expected one sine and one step calibration")
        result[role] = values-values.mean()
        cursor += count
    return result


def bands(data, fit, draws, method, empirical=None, block_length=60, seed=1729, paths_per_draw=8):
    bests = [row["fit"]["best"] for row in draws]
    means = np.vstack([future_mean(data, fit, best)[1:] for best in bests])
    responses = np.vstack([-np.expm1(-(data.time[1:]-data.time[0])/best["tau"]) for best in bests])
    result = {"mean_lower": np.quantile(means, .025, axis=0),
              "mean_upper": np.quantile(means, .975, axis=0),
              "ambient_mean_lower": np.quantile(means-2*responses, .025, axis=0),
              "ambient_mean_upper": np.quantile(means+2*responses, .975, axis=0)}
    audit = {"draws": len(draws), "method": method, "pointwise_not_simultaneous": True}
    if method == "gaussian":
        scales = np.array([b["sigma_total_ML_C"] for b in bests])
        result["observation_lower"], e1 = gaussian_mixture_quantile(means, scales, .025)
        result["observation_upper"], e2 = gaussian_mixture_quantile(means, scales, .975)
        result["ambient_observation_lower"], e3 = gaussian_mixture_quantile(means-2*responses, scales, .025)
        result["ambient_observation_upper"], e4 = gaussian_mixture_quantile(means+2*responses, scales, .975)
        audit["max_mixture_CDF_error"] = max(e1, e2, e3, e4)
        # Check the Monte Carlo parameter distribution at declared time fractions.
        indices = np.unique(np.linspace(0, means.shape[1]-1, 9).astype(int))
        half = len(draws)//2
        half_lower, _ = gaussian_mixture_quantile(means[:half, indices], scales[:half], .025)
        half_upper, _ = gaussian_mixture_quantile(means[:half, indices], scales[:half], .975)
        full_width = result["observation_upper"][indices]-result["observation_lower"][indices]
        audit["half_draw_max_relative_width_change_at_9_times"] = float(np.max(np.abs((half_upper-half_lower)-full_width)/np.maximum(full_width, 1e-12)))
    elif method == "innovation_blocks":
        if empirical is None:
            raise ValueError("Missing calibration innovation distribution")
        errors = []
        for row, best in zip(draws, bests):
            rng = np.random.RandomState(seed+104729*row["index"])
            innovations = np.column_stack([moving_blocks(empirical, len(data.time)-1, block_length, rng) for _ in range(paths_per_draw)])
            W = Whitening([data.time[1:]], fit["kernel"], best["noise_parameters"])
            errors.append((W.inverse(innovations)*best["sigma_total_ML_C"]).T)
        paths = np.concatenate(errors)
        all_means = np.repeat(means, paths_per_draw, axis=0)
        all_responses = np.repeat(responses, paths_per_draw, axis=0)
        observations = all_means+paths
        result["observation_lower"] = np.quantile(observations, .025, axis=0)
        result["observation_upper"] = np.quantile(observations, .975, axis=0)
        result["ambient_observation_lower"] = np.quantile(observations-2*all_responses, .025, axis=0)
        result["ambient_observation_upper"] = np.quantile(observations+2*all_responses, .975, axis=0)
        indices = np.unique(np.linspace(0, means.shape[1]-1, 9).astype(int))
        half = len(draws)//2*paths_per_draw
        q = np.quantile(observations[:half, indices], [.025, .975], axis=0)
        full_width = result["observation_upper"][indices]-result["observation_lower"][indices]
        audit["future_paths_per_parameter_draw"] = paths_per_draw
        audit["half_draw_max_relative_width_change_at_9_times"] = float(np.max(np.abs(q[1]-q[0]-full_width)/np.maximum(full_width, 1e-12)))
        audit["transport_assumption"] = "Step calibration innovations used for step evaluation; sine calibration innovations used for every evaluation sine period. This is a working transport sensitivity and cannot correct deterministic input bias."
    else:
        raise ValueError("Unknown prediction method")
    return result, audit


def coverage(values, lower, upper):
    outside = (values < lower) | (values > upper)
    longest = current = 0
    for value in outside:
        current = current+1 if value else 0
        longest = max(longest, current)
    return {"fraction_inside": float(np.mean(~outside)), "mean_width_C": float(np.mean(upper-lower)),
            "max_consecutive_outside_samples": longest,
            "notice": "Pointwise empirical fraction; serial dependence and reused records prevent independent-binomial or simultaneous-coverage interpretations."}


def conditional_ambient_reconstruction(data, fit):
    mean = future_mean(data, fit, fit["best"])
    response = -np.expm1(-(data.time[1:]-data.time[0])/fit["best"]["tau"])
    W = Whitening([data.time[1:]], fit["kernel"], fit["best"]["noise_parameters"])
    a = W(response)
    y = W(data.temperature[1:]-mean[1:])
    raw = float(a@y/(a@a))
    delta = float(np.clip(raw, -2., 2.))
    adjusted = mean.copy(); adjusted[1:] += delta*response
    return adjusted, {"fitted_offset_C": delta, "unbounded_offset_C": raw,
                      "RMSE_C": float(np.sqrt(np.mean((data.temperature[1:]-adjusted[1:])**2))),
                      "notice": "Full evaluation response used to fit an ambient-channel correction. This is a retrospective conditional reconstruction, not a prospective prediction or measured room temperature."}


def quantile_preflight():
    means = np.array([[0., 22., 50.], [0., 22., 50.]])
    sigma = np.array([.2, .2])
    low, _ = gaussian_mixture_quantile(means, sigma, .025)
    high, _ = gaussian_mixture_quantile(means, sigma, .975)
    error = max(float(np.max(np.abs(low-(means[0]-1.959963984540054*.2)))),
                float(np.max(np.abs(high-(means[0]+1.959963984540054*.2)))))
    if error > 1e-7:
        raise AssertionError("Gaussian-mixture quantile limit failed")
    return {"status": "passed", "identical_component_Gaussian_limit_error_C": error}
