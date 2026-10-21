"""Conditional residual/block resampling with explicit initialization assumptions."""
from __future__ import annotations

import hashlib

import numpy as np

from data import Experiment
from inference import ModelSpec, predict
from models import to_physics


def replicate_seed(case, method, replicate, seed):
    text = f"{seed}:{case}:{method}:{replicate}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little")


def circular_indices(length, block_length, rng):
    if length < 1 or block_length < 1:
        raise ValueError("Positive sample and block lengths required")
    starts = rng.integers(0, length, size=int(np.ceil(length/block_length)))
    return ((starts[:, None]+np.arange(block_length)[None, :]) % length).ravel()[:length]


def resample_experiments(datasets, parameters, spec, seed, block_length=1, sigma=None):
    rng = np.random.default_rng(seed)
    sampled = []
    for data in datasets:
        prediction = predict(data, parameters, spec)
        residual = data.temperature[1:]-prediction[1:]
        if sigma is None:
            centered = residual-residual.mean()
            noise = centered[circular_indices(len(centered), block_length, rng)]
        else:
            noise = rng.normal(0, sigma, size=len(residual))
        output = np.asarray(prediction).copy()
        output[1:] += noise
        output[0] = data.temperature[0]
        sampled.append(Experiment(data.name, data.time.copy(), output, data.input.copy(), data.ambient,
                                  data.power, source=data.source))
    return sampled


def summarize_samples(samples, success, nonestimable, power=200.0):
    good = np.asarray(samples)[np.asarray(success)]
    if len(good) < 2:
        return {"successful_draws": len(good), "status": "insufficient_converged_draws"}
    physical = np.array([to_physics(p, power=power) for p in good])
    result = {"successful_draws": len(good), "status": "conditional_percentile_summary",
              "interval_definition": "2.5 and 97.5 percentiles of converged bootstrap refits",
              "io": {}, "physics": {}}
    for key, values, names in (("io", good, ["K", "tau", "theta"]), ("physics", physical, ["Ua", "Cp", "theta"])):
        for index, name in enumerate(names):
            if name == "theta" and "theta" in nonestimable:
                result[key][name] = {"status": "unidentifiable_no_bootstrap_interval", "lower": None, "median": None, "upper": None}
            else:
                lower, median, upper = np.quantile(values[:, index], [.025, .5, .975])
                result[key][name] = {"status": "conditional_on_model_noise_and_initialization", "lower": float(lower),
                                     "median": float(median), "upper": float(upper), "std": float(np.std(values[:, index], ddof=1))}
    if not nonestimable:
        result["cov_log_io"] = np.cov(np.log(good).T).tolist()
        result["cov_log_physics"] = np.cov(np.log(physical).T).tolist()
    else:
        result["cov_log_io"] = result["cov_log_physics"] = None
    return result
