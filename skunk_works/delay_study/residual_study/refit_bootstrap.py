"""Full nuisance/noise refits for diagnostic calibration and parameter bootstrap."""
import copy
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from audit_saved_mean import acf
from covariance import Whitening
from covariance_fit import joint_fit
from data import Experiment, load
from diagnostics import input_correlation
from mean import fitted_records
from simulation import NoiseProcess

STATE = None


def quantization_grid(records):
    levels = np.unique(np.concatenate([d.temperature for d in records]))
    index = np.rint((levels-levels.min())/.3225)
    X = np.column_stack((np.ones(len(levels)), index))
    intercept, spacing = np.linalg.lstsq(X, levels, rcond=None)[0]
    error = float(np.max(np.abs(X@np.array([intercept, spacing])-levels)))
    exact = bool(np.array_equal(np.round(intercept+spacing*index, 2), levels))
    if not .32 <= spacing <= .33 or error > .0051 or not exact:
        raise ValueError("Observed temperatures do not support this simple rounding-grid sensitivity")
    return {"offset_C": float(intercept), "spacing_C": float(spacing),
            "unique_levels": len(levels), "max_grid_error_C": error,
            "two_decimal_levels_reconstructed": exact,
            "scope": "Grid inferred only from the reference fitting records; interpretation as a measurement mechanism remains a sensitivity"}


def statistics(records, fit):
    predictions = fitted_records(records, fit)
    raw = np.concatenate([(d.temperature-p)[1:] for d, p in zip(records, predictions)])
    W = Whitening([d.time[1:] for d in records], fit["kernel"], fit["best"]["noise_parameters"])
    standardized = W(raw)/fit["best"]["sigma_total_ML_C"]
    rows = []
    cursor = 0
    for d in records:
        count = len(d.time)-1
        e = standardized[cursor:cursor+count]
        correlations = np.asarray(acf(e))
        lags = np.arange(1, len(correlations)+1)
        Q = float(count*(count+2)*np.sum(correlations**2/(count-lags)))
        input_values = input_correlation(e, d.input[1:])
        max_input = max((abs(v["correlation"]) for v in input_values if v["correlation"] is not None), default=None)
        rows.append({"record": d.name, "acf1": float(correlations[0]),
                     "max_abs_acf_1_to_30": float(np.max(np.abs(correlations))),
                     "max_abs_input_correlation": max_input, "ljung_box_Q_30": Q})
        cursor += count
    return {"per_run": rows, "global_max_abs_acf": max(r["max_abs_acf_1_to_30"] for r in rows),
            "global_max_abs_input_correlation": max((r["max_abs_input_correlation"] for r in rows if r["max_abs_input_correlation"] is not None), default=None),
            "global_max_Q30": max(r["ljung_box_Q_30"] for r in rows)}


def initialize(reference_dir, starts, measurement, seed_base):
    global STATE
    directory = Path(reference_dir)
    fit = json.loads((directory/"fit.json").read_text())
    config = json.loads((directory/"effective_config.json").read_text())
    config["multistarts"] = starts
    records = [load(name) for name in fit["record_names"]]
    source = json.loads((directory/"source_snapshot.json").read_text())
    root = Path(__file__).resolve().parents[3]
    for data in records:
        if hashlib.sha256((root/data.source).read_bytes()).hexdigest() != source["sha256"][data.source]:
            raise ValueError("Reference raw-data binding changed")
    best = fit["best"]
    predictions = fitted_records(records, fit)
    kernel, parameters, sigma = fit["kernel"], best["noise_parameters"], best["sigma_total_ML_C"]
    grid = None
    if measurement == "rounded":
        if not kernel.endswith("_nugget"):
            raise ValueError("Rounding sensitivity requires a separated measurement component")
        grid = quantization_grid(records)
        sigma *= np.sqrt(1-parameters[1])
        parameters = [parameters[0], 0.]
    processes = [NoiseProcess(d.time[1:], kernel, parameters, sigma) for d in records]
    STATE = {"records": records, "reference": fit, "config": config, "predictions": predictions,
             "processes": processes, "measurement": measurement, "grid": grid, "seed_base": seed_base}


def synthetic(index):
    S = STATE
    seed = S["seed_base"]+104729*index
    rng = np.random.RandomState(seed)
    result = []
    for data, predicted, process in zip(S["records"], S["predictions"], S["processes"]):
        temperature = predicted.copy()
        temperature[1:] += process.draw(rng)
        if S["measurement"] == "rounded":
            q, offset = S["grid"]["spacing_C"], S["grid"]["offset_C"]
            temperature[1:] = np.round(offset+q*np.rint((temperature[1:]-offset)/q), 2)
        temperature[0] = data.temperature[0]
        result.append(Experiment(data.name, data.time.copy(), temperature, data.input.copy(),
                                 data.ambient, data.power, source="synthetic-from:"+data.source))
    return seed, result


def baseline_seeds(records):
    reference = STATE["reference"]
    best = reference["best"]
    alternatives = [r for r in reference["starts"] if r.get("success") and
                    (r["tau"] > r["theta"]) != (best["tau"] > best["theta"])]
    rows = [best]+([min(alternatives, key=lambda r: r["negative_log_likelihood"])] if alternatives else [])
    seeds = []
    for row in rows:
        model = dict(reference, best=row)
        predictions = fitted_records(records, model)
        sse = float(sum(np.sum((d.temperature[1:]-p[1:])**2) for d, p in zip(records, predictions)))
        seeds.append({"success": True, "tau": row["tau"], "theta": row["theta"],
                      "coefficients": row["coefficients"], "SSE": sse})
    return {"best": seeds[0], "starts": seeds}


def execute(index):
    S = STATE
    seed, records = synthetic(index)
    config = copy.deepcopy(S["config"])
    reference = S["reference"]
    try:
        fit = joint_fit(records, config, reference["mode"], reference["hold"], reference["kernel"],
                        baseline_seeds(records), noise_start=reference["best"]["noise_parameters"])
        status = "requires_revisit" if fit["stationarity_warning"] else "complete"
        return {"index": index, "seed": seed, "status": status, "fit": fit,
                "statistics": statistics(records, fit)}
    except Exception as exc:
        return {"index": index, "seed": seed, "status": "failed", "exception": repr(exc),
                "traceback": traceback.format_exc()}
