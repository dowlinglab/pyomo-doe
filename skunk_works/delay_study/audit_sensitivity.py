"""Audit saved nuisance/reconstruction fits without rerunning optimization.

Chain nuisance means use direct state initialization, independently of the
producer's cold/equilibrated interpolation. A finite-difference Gram check is
separate from the exact arithmetic checks on the saved SVD and covariance.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import load
from inference import ModelSpec, parameter_box, predict
from models import chain_matrices, propagate_lti, simulate_pure_delay
from run_fits import save_json
from run_sensitivity import task_matrix

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_mean(data, parameters, spec, offset=0., fraction=0.):
    """Initialize the actual nuisance state rather than interpolate its output."""
    ambient = data.ambient+offset
    if spec.family == "chain":
        A, B = chain_matrices(parameters, spec.order, power=data.power)
        initial = np.r_[np.full(spec.order, fraction*data.input[0]),
                        data.temperature[0]-ambient]
        return ambient+propagate_lti(A, B, initial, data.time, data.input,
                                     spec.hold, spec.method)[:, -1]
    if spec.family == "pure_delay":
        shifted = replace(data, ambient=ambient)
        cold = simulate_pure_delay(shifted, parameters, "cold", spec.hold)
        eq = simulate_pure_delay(shifted, parameters, "input_equilibrated", spec.hold)
        return (1-fraction)*cold+fraction*eq
    raise ValueError("Direct nuisance states are defined only for delayed models")


def finite_nuisance_jacobian(datasets, parameters, spec, offsets, histories,
                             free_ambient, free_history, step=1e-4):
    count = len(datasets)
    q = np.r_[np.log(parameters), offsets if free_ambient else [],
              histories if free_history else []]

    def values(point):
        cursor = 3
        a = point[cursor:cursor+count] if free_ambient else offsets
        cursor += count*int(free_ambient)
        g = point[cursor:cursor+count] if free_history else histories
        return np.concatenate([direct_mean(d, np.exp(point[:3]), spec, ai, gi)[1:]
                               for d, ai, gi in zip(datasets, a, g)])

    columns = []
    for i in range(len(q)):
        delta = np.zeros(len(q))
        delta[i] = step
        columns.append((values(q-2*delta)-8*values(q-delta)+
                        8*values(q+delta)-values(q+2*delta))/(12*step))
    return np.column_stack(columns)


def relative_error(actual, expected):
    return float(np.linalg.norm(np.asarray(actual)-expected)/max(1., np.linalg.norm(expected)))


def audit_nuisance_geometry(fit, datasets, spec, parameters):
    offsets, histories = np.asarray(fit["ambient_offsets_C"]), np.asarray(fit["history_fractions"])
    q = np.asarray(fit["mixed_coordinates"])
    expected_q = np.r_[np.log(parameters), offsets if fit["free_ambient"] else [],
                       histories if fit["free_history"] else []]
    if not np.allclose(q, expected_q, atol=1e-12, rtol=0):
        raise RuntimeError("Mixed coordinates do not represent the saved parameters")
    lo, hi = np.asarray(fit["bounds_mixed"])
    expected_lo, expected_hi = parameter_box(spec)
    count = len(datasets)
    lower = np.r_[np.log(expected_lo), [-2.]*count if fit["free_ambient"] else [],
                  [0.]*count if fit["free_history"] else []]
    upper = np.r_[np.log(expected_hi), [2.]*count if fit["free_ambient"] else [],
                  [1.]*count if fit["free_history"] else []]
    if not np.array_equal(lo, lower) or not np.array_equal(hi, upper):
        raise RuntimeError("Saved nuisance bounds differ from the declared study")
    expected_names = spec.names+(["ambient_offset_"+str(i) for i in range(count)] if fit["free_ambient"] else [])
    expected_names += ["history_fraction_"+str(i) for i in range(count)] if fit["free_history"] else []
    if fit["mixed_coordinate_names"] != expected_names:
        raise RuntimeError("Saved mixed-coordinate labels are inconsistent")
    if np.any(q < lo-1e-10) or np.any(q > hi+1e-10):
        raise RuntimeError("Selected nuisance fit violates its bounds")
    s, V = np.asarray(fit["singular_values_mixed_metric"]), np.asarray(fit["right_singular_vectors_rows"])
    keep = s > 1e-8*s[0]
    rank = int(keep.sum())
    if rank != fit["rank_relative_1e-8"] or relative_error(V@V.T, np.eye(len(q))) > 1e-10:
        raise RuntimeError("Saved rank or right singular basis is inconsistent")
    J = finite_nuisance_jacobian(datasets, parameters, spec, offsets, histories,
                                  fit["free_ambient"], fit["free_history"])
    finite_singular = np.linalg.svd(J, compute_uv=False)
    if int(np.sum(finite_singular > 1e-8*finite_singular[0])) != rank:
        raise RuntimeError("Independent finite-difference rank disagrees with the saved numerical convention")
    gram_error = relative_error(J.T@J, V.T@np.diag(s*s)@V)
    if gram_error > 5e-4:
        raise RuntimeError("Independent finite-difference information check failed: "+str(gram_error))
    bread = (V[keep].T/s[keep]**2)@V[keep]
    covariance_error = 0.
    for key, scale in [("cov_mixed_iid_unknown_sigma", fit["sse"]/max(1, fit["n_conditional"]-len(q))),
                       ("cov_mixed_known_sigma_0p25", .25**2)]:
        if rank < len(q):
            if fit[key] is not None:
                raise RuntimeError("A rank-deficient full covariance was assigned")
        else:
            if fit[key] is None:
                raise RuntimeError("Missing full-rank conditional covariance")
            covariance_error = max(covariance_error, relative_error(fit[key], bread*scale))
    expected_nonestimable = [name for name, flag in zip(fit["mixed_coordinate_names"],
                                np.sum(V[~keep]**2, axis=0) > 1e-10) if flag]
    if expected_nonestimable != fit["nonestimable_mixed_coordinates"]:
        raise RuntimeError("Saved null-direction labels are inconsistent")
    if covariance_error > 1e-9:
        raise RuntimeError("Saved conditional covariance arithmetic failed")
    return gram_error, covariance_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--sensitivity", type=Path)
    parser.add_argument("--allow-trial", action="store_true")
    args = parser.parse_args()
    root = args.sensitivity or args.results/"sensitivity"
    manifest_path = root/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or manifest["failed"]:
        raise RuntimeError("Only a completed run can be audited")
    config_path = HERE/"sensitivity_config.json"
    config = json.loads(config_path.read_text())
    tasks = task_matrix(config)
    if manifest["filter"]:
        if not args.allow_trial:
            raise RuntimeError("A filtered trial is not the full sensitivity study")
        tasks = [t for t in tasks if manifest["filter"] in t["key"]]
    if set(manifest["completed"]) != {t["key"] for t in tasks} or manifest["expected_jobs"] != len(tasks):
        raise RuntimeError("Manifest task counts differ from the declared matrix")
    identity = manifest["identity"]
    if identity["config_sha256"] != sha(config_path):
        raise RuntimeError("Measured configuration changed")
    for name, digest in identity["source_sha256"].items():
        if sha(HERE/name) != digest:
            raise RuntimeError("Audit requires the measured scientific source: "+name)
    errors = dict.fromkeys(["trajectory_C", "sse_C2", "selected_minimum_C2", "validation_metric_C",
                           "mixed_gram_relative", "covariance_arithmetic_relative", "exchange_C"], 0.)
    inputs = {str(manifest_path): sha(manifest_path)}
    rows = []
    for task in tasks:
        key = task["key"]
        path = root/(key+".json")
        record = json.loads(path.read_text())
        baseline_path = args.results/"fits"/record["baseline_fit"]
        baseline = json.loads(baseline_path.read_text())
        if record["identity"] != identity or record["task"] != task or record["baseline_fit_sha256"] != sha(baseline_path):
            raise RuntimeError("Case source/input/task identity mismatch: "+key)
        if record["requires_revisit"] or manifest["completed"][key]["requires_revisit"]:
            raise RuntimeError("Unresolved nested-minimum flag: "+key)
        fit, spec = record["fit"], ModelSpec(**task["spec"])
        nuisance = task["phase"] == "nuisance"
        p = np.array([fit["parameters_io" if nuisance else "parameters"][n] for n in spec.names])
        datasets = [load(n) for n in task["datasets"]]
        offsets = fit["ambient_offsets_C"] if nuisance else [0.]*len(datasets)
        histories = fit["history_fractions"] if nuisance else [float(spec.initialization == "input_equilibrated")]*len(datasets)
        if not (len(offsets) == len(histories) == len(record["calibration_traces"]) == len(datasets)):
            raise RuntimeError("Incomplete calibration records: "+key)
        residuals = []
        for d, a, g, trace in zip(datasets, offsets, histories, record["calibration_traces"]):
            if trace["dataset"] != d.name or not all(np.array_equal(trace[k], value) for k, value in
                [("time_s", d.time), ("input_percent", d.input), ("measured_C", d.temperature)]):
                raise RuntimeError("Saved trace disagrees with the raw data: "+key)
            mean = direct_mean(d, p, spec, a, g) if nuisance else predict(d, p, spec)
            errors["trajectory_C"] = max(errors["trajectory_C"], float(np.max(abs(mean-trace["predicted_C"]))))
            residuals.append((mean-d.temperature)[1:])
        residual = np.concatenate(residuals)
        sse = float(residual@residual)
        errors["sse_C2"] = max(errors["sse_C2"], abs(sse-fit["sse"]),
                                  abs(record["sse_change_from_baseline"]-(sse-baseline["fit"]["sse"])))
        attempts = fit["attempts" if nuisance else "starts"]
        if len(attempts) != config["n_starts"] or not any(a["success"] for a in attempts):
            raise RuntimeError("Incomplete multistart history: "+key)
        if any(a["nfev"] > config["max_nfev"] for a in attempts):
            raise RuntimeError("An attempt exceeded the declared function-evaluation budget: "+key)
        errors["selected_minimum_C2"] = max(errors["selected_minimum_C2"],
                         abs(fit["sse"]-min(a["sse"] for a in attempts if a["success"])))
        if nuisance:
            gram, cov = audit_nuisance_geometry(fit, datasets, spec, p)
            errors["mixed_gram_relative"] = max(errors["mixed_gram_relative"], gram)
            errors["covariance_arithmetic_relative"] = max(errors["covariance_arithmetic_relative"], cov)
            exchange = fit["exchanged_n1_representation"]
            if exchange:
                swapped = p[[0,2,1]]
                lower, upper = parameter_box(spec)
                in_box = bool(np.all(swapped >= lower) and np.all(swapped <= upper))
                if not np.array_equal(exchange["parameters"], swapped) or exchange["within_io_fit_box"] != in_box:
                    raise RuntimeError("Incorrect exchanged pole parameters or box flag")
                if len(exchange["history_fractions"]) != len(datasets) or not all(0 <= g <= 1 for g in exchange["history_fractions"]):
                    raise RuntimeError("Inadmissible exchanged history fractions")
                for d, a, g, other in zip(datasets, offsets, histories, exchange["history_fractions"]):
                    delta = float(np.max(abs(direct_mean(d,p,spec,a,g)-direct_mean(d,exchange["parameters"],spec,a,other))))
                    errors["exchange_C"] = max(errors["exchange_C"], delta,
                                       abs(delta-exchange["maximum_prediction_difference_C"]))
        if nuisance and task["mode"] != "fixed":
            if record["validation"] is not None:
                raise RuntimeError("Calibration nuisance values were silently assigned to new records")
        else:
            if [v["dataset"] for v in record["validation"]] != [v["dataset"] for v in baseline["validation"]]:
                raise RuntimeError("Held-out file set changed")
            for row in record["validation"]:
                d = load(row["dataset"])
                mean = direct_mean(d,p,spec,offsets[0],histories[0]) if nuisance else predict(d,p,spec)
                e = (mean-d.temperature)[1:]
                errors["validation_metric_C"] = max(errors["validation_metric_C"],
                       abs(float(np.sqrt(e@e/len(e)))-row["rmse_C"]), abs(float(e.mean())-row["bias_C"]))
        rows.append({"case":key,"phase":task["phase"],"mode":task.get("mode","reconstruction"),
                     "calibration":task["calibration"],"family":spec.family,"order":spec.order,
                     "sse_C2":sse,"baseline_sse_C2":baseline["fit"]["sse"],
                     "rmse_C":float(np.sqrt(sse/len(residual))),
                     "rank":fit["rank_relative_1e-8"] if nuisance else fit["uncertainty"]["rank_relative_1e-8"],
                     "parameters":json.dumps(dict(zip(spec.names,p.tolist()))),
                     "ambient_offsets_C":json.dumps(offsets),"history_fractions":json.dumps(histories),
                     "failed_starts":sum(not a["success"] for a in attempts)})
        inputs[str(path)], inputs[str(baseline_path)] = sha(path), sha(baseline_path)
    tolerances = {"trajectory_C":1e-8,"sse_C2":1e-6,"selected_minimum_C2":1e-6,
                  "validation_metric_C":1e-8,"mixed_gram_relative":5e-4,
                  "covariance_arithmetic_relative":1e-9,"exchange_C":1e-8}
    if any(errors[k] > tolerances[k] for k in errors):
        raise RuntimeError("Sensitivity audit failed: "+json.dumps(errors))
    pd.DataFrame(rows).to_csv(root/"audited_summary.csv",index=False,float_format="%.12g")
    report={"status":"passed_trial" if manifest["filter"] else "passed", "cases":len(rows),
            "maximum_errors":errors,"tolerances":tolerances,"input_sha256":inputs,
            "summary_sha256":sha(root/"audited_summary.csv"),
            "notice":"Checks saved sources, data, means, SSE, selected converged starts, nuisance SVD/IID-covariance arithmetic, finite-difference information and held-out metrics. It does not certify global minima, measured nuisance probabilities, HAC covariance/coverage or physical causation. Pure-delay nuisance means reuse the independently tested pure-delay propagator; covariance arithmetic and finite-difference geometry are separate checks."}
    save_json(root/"audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in {"input_sha256","notice"}}))


if __name__ == "__main__":
    main()
