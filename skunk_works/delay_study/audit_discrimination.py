"""Recompute separation from direct means, including every alternative refit.

Uses no input-to-response separation maps or optimizer objective helper. The
simulator and finite scenario set are shared with the separately checked study.
Distances after refitting remain upper bounds on global closest alternatives.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from audit_design import DirectStudy, command_resources, scenario_label, sha
from audit_prediction_uncertainty import compare_values
from design import candidates
from inference import parameter_box, predict
from run_fits import save_json

HERE = Path(__file__).resolve().parent
START_NAMES = {"constant", "sine_300s", "sine_150s", "multisine",
               "random_binary_knots", "alternating_knots"}


def direct_distance(first, second, time, length, sigma):
    """Stationary Gaussian precision norm on retained times; first point omitted."""
    delta = (np.asarray(first)-np.asarray(second))[1:]
    if length is not None:
        phi = np.exp(-np.diff(np.asarray(time)[1:])/length)
        delta = np.r_[delta[0], (delta[1:]-phi*delta[:-1])/np.sqrt(1-phi**2)]
    return float(delta@delta/sigma**2)


def direct_mean(study, name, amplitudes, parameters=None):
    model = study.scenarios[name][0]
    fitted, spec = study.models[model]
    if parameters is not None:
        return predict(study.experiment(amplitudes), np.asarray(parameters), spec)
    key = (model, np.asarray(amplitudes, dtype=float).tobytes())
    if key not in study.mean_cache:
        study.mean_cache[key] = predict(study.experiment(amplitudes), fitted, spec)
    return study.mean_cache[key]


def distances(study, names, amplitudes, length):
    means = {name: direct_mean(study, name, amplitudes) for name in names}
    return [direct_distance(means[a], means[b], study.time, length, study.config["sigma_C"])
            for a, b in combinations(names, 2)]


def audit_dense(evaluation, study, amplitudes, selected, errors):
    dense = evaluation["dense_audit"]
    if (dense["interval_s"], dense["tail_s"]) != (study.config["audit_interval_s"], study.config["audit_tail_s"]):
        raise RuntimeError("Dense discrimination audit horizon changed")
    expected = study.dense(amplitudes)
    if len(dense["scenarios"]) != len(expected) or {r["scenario"] for r in dense["scenarios"]} != set(expected):
        raise RuntimeError("Missing or duplicate dense mean scenario")
    for row in dense["scenarios"]:
        reference = expected[row["scenario"]]
        for field in ("maximum_C", "time_of_maximum_s"):
            compare_values(row[field], reference[field], "dense_"+field, errors, atol=1e-7, rtol=1e-9)
        if row["exceeds_assumed_limit"] != reference["exceeds_assumed_limit"] or (selected and reference["exceeds_assumed_limit"]):
            raise RuntimeError("Dense discrimination temperature flag failed")


def audit_design_record(record, task, study, config, errors):
    key, initialization, length, aggregation = task
    if record["key"] != key or record["phase"] != "designs" or record["requires_revisit"]:
        raise RuntimeError("Design identity or unresolved discrimination flag")
    compare_values(record["knots_s"], study.knots, "knots_s", errors)
    optimization, evaluations = record["optimization"], record["evaluations"]
    if optimization["aggregation"] != aggregation:
        raise RuntimeError("Separation aggregation changed")
    attempts = optimization["attempts"]
    if set(evaluations) != START_NAMES|{"selected_local_design"} or len(attempts) != 6 or {a["start"] for a in attempts} != START_NAMES:
        raise RuntimeError("Missing or duplicate candidate/attempt")
    starts = candidates(study.knots, study.config["mean_input_percent"], config["seed"])
    names = [scenario_label(order, initialization, length) for order in config["orders"]]
    pairs = list(combinations(names, 2))
    values = {}
    for name, evaluation in evaluations.items():
        amplitudes = np.asarray(evaluation["amplitudes_percent"])
        if name in starts:
            compare_values(amplitudes, starts[name], "candidate_"+name, errors)
        resources = command_resources(amplitudes, study.knots)
        for field in ("mean_input_percent", "maximum_slew_percent_per_s"):
            compare_values(evaluation[field], resources[field], field, errors)
        if (resources["minimum_input_percent"] < -1e-7 or resources["maximum_input_percent"] > 100+1e-7
                or abs(resources["mean_input_percent"]-study.config["mean_input_percent"]) > 1e-6
                or resources["maximum_slew_percent_per_s"] > study.config["maximum_slew_percent_per_s"]+1e-6):
            raise RuntimeError("Discrimination command violates amplitude/dose/slew assumptions")
        rows = evaluation["pairs"]
        if len(rows) != len(pairs) or [(r["first"], r["second"]) for r in rows] != pairs:
            raise RuntimeError("Missing, duplicate or reordered fixed-mean pair")
        values[name] = distances(study, names, amplitudes, length)
        for row, distance in zip(rows, values[name]):
            compare_values(row["distance_squared"], distance, "fixed_distance_squared", errors, atol=1e-7)
            error = .5*erfc(sqrt(distance)/sqrt(8.))
            compare_values(row["ideal_fixed_gaussian_equal_prior_error"], error, "ideal_gaussian_error", errors, atol=1e-13)
        audit_dense(evaluation, study, amplitudes, name == "selected_local_design", errors)
    reference = np.asarray(values[config["reference"]])
    if np.any(reference <= 0):
        raise RuntimeError("The reference does not separate all fixed pairs")
    compare_values(optimization["reference_distance_squared"], reference, "reference_distances", errors)
    for attempt in attempts:
        if attempt.get("status") == "zero_separation_start":
            if attempt["success"] or min(values[attempt["start"]]) > 0:
                raise RuntimeError("A positive-separation candidate was labelled skipped")
            continue
        if not 0 <= attempt["iterations"] <= study.config["maximum_iterations"]:
            raise RuntimeError("Separation attempt exceeds declared iteration budget")
        amplitudes = np.asarray(attempt["amplitudes_percent"])
        actual = np.asarray(distances(study, names, amplitudes, length))
        if np.any(actual <= 0):
            if attempt["objective"] is not None:
                raise RuntimeError("Finite objective assigned to zero separation")
        else:
            relative = np.log(actual/reference)
            objective = -float(relative.mean() if aggregation == "mean" else relative.min())
            compare_values(attempt["objective"], objective, "attempt_objective", errors, atol=1e-7)
        compare_values(attempt["maximum_constraint_violation"], study.coarse_violation(amplitudes),
                       "coarse_constraint_violation", errors, atol=1e-7)
    accepted = [a for a in attempts if a["success"] and a.get("objective") is not None and a["maximum_constraint_violation"] < 1e-6]
    if not accepted or optimization["best"] != min(accepted, key=lambda a: a["objective"]):
        raise RuntimeError("Selected separation design is not the best converged feasible attempt")
    selected = np.asarray(evaluations["selected_local_design"]["amplitudes_percent"])
    compare_values(selected, optimization["best"]["amplitudes_percent"], "selected_amplitudes", errors)
    if study.coarse_violation(selected) >= 1e-6:
        raise RuntimeError("Selected discrimination design violates coarse constraints")
    return {"case": key, "phase": "designs", "initialization": initialization, "noise_length_s": length,
            "aggregation": aggregation, "relative_log_distance_utility": -optimization["best"]["objective"],
            "smallest_selected_pair_distance_squared": min(values["selected_local_design"]),
            "successful_feasible_starts": len(accepted),
            "dense_maximum_C": max(r["maximum_C"] for r in study.dense(selected).values())}


def audit_refit_record(record, task, study, config, errors, designs):
    key, initialization, length, waveform, order = task
    generating = scenario_label(config["refit_generating_order"], initialization, length)
    alternative = scenario_label(order, initialization, length)
    if (record["key"], record["phase"], record["generating_scenario"], record["alternative"], record["requires_revisit"]) != (key, "refits", generating, alternative, False):
        raise RuntimeError("Directional refit identity or unresolved flag")
    compare_values(record["knots_s"], study.knots, "knots_s", errors)
    if waveform.startswith("selected_"):
        name = initialization+"__"+("iid" if length is None else "ou"+str(length))+"__"+waveform.split("_", 1)[1]+".json"
        path = Path(designs)/name
        source = json.loads(path.read_text())
        if source["requires_revisit"] or record["source_design_sha256"] != {name: sha(path)}:
            raise RuntimeError("Refit source design identity changed")
        amplitudes = np.asarray(source["evaluations"]["selected_local_design"]["amplitudes_percent"])
    else:
        amplitudes = candidates(study.knots, study.config["mean_input_percent"], config["seed"])[waveform]
        if record["source_design_sha256"]:
            raise RuntimeError("Common candidate has an undeclared selected-design dependency")
    compare_values(record["amplitudes_percent"], amplitudes, "refit_amplitudes", errors)
    truth, fitted = direct_mean(study, generating, amplitudes), direct_mean(study, alternative, amplitudes)
    fixed = direct_distance(truth, fitted, study.time, length, study.config["sigma_C"])
    result = record["refit"]
    compare_values(result["fixed_parameter_distance_squared"], fixed, "refit_fixed_distance_squared", errors, atol=1e-7)
    spec = study.models[study.scenarios[alternative][0]][1]
    lower, upper = parameter_box(spec)
    attempts = result["attempts"]
    if len(attempts) != config["refit_starts"]:
        raise RuntimeError("Alternative refit did not preserve every declared attempt")
    for attempt in attempts:
        parameters = np.asarray(attempt["parameters"])
        if parameters.shape != (3,) or not np.isfinite(parameters).all() or np.any(parameters < lower*(1-1e-12)) or np.any(parameters > upper*(1+1e-12)):
            raise RuntimeError("Alternative parameters outside declared box")
        q, lo, hi = np.log(parameters), np.log(lower), np.log(upper)
        if attempt["active_bounds"] != (np.minimum(q-lo, hi-q) < 1e-5).tolist():
            raise RuntimeError("Alternative active-bound flags disagree")
        if (not 1 <= attempt["nfev"] <= config["refit_max_nfev"] or not np.isfinite(attempt["optimality"])
                or attempt["optimality"] < 0 or attempt["success"] != (attempt["status"] > 0)):
            raise RuntimeError("Alternative convergence/budget record is inconsistent")
        distance = direct_distance(truth, direct_mean(study, alternative, amplitudes, parameters), study.time, length, study.config["sigma_C"])
        compare_values(attempt["distance_squared"], distance, "refitted_distance_squared", errors, atol=1e-7)
    accepted = [a for a in attempts if a["success"]]
    if not accepted or result["best"] != min(accepted, key=lambda a: a["distance_squared"]):
        raise RuntimeError("Refit is not the smallest recorded converged distance")
    distance = result["best"]["distance_squared"]
    if result["requires_revisit"] or distance > fixed+1e-6*max(1., fixed):
        raise RuntimeError("Refit fails its feasible fixed-parameter upper bound")
    if waveform == "constant" and initialization == "input_equilibrated":
        # All orders share K/(1+tau*s) here; theta is unobservable. Matching
        # generating K,tau is a feasible exact witness within the common box.
        witness = study.models[study.scenarios[generating][0]][0]
        witness_distance = direct_distance(truth, direct_mean(study, alternative, amplitudes, witness), study.time, length, study.config["sigma_C"])
        if witness_distance > 1e-12 or distance > 1e-10:
            raise RuntimeError("Constant equilibrated zero-distance witness was missed")
    return {"case": key, "phase": "refits", "initialization": initialization, "noise_length_s": length,
            "waveform": waveform, "alternative_order": order, "fixed_distance_squared": fixed,
            "closest_found_distance_squared_upper_bound": distance, "successful_starts": len(accepted),
            "selected_parameter_on_bound": any(result["best"]["active_bounds"])}


def declared_tasks(config, phase):
    tasks = []
    for initialization in config["initializations"]:
        for length in config["noise_lengths_s"]:
            prefix = initialization+"__"+("iid" if length is None else "ou"+str(length))
            if phase == "designs":
                tasks.extend((prefix+"__"+a, initialization, length, a) for a in config["aggregations"])
            else:
                tasks.extend((prefix+"__"+w+"__alternative"+str(n), initialization, length, w, n)
                             for w in config["refit_waveforms"] for n in config["refit_alternative_orders"])
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("designs", "refits"), required=True)
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--discrimination", type=Path)
    parser.add_argument("--designs", type=Path)
    parser.add_argument("--allow-trial", action="store_true")
    args = parser.parse_args()
    root = args.discrimination or args.results/("discrimination_"+args.phase)
    designs = args.designs or args.results/"discrimination_designs"
    manifest_path = root/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or manifest["failed"] or (manifest["filter"] and not args.allow_trial):
        raise RuntimeError("Completed production or explicit completed trial required")
    config_path, design_path = HERE/"discrimination_config.json", HERE/"design_config.json"
    config, design_config = json.loads(config_path.read_text()), json.loads(design_path.read_text())
    identity = manifest["identity"]
    sources = {"models.py", "data.py", "inference.py", "noise.py", "design.py", "run_design.py",
               "discrimination.py", "run_discrimination.py", "run_fits.py"}
    if (identity["phase"] != args.phase or set(identity["sources"]) != sources
            or identity["config_sha256"] != sha(config_path) or identity["design_config_sha256"] != sha(design_path)
            or any(sha(HERE/n) != h for n, h in identity["sources"].items())):
        raise RuntimeError("Audit requires measured scientific source/configuration")
    study = DirectStudy(design_config, args.results/"fits")
    if study.inputs != identity["fit_sha256"]:
        raise RuntimeError("Discrimination fitted inputs changed")
    tasks = [t for t in declared_tasks(config, args.phase) if manifest["filter"] in t[0]]
    if (len(tasks) != manifest["expected_jobs"] or set(manifest["completed"]) != {t[0] for t in tasks}
            or (not manifest["filter"] and len(tasks) != {"designs": 12, "refits": 72}[args.phase])):
        raise RuntimeError("Discrimination task matrix incomplete")
    inputs, errors, rows = {str(manifest_path): sha(manifest_path)}, {}, []
    if args.phase == "refits":
        source_path = designs/"manifest.json"
        source = json.loads(source_path.read_text())
        expected_identity = {k: v for k, v in identity.items() if not k.startswith("source_design")}
        expected_identity["phase"] = "designs"
        expected_names = {t[0] for t in declared_tasks(config, "designs")}
        if (sha(source_path) != identity["source_design_manifest_sha256"] or source["status"] != "complete"
                or source["failed"] or source["filter"] or source["expected_jobs"] != 12
                or set(source["completed"]) != expected_names or source["identity"] != expected_identity
                or any(r["requires_revisit"] for r in source["completed"].values())
                or identity["source_design_sha256"] != {n+".json": sha(designs/(n+".json")) for n in expected_names}):
            raise RuntimeError("Refit source-design manifest or collection changed")
        audit_path = designs/"audit.json"
        audit = json.loads(audit_path.read_text())
        if audit["status"] != "passed" or audit["checked_measurement_identity"] != source["identity"] or audit["input_sha256"].get(str(source_path)) != sha(source_path):
            raise RuntimeError("Full fixed-mean design audit must pass before refit audit")
        for name, digest in identity["source_design_sha256"].items():
            if audit["input_sha256"].get(str(designs/name)) != digest:
                raise RuntimeError("Source design no longer matches its independent audit")
            inputs[str(designs/name)] = digest
        inputs.update({str(source_path): sha(source_path), str(audit_path): sha(audit_path)})
    for task in tasks:
        path = root/(task[0]+".json")
        record = json.loads(path.read_text())
        if record["identity"] != identity or manifest["completed"][task[0]]["requires_revisit"]:
            raise RuntimeError("Discrimination record source/flag mismatch")
        row = (audit_design_record(record, task, study, config, errors) if args.phase == "designs"
               else audit_refit_record(record, task, study, config, errors, designs))
        rows.append(row)
        inputs[str(path)] = sha(path)
    summary = root/"audited_summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False, float_format="%.12g")
    report = {"status": "passed_trial" if manifest["filter"] else "passed", "phase": args.phase,
              "cases": len(rows), "maximum_errors": errors, "checked_measurement_identity": identity,
              "auditor_sources_sha256": {n: sha(HERE/n) for n in ["audit_discrimination.py", "audit_design.py", "audit_prediction_uncertainty.py"]},
              "input_sha256": inputs, "summary_sha256": sha(summary),
              "scope": "Direct mean simulation/common Gaussian precision for every saved fixed pair and alternative-refit attempt; candidate resources/dense maxima, objectives, budget/box/active-bound flags, best converged feasible selection and exact equilibrated constant-input witness. Reuses checked simulator; no global separation optimum, real-data testing power, continuous-time temperature bound or hardware certificate. Refit optimality scalars are checked for finite nonnegative values, not independently differentiated."}
    save_json(root/"audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"input_sha256", "checked_measurement_identity", "scope"}}))


if __name__ == "__main__":
    main()
