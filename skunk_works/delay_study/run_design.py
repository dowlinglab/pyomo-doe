"""Frozen, resumable local experiment designs; no hardware operation.

Compute only. Candidate waveforms, all optimization attempts, scenario utilities,
information spectra and an independent dense forward audit remain in each record.
"""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from data import Experiment, load
from design import (build_scenario, candidates, energy_weights, information,
                    interpolation_basis, optimize, score_and_gradient)
from inference import ModelSpec, predict, sensitivities
from noise import whiten_ou
from run_fits import save_json

HERE = Path(__file__).resolve().parent
SCENARIOS, TARGETS, CONFIG, KNOTS = {}, {}, None, None


def label(order, initialization, noise_length):
    family = "pure" if order == 0 else "n"+str(order)
    noise = "iid" if noise_length is None else "ou"+str(noise_length)
    return family+"__"+initialization+"__"+noise


def prepare(config, fits):
    """Build exact linear maps once; no estimated OU length is reused as truth."""
    time = np.arange(0., config["horizon_s"]+config["observation_interval_s"], config["observation_interval_s"])
    knots = np.arange(0., config["horizon_s"]+config["knot_interval_s"], config["knot_interval_s"])
    basis = interpolation_basis(time, knots)
    scenarios, targets, provenance = {}, {}, {}
    for order in config["ensemble_orders"]:
        for initialization in config["initializations"]:
            spec = ModelSpec(family="pure_delay" if order == 0 else "chain", order=order,
                             initialization=initialization)
            path = Path(fits)/("joint__"+spec.identifier+".json")
            record = json.loads(path.read_text())
            parameters = np.array([record["fit"]["parameters"][name] for name in spec.names])
            provenance[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            # Prediction target is unwhitened thermal-mean error under a fixed
            # 50-percent reference, averaged over the future observation times.
            target_data = Experiment("prediction_target", time, np.full(len(time), config["ambient_C"]),
                                     np.full(len(time), config["mean_input_percent"]), config["ambient_C"])
            target_J = sensitivities(parameters, [target_data], spec)
            target = target_J.T@target_J/len(target_J)
            for length in config["noise_lengths_s"]:
                name = label(order, initialization, length)
                prior = np.zeros((3, 3))
                for dataset in record["datasets"]:
                    data = load(dataset)
                    J = sensitivities(parameters, [data], spec)
                    if length is not None:
                        J, _ = whiten_ou(J, data.time[1:], length)
                    prior += J.T@J/config["sigma_C"]**2
                scenarios[name] = build_scenario(name, parameters, spec, time, basis,
                                                 ambient=config["ambient_C"], sigma=config["sigma_C"],
                                                 prior_information=prior, noise_length=length)
                targets[name] = target
    return scenarios, targets, knots, provenance


def dense_audit(scenarios, amplitudes, knots, config):
    """Re-simulate without the cached input map, including a finite shutdown tail."""
    horizon = config["horizon_s"]
    end = horizon+config["audit_tail_s"]
    shutdown = horizon+float(amplitudes[-1])/config["maximum_slew_percent_per_s"]
    if shutdown > end:
        raise ValueError("Shutdown ramp exceeds the specified audit horizon")
    time = np.unique(np.r_[np.arange(0., end+config["audit_interval_s"]/2, config["audit_interval_s"]),
                           knots, shutdown])
    # Ramp the last command down at the same allowed slew, then hold zero.
    command_time = np.r_[knots, shutdown, end] if shutdown > horizon else np.r_[knots, end]
    command_values = np.r_[amplitudes, 0., 0.] if shutdown > horizon else np.r_[amplitudes, 0.]
    command = np.interp(time, command_time, command_values)
    audit = []
    for scenario in scenarios:
        data = Experiment("dense_forward_audit", time, np.full(len(time), scenario.ambient), command, scenario.ambient)
        temperature = predict(data, scenario.parameters, scenario.spec)
        index = int(np.argmax(temperature))
        audit.append({"scenario": scenario.name, "maximum_C": float(temperature[index]),
                      "time_of_maximum_s": float(time[index]),
                      "exceeds_assumed_limit": bool(temperature[index] > config["maximum_temperature_C"]+1e-6)})
    return {"interval_s": config["audit_interval_s"], "tail_s": config["audit_tail_s"],
            "shutdown_command": "After the design horizon, ramp the last command to zero at the assumed maximum slew, then hold zero.",
            "notice": "Independent dense simulation at the selected model points, not a continuous-time bound, uncertainty guarantee or certified hardware limit.",
            "scenarios": audit}


def tasks(config):
    result = []
    for order in config["nominal_orders"]:
        for length in config["noise_lengths_s"]:
            scenario = label(order, "cold", length)
            for criterion in config["criteria"]:
                for prior in config["include_prior"]:
                    key = "nominal__"+scenario+"__"+criterion+"__prior"+str(int(prior))
                    result.append((key, [scenario], criterion, prior, "mean"))
    ensemble = [label(order, init, length) for order in config["ensemble_orders"]
                for init in config["initializations"] for length in config["noise_lengths_s"]]
    for aggregation in config["aggregations"]:
        for criterion in config["criteria"]:
            for prior in config["include_prior"]:
                key = "ensemble__"+aggregation+"__"+criterion+"__prior"+str(int(prior))
                result.append((key, ensemble, criterion, prior, aggregation))
    return result


def initialize(scenarios, targets, config, knots):
    global SCENARIOS, TARGETS, CONFIG, KNOTS
    SCENARIOS, TARGETS, CONFIG, KNOTS = scenarios, targets, config, knots


def execute(task, destination, identity):
    key, names, criterion, prior, aggregation = task
    path = Path(destination)/(key+".json")
    if path.exists():
        record = json.loads(path.read_text())
        if record["identity"] != identity:
            raise RuntimeError("Design identity changed: "+key)
        return key, "reused", record["requires_revisit"]
    selected = [SCENARIOS[name] for name in names]
    # Noise variants have identical temperature maps. Enforce the temperature
    # assumption for all sixteen fitted mean/startup combinations, even for a
    # nominal utility. These are a finite scenario set, not an uncertainty set.
    safety = [scenario for scenario in SCENARIOS.values() if scenario.noise_length_s is None]
    starts = candidates(KNOTS, CONFIG["mean_input_percent"], CONFIG["seed"])
    result = optimize(selected, KNOTS, starts, criterion, prior, aggregation,
                      mean_input=CONFIG["mean_input_percent"], maximum_temperature=CONFIG["maximum_temperature_C"],
                      maximum_slew=CONFIG["maximum_slew_percent_per_s"], maxiter=CONFIG["maximum_iterations"],
                      prediction_information=TARGETS, constraint_scenarios=safety)
    waveforms = dict(starts)
    if result["best"] is not None:
        waveforms["selected_local_design"] = np.array(result["best"]["amplitudes"])
    evaluations = {}
    for name, amplitudes in waveforms.items():
        rows = []
        for scenario in SCENARIOS.values():
            F = information(scenario, amplitudes, prior)
            values, vectors = np.linalg.eigh(F)
            scores = {target: score_and_gradient(scenario, amplitudes, target, prior, TARGETS[scenario.name])[0]
                      for target in CONFIG["criteria"]}
            rows.append({"scenario": scenario.name, "information_log_io": F.tolist(),
                         "eigenvalues_ascending": values.tolist(), "eigenvectors_columns": vectors.tolist(),
                         "scores": {k: float(v) if np.isfinite(v) else None for k, v in scores.items()}})
        evaluations[name] = {"amplitudes_percent": amplitudes.tolist(), "scenario_information": rows,
                             "mean_input_percent": float(energy_weights(KNOTS)@amplitudes),
                             "maximum_slew_percent_per_s": float(np.max(abs(np.diff(amplitudes))/np.diff(KNOTS))),
                             "dense_audit": dense_audit(safety, amplitudes, KNOTS, CONFIG)}
    best_audit = evaluations.get("selected_local_design", {}).get("dense_audit", {})
    revisit = result["best"] is None or any(row["exceeds_assumed_limit"] for row in best_audit.get("scenarios", []))
    save_json(path, {"identity": identity, "key": key, "utility_scenarios": names, "knots_s": KNOTS.tolist(),
                     "optimization": result, "evaluations": evaluations, "requires_revisit": revisit,
                     "parameter_metric": "Natural logarithms of K,tau,theta; E-optimality depends on this chosen metric.",
                     "prediction_target": "Average thermal-mean squared error along a fixed 50-percent reference command at the future observation times, with a scenario-specific derivative matrix. No future measurement-noise variance is added.",
                     "noise_assumptions": "Stationary marginal standard deviation 0.25 C with IID, OU10 s or OU60 s correlations. These are working assumptions, not known TC-Lab noise parameters.",
                     "prior_notice": "If enabled, information from the two baseline experiments is added under the same local model and working covariance. It is expected information, not a Bayesian distribution over parameters.",
                     "resource_notice": "Equal integral of percentage command over 900 s. This equals heat input only under the inherited linear heater calibration. No experiment was executed on hardware."})
    return key, "complete", revisit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", type=Path, default=HERE/"results/fits")
    parser.add_argument("--output", type=Path, default=HERE/"results/design")
    parser.add_argument("--config", type=Path, default=HERE/"design_config.json")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    selected = [task for task in tasks(config) if args.only in task[0]]
    if not selected:
        raise ValueError("No design jobs selected")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        scenarios, targets, knots, input_hashes = prepare(config, args.fits)
        identity = {"sources": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                for name in ("data.py", "models.py", "inference.py", "noise.py", "design.py", "run_design.py", "run_fits.py")},
                    "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "fit_sha256": input_hashes}
        if (args.output/"manifest.json").exists():
            old = json.loads((args.output/"manifest.json").read_text())
            if old["identity"] != identity:
                raise RuntimeError("Preserve designs from the previous scientific source")
        manifest = {"identity": identity, "filter": args.only, "expected_jobs": len(selected),
                    "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                    "completed": {}, "failed": {}}
        save_json(args.output/"manifest.json", manifest)
        with ProcessPoolExecutor(max_workers=min(4, config["workers"]), initializer=initialize,
                                 initargs=(scenarios, targets, config, knots)) as pool:
            futures = {pool.submit(execute, task, str(args.output), identity): task[0] for task in selected}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    _, status, revisit = future.result()
                    manifest["completed"][key] = {"status": status, "requires_revisit": revisit}
                    print(f"{len(manifest['completed'])}/{len(selected)} {key} revisit={revisit}", flush=True)
                except Exception:
                    manifest["failed"][key] = traceback.format_exc()
                    print("FAILED "+key, flush=True)
                save_json(args.output/"manifest.json", manifest)
        manifest["status"] = "complete" if not manifest["failed"] else "complete_with_failures"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(args.output/"manifest.json", manifest)
        return bool(manifest["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
