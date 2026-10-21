"""Separate fixed-mean input optimization from alternative-parameter refits."""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
from itertools import combinations
import json
from pathlib import Path
import traceback

import numpy as np

from design import candidates, energy_weights, interpolation_basis
from discrimination import (fixed_gaussian_error, optimize_separation,
                            refitted_separation, separation_map, squared_distance_and_gradient)
from run_design import dense_audit, label, prepare
from run_fits import save_json

HERE = Path(__file__).resolve().parent
STATE = None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_key(initialization, length):
    return initialization+"__"+("iid" if length is None else "ou"+str(length))


def tasks(config, phase):
    result = []
    for initialization in config["initializations"]:
        for length in config["noise_lengths_s"]:
            group = group_key(initialization, length)
            if phase == "designs":
                result.extend((group+"__"+aggregation, initialization, length, aggregation)
                              for aggregation in config["aggregations"])
            else:
                result.extend((group+"__"+waveform+"__alternative"+str(order), initialization, length, waveform, order)
                              for waveform in config["refit_waveforms"] for order in config["refit_alternative_orders"])
    return result


def initialize(state):
    global STATE
    STATE = state


def execute(task, phase, output, identity):
    config, design_config, scenarios, knots, time, basis, design_directory = STATE
    key, initialization, length = task[:3]
    destination = Path(output)/(key+".json")
    if destination.exists():
        record = json.loads(destination.read_text())
        if record["identity"] != identity:
            raise RuntimeError("Discrimination reuse identity changed")
        return key, "reused", record["requires_revisit"]
    safety = [scenario for scenario in scenarios.values() if scenario.noise_length_s is None]
    starts = candidates(knots, design_config["mean_input_percent"], config["seed"])
    source_designs = {}
    if phase == "designs":
        aggregation = task[3]
        selected = [scenarios[label(order, initialization, length)] for order in config["orders"]]
        pairs = list(combinations(selected, 2))
        maps = [separation_map(first, second, time) for first, second in pairs]
        result = optimize_separation(maps, knots, starts, safety, starts[config["reference"]], aggregation,
                                     mean_input=design_config["mean_input_percent"],
                                     maximum_temperature=design_config["maximum_temperature_C"],
                                     maximum_slew=design_config["maximum_slew_percent_per_s"],
                                     maxiter=design_config["maximum_iterations"])
        waveforms = dict(starts)
        if result["best"]:
            waveforms["selected_local_design"] = np.asarray(result["best"]["amplitudes_percent"])
        evaluations = {}
        for name, amplitudes in waveforms.items():
            rows = []
            for (first, second), matrix in zip(pairs, maps):
                distance, _ = squared_distance_and_gradient(matrix, amplitudes)
                rows.append({"first": first.name, "second": second.name, "distance_squared": distance,
                             "ideal_fixed_gaussian_equal_prior_error": fixed_gaussian_error(distance)})
            evaluations[name] = {"amplitudes_percent": amplitudes.tolist(), "pairs": rows,
                                 "mean_input_percent": float(energy_weights(knots)@amplitudes),
                                 "maximum_slew_percent_per_s": float(np.max(abs(np.diff(amplitudes))/np.diff(knots))),
                                 "dense_audit": dense_audit(safety, amplitudes, knots, design_config)}
        revisit = result["best"] is None or any(row["exceeds_assumed_limit"] for row in
                     evaluations.get("selected_local_design", {}).get("dense_audit", {}).get("scenarios", []))
        payload = {"optimization": result, "evaluations": evaluations}
    else:
        waveform, order = task[3:]
        if waveform.startswith("selected_"):
            aggregation = waveform.split("_", 1)[1]
            path = Path(design_directory)/(group_key(initialization, length)+"__"+aggregation+".json")
            source = json.loads(path.read_text())
            if sha(path) != identity["source_design_sha256"][path.name]:
                raise RuntimeError("Selected source design changed after the refit phase started")
            if source["requires_revisit"]:
                raise RuntimeError("Selected discrimination design requires review")
            source_designs[path.name] = sha(path)
            amplitudes = np.asarray(source["evaluations"]["selected_local_design"]["amplitudes_percent"])
        else:
            amplitudes = starts[waveform]
        generating = scenarios[label(config["refit_generating_order"], initialization, length)]
        alternative = scenarios[label(order, initialization, length)]
        result = refitted_separation(generating, alternative, amplitudes, time, basis,
                                     config["refit_starts"], config["seed"], config["refit_max_nfev"])
        revisit = result["requires_revisit"]
        payload = {"generating_scenario": generating.name, "alternative": alternative.name,
                   "amplitudes_percent": amplitudes.tolist(), "refit": result,
                   "source_design_sha256": source_designs}
    save_json(destination, {"identity": identity, "key": key, "phase": phase, "knots_s": knots.tolist(),
                            "requires_revisit": revisit, "scope": config["scope"], **payload,
                            "assumptions": "Common known hypothetical covariance, baseline fitted mean parameters, finite horizon and the parameter-design amplitude/dose/slew/temperature assumptions. Gaussian fixed-mean error is an ideal simple-hypothesis calculation only; model discrepancy and nuisance fitting invalidate its use as real experimental power. No hardware run."})
    return key, "complete", revisit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("designs", "refits"), required=True)
    parser.add_argument("--fits", type=Path, default=HERE/"results/fits")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--designs", type=Path, default=HERE/"results/discrimination_designs")
    parser.add_argument("--config", type=Path, default=HERE/"discrimination_config.json")
    parser.add_argument("--design-config", type=Path, default=HERE/"design_config.json")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    design_config = json.loads(args.design_config.read_text())
    selected = [task for task in tasks(config, args.phase) if args.only in task[0]]
    if not selected:
        raise ValueError("No discrimination jobs selected")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        scenarios, targets, knots, inputs = prepare(design_config, args.fits)
        time = np.arange(0., design_config["horizon_s"]+design_config["observation_interval_s"], design_config["observation_interval_s"])
        basis = interpolation_basis(time, knots)
        sources = ("models.py", "data.py", "inference.py", "noise.py", "design.py", "run_design.py",
                   "discrimination.py", "run_discrimination.py", "run_fits.py")
        identity = {"sources": {name: sha(HERE/name) for name in sources}, "fit_sha256": inputs,
                    "config_sha256": sha(args.config), "design_config_sha256": sha(args.design_config), "phase": args.phase}
        if args.phase == "refits":
            source_manifest = json.loads((args.designs/"manifest.json").read_text())
            if source_manifest["status"] != "complete" or source_manifest["failed"] or set(source_manifest["completed"]) != {task[0] for task in tasks(config, "designs")}:
                raise RuntimeError("All twelve fixed-mean designs must finish before refit diagnostics")
            if any(row["requires_revisit"] for row in source_manifest["completed"].values()):
                raise RuntimeError("Unresolved discrimination design flag")
            expected_source_identity = dict(identity, phase="designs")
            if source_manifest["identity"] != expected_source_identity:
                raise RuntimeError("Design scientific sources/configuration changed before refit diagnostics")
            identity["source_design_manifest_sha256"] = sha(args.designs/"manifest.json")
            identity["source_design_sha256"] = {task[0]+".json": sha(args.designs/(task[0]+".json"))
                                                 for task in tasks(config, "designs")}
        if (args.output/"manifest.json").exists() and json.loads((args.output/"manifest.json").read_text())["identity"] != identity:
            raise RuntimeError("Preserve outputs from earlier discrimination sources")
        manifest = {"identity": identity, "status": "running", "expected_jobs": len(selected), "filter": args.only,
                    "started_utc": datetime.now(timezone.utc).isoformat(), "completed": {}, "failed": {}, "scope": config["scope"]}
        save_json(args.output/"manifest.json", manifest)
        state = (config, design_config, scenarios, knots, time, basis, str(args.designs))
        with ProcessPoolExecutor(max_workers=min(4, config["workers"]), initializer=initialize, initargs=(state,)) as pool:
            futures = {pool.submit(execute, task, args.phase, str(args.output), identity): task[0] for task in selected}
            for future in as_completed(futures):
                try:
                    key, status, revisit = future.result()
                    manifest["completed"][key] = {"status": status, "requires_revisit": revisit}
                    print(str(len(manifest["completed"]))+"/"+str(len(selected))+" "+key+" revisit="+str(revisit), flush=True)
                except Exception:
                    manifest["failed"][futures[future]] = traceback.format_exc()
                    print("FAILED "+futures[future], flush=True)
                save_json(args.output/"manifest.json", manifest)
        manifest["status"] = "complete" if not manifest["failed"] else "complete_with_failures"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(args.output/"manifest.json", manifest)
        return bool(manifest["failed"] or any(row["requires_revisit"] for row in manifest["completed"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
