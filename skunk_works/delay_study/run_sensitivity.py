"""Prespecified ambient/history and input-reconstruction sensitivity matrix."""
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

from data import load
from inference import ModelSpec, fit, predict, uncertainty
from nuisance import fit_nuisance, prediction as nuisance_prediction
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def task_matrix(config):
    tasks = []
    for order in config["orders"]:
        family = "chain" if order else "pure_delay"
        for calibration, datasets in config["calibrations"].items():
            for mode in config["nuisance_modes"]:
                spec = ModelSpec(family=family, order=order)
                tasks.append({"key": calibration+"__"+spec.identifier+"__"+mode, "phase": "nuisance", "mode": mode,
                              "calibration": calibration, "datasets": datasets, "spec": asdict(spec)})
        for initialization in config["initializations"]:
            spec = ModelSpec(family=family, order=order, initialization=initialization)
            for offset in config["ambient_offsets_C"]:
                tasks.append({"key": "joint__"+spec.identifier+"__ambient_fixed_"+str(offset), "phase": "nuisance", "mode": "fixed",
                              "ambient_offset_C": offset, "calibration": "joint", "datasets": ["sine", "step"], "spec": asdict(spec)})
            for hold in ("left", "right"):
                other = ModelSpec(family=family, order=order, initialization=initialization, hold=hold)
                tasks.append({"key": "joint__"+other.identifier, "phase": "reconstruction", "calibration": "joint",
                              "datasets": ["sine", "step"], "spec": asdict(other)})
            if order:
                other = ModelSpec(order=order, initialization=initialization, hold="right", method="backward_euler")
                tasks.append({"key": "joint__"+other.identifier, "phase": "reconstruction", "calibration": "joint",
                              "datasets": ["sine", "step"], "spec": asdict(other)})
    for hold, method in (("left", "exact"), ("right", "exact"), ("right", "backward_euler")):
        spec = ModelSpec(family="original", order=0, coordinates="physics", hold=hold, method=method)
        tasks.append({"key": "joint__"+spec.identifier, "phase": "reconstruction", "calibration": "joint",
                      "datasets": ["sine", "step"], "spec": asdict(spec)})
    return tasks


def baseline_path(task, fits):
    spec = ModelSpec(**task["spec"])
    initialization = spec.initialization
    if task.get("mode") == "ambient_equilibrated":
        initialization = "input_equilibrated"
    reference = ModelSpec(family=spec.family, order=spec.order, coordinates=spec.coordinates, initialization=initialization)
    return Path(fits)/(task["calibration"]+"__"+reference.identifier+".json")


def execute(task, config, fits, output, identity):
    path = Path(output)/(task["key"]+".json")
    source_path = baseline_path(task, fits)
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["identity"] != identity or previous["baseline_fit_sha256"] != source_sha:
            raise RuntimeError("Sensitivity identity changed")
        return "reused", previous["requires_revisit"]
    baseline = json.loads(source_path.read_text())
    data = [load(name) for name in task["datasets"]]
    spec = ModelSpec(**task["spec"])
    starting = np.array([baseline["fit"]["parameters"][name] for name in spec.names])
    options = {}
    if task["phase"] == "nuisance":
        mode = task["mode"]
        options = {"free_ambient": mode in {"ambient_cold", "ambient_equilibrated", "ambient_history"},
                   "free_history": mode in {"history", "ambient_history"},
                   "ambient_offset": task.get("ambient_offset_C", 0.),
                   "history_fraction": .5 if mode in {"history", "ambient_history"} else
                       float(mode == "ambient_equilibrated" or (mode == "fixed" and spec.initialization == "input_equilibrated"))}
        result = fit_nuisance(data, spec, starting, **options, n_starts=config["n_starts"],
                              seed=config["seed"], max_nfev=config["max_nfev"])
        parameters = np.array([result["parameters_io"][name] for name in spec.names])
        predicted = [nuisance_prediction(d, parameters, spec, a, g)
                     for d, a, g in zip(data, result["ambient_offsets_C"], result["history_fractions"])]
        sse = result["sse"]
        # The added nuisance family contains its fixed-history baseline only
        # when ambient is free or history starts at the chosen endpoint.
        revisit = mode != "fixed" and sse > baseline["fit"]["sse"]+1e-5
        validation = None
        validation_notice = "Nuisance ambient/history was fitted separately for each calibration record. No automatic nuisance value is assigned to a new validation record; measured ambient/history or an explicit transfer scenario is required. Baseline held-out predictions remain the comparable fixed-assumption reference."
        if mode == "fixed":
            validation = []
            for row in baseline["validation"]:
                experiment = load(row["dataset"])
                residual = (nuisance_prediction(experiment, parameters, spec, options["ambient_offset"], options["history_fraction"])-experiment.temperature)[1:]
                validation.append({"dataset": experiment.name, "rmse_C": float(np.sqrt(residual@residual/len(residual))),
                                   "bias_C": float(residual.mean())})
            validation_notice = "The same hypothetical fixed ambient offset and prescribed history fraction are applied to every held-out record; no validation observation was used to fit those nuisance quantities."
    else:
        result = fit(data, spec, n_starts=config["n_starts"], seed=config["seed"], starting=starting, max_nfev=config["max_nfev"])
        result["uncertainty"] = uncertainty(result, data)
        parameters = np.array([result["parameters"][name] for name in spec.names])
        predicted = [predict(d, parameters, spec) for d in data]
        sse = result["sse"]
        revisit = False
        validation = []
        for row in baseline["validation"]:
            experiment = load(row["dataset"])
            residual = (predict(experiment, parameters, spec)-experiment.temperature)[1:]
            validation.append({"dataset": experiment.name, "rmse_C": float(np.sqrt(residual@residual/len(residual))),
                               "bias_C": float(residual.mean())})
        validation_notice = "Same eleven duplicate-excluded records, using the explicitly declared input reconstruction and prescribed startup."
    traces = [{"dataset": d.name, "time_s": d.time.tolist(), "input_percent": d.input.tolist(),
               "measured_C": d.temperature.tolist(), "predicted_C": values.tolist(),
               "mean_residual_C_excluding_initial": float(np.mean((values-d.temperature)[1:]))}
              for d, values in zip(data, predicted)]
    save_json(path, {"identity": identity, "task": task, "options": options, "fit": result,
                     "baseline_fit": source_path.name, "baseline_fit_sha256": source_sha,
                     "baseline_sse": baseline["fit"]["sse"], "sse_change_from_baseline": sse-baseline["fit"]["sse"],
                     "calibration_traces": traces, "validation": validation, "validation_notice": validation_notice,
                     "requires_revisit": revisit,
                     "notice": "Sensitivity analysis of assumptions. A lower calibration residual after adding nuisance parameters is expected and is not independent validation. Fixed ambient offsets are hypothetical perturbations, not measured uncertainty; left/right holds reconstruct unobserved intersample commands differently."})
    return "complete", revisit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", type=Path, default=HERE/"results/fits")
    parser.add_argument("--output", type=Path, default=HERE/"results/sensitivity")
    parser.add_argument("--config", type=Path, default=HERE/"sensitivity_config.json")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    tasks = [task for task in task_matrix(config) if args.only in task["key"]]
    if not tasks:
        raise RuntimeError("No sensitivity tasks selected")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"source_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                  for name in ("models.py", "data.py", "inference.py", "nuisance.py", "run_sensitivity.py", "run_fits.py")},
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()}
    with (args.output/"run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/"manifest.json").exists():
            old = json.loads((args.output/"manifest.json").read_text())
            if old["identity"] != identity:
                raise RuntimeError("Preserve sensitivity results from earlier sources")
        manifest = {"identity": identity, "expected_jobs": len(tasks), "filter": args.only,
                    "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "completed": {}, "failed": {}}
        save_json(args.output/"manifest.json", manifest)
        with ProcessPoolExecutor(max_workers=min(config["workers"], 4)) as pool:
            futures = {pool.submit(execute, task, config, str(args.fits), str(args.output), identity): task["key"] for task in tasks}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    status, revisit = future.result()
                    manifest["completed"][key] = {"status": status, "requires_revisit": revisit}
                    print(f"{len(manifest['completed'])}/{len(tasks)} {key} revisit={revisit}", flush=True)
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
