"""Resumable bootstrap refits; stores every successful and failed attempt."""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from data import load
from inference import fit, ModelSpec
from bootstrap import resample_experiments, replicate_seed, summarize_samples
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def save_arrays(path, parameters, success, completed, sse):
    temporary = path.with_name(path.stem+".tmp.npz")
    np.savez_compressed(temporary, parameters=parameters, success=success, completed=completed, sse=sse)
    temporary.replace(path)


def execute(task):
    input_path, method, config, output, identity, replicate_override = task
    record = json.loads(Path(input_path).read_text())
    spec = ModelSpec(**record["fit"]["spec"])
    if spec.coordinates != "io" or spec.family == "original":
        raise ValueError("Bootstrap uses identified IO fits; physical draws are transformed exactly")
    key = Path(input_path).stem+"__"+method
    path = Path(output)/(key+".json")
    arrays_path = Path(output)/(key+".npz")
    target = config["bootstrap_parametric_replicates"] if method == "parametric" else config["bootstrap_replicates"]
    target = replicate_override or target
    input_sha = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    completed = np.zeros(target, dtype=bool)
    success = completed.copy()
    parameters = np.full((target, 3), np.nan)
    sse = np.full(target, np.nan)
    attempts = {}
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["identity"] != identity or previous["fit_sha256"] != input_sha or previous["target_draws"] != target:
            raise RuntimeError("Bootstrap provenance/target mismatch: "+key)
        if previous["status"] in {"complete", "complete_with_fit_failures"}:
            return key, "reused", previous["successful_draws"], target
        if not arrays_path.exists():
            raise RuntimeError("Missing bootstrap checkpoint arrays: "+key)
        arrays = np.load(arrays_path)
        completed, success, parameters, sse = [arrays[name].copy() for name in ("completed", "success", "parameters", "sse")]
        attempts = previous["attempts"]
    datasets = [load(name) for name in record["datasets"]]
    initial = np.array([record["fit"]["parameters"][name] for name in spec.names])
    nonestimable = record["uncertainty"]["nonestimable_native_parameters"]
    state = {"identity": identity, "fit_file": Path(input_path).name, "fit_sha256": input_sha,
             "target_draws": target, "method": method, "status": "running", "attempts": attempts,
             "replicate_override": replicate_override,
             "noise_model": "IID Gaussian sigma fixed at configured known_sigma_C" if method == "parametric" else "Experiment-centered circular block residual resampling",
             "limits": "Conditional on chosen mean model, fixed first-temperature reading, ambient, heater calibration and prehistory. Blocks assume within-experiment stationarity. Resampling does not remove mean-model bias or identify a structural null direction."}
    if not arrays_path.exists():
        save_arrays(arrays_path, parameters, success, completed, sse)
    save_json(path, state)
    for replicate in range(target):
        if completed[replicate]:
            continue
        seed = replicate_seed(Path(input_path).stem, method, replicate, config["seed"])
        block = 1 if method == "parametric" else int(method.removeprefix("block"))
        sigma = config["known_sigma_C"] if method == "parametric" else None
        sampled = resample_experiments(datasets, initial, spec, seed, block, sigma)
        entry = {"seed": seed, "attempts": []}
        for number_of_starts, max_nfev in ((config["bootstrap_starts"], config["max_nfev"]), (max(4, config["bootstrap_starts"]), 2*config["max_nfev"])):
            try:
                result = fit(sampled, spec, n_starts=number_of_starts, seed=seed, starting=initial, max_nfev=max_nfev)
                parameters[replicate] = [result["parameters"][name] for name in spec.names]
                sse[replicate] = result["sse"]
                success[replicate] = True
                entry["attempts"].append({"n_starts": number_of_starts, "max_nfev": max_nfev,
                                           "status": "converged", "starts": result["starts"]})
                break
            except Exception:
                entry["attempts"].append({"n_starts": number_of_starts, "max_nfev": max_nfev,
                                           "status": "failed", "error": traceback.format_exc()})
        completed[replicate] = True
        attempts[str(replicate)] = entry
        if (replicate+1) % 10 == 0 or replicate == target-1:
            save_arrays(arrays_path, parameters, success, completed, sse)
            state.update(completed_draws=int(completed.sum()), successful_draws=int(success.sum()))
            save_json(path, state)
    state["summary"] = summarize_samples(parameters, success, nonestimable, power=datasets[0].power)
    state["status"] = "complete" if success.all() else "complete_with_fit_failures"
    state["arrays_sha256"] = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    save_json(path, state)
    return key, state["status"], int(success.sum()), target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", type=Path, default=HERE/"results/fits")
    parser.add_argument("--output", type=Path, default=HERE/"results/bootstrap")
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    parser.add_argument("--only", default="")
    parser.add_argument("--replicates", type=int, default=None, help="Explicit short-trial override, recorded in every result")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"source_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ("models.py", "data.py", "inference.py", "bootstrap.py", "run_bootstrap.py", "run_fits.py")},
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()}
    with (args.output/"run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks = []
        for path in sorted(args.fits.glob("*.json")):
            record = json.loads(path.read_text())
            spec = ModelSpec(**record["fit"]["spec"])
            if spec.coordinates != "io" or spec.family == "original":
                continue
            for method in ["block"+str(b) for b in config["bootstrap_blocks_samples"]]+["parametric"]:
                if args.only in path.stem+"__"+method:
                    tasks.append((str(path), method, config, str(args.output), identity, args.replicates))
        manifest = {"identity": identity, "expected_jobs": len(tasks), "started_utc": datetime.now(timezone.utc).isoformat(),
                    "completed": {}, "failed": {}, "status": "running", "filter": args.only,
                    "replicate_override": args.replicates,
                    "formulation_policy": "Fit in IO coordinates; transform every draw to Ua,Cp,theta. Original physical uncertainty is structurally undefined except identified functions; its zero-state transfer is represented by the cold one-stage model."}
        save_json(args.output/"manifest.json", manifest)
        with ProcessPoolExecutor(max_workers=min(config["workers"], 4)) as pool:
            futures = {pool.submit(execute, task): Path(task[0]).stem+"__"+task[1] for task in tasks}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    _, status, successful, target = future.result()
                    manifest["completed"][key] = {"status": status, "successful": successful, "target": target}
                    print(f"{len(manifest['completed'])}/{len(tasks)} {key} {successful}/{target}", flush=True)
                except Exception:
                    manifest["failed"][key] = traceback.format_exc()
                    print("FAILED "+key, flush=True)
                save_json(args.output/"manifest.json", manifest)
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "complete" if not manifest["failed"] else "complete_with_failures"
        save_json(args.output/"manifest.json", manifest)
        return bool(manifest["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
