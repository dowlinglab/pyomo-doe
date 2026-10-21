"""Restartable independent Monte Carlo datasets with delay likelihood profiles.

One immutable phase, bounded parallel chunks, atomic records after every draw.
Failed fits/profiles remain visible and must be included in coverage accounting.
"""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from bootstrap import replicate_seed
from coverage_study import simulate_experiments, local_intervals, profile_interval_summary
from data import load
from inference import ModelSpec, fit, uncertainty
from models import to_physics
from profiles import profile
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def save_draw(path, result):
    content = (json.dumps(result, separators=(",", ":"), allow_nan=False)+"\n").encode()
    temporary = path.with_name(path.name+".tmp")
    temporary.write_bytes(gzip.compress(content, mtime=0))
    temporary.replace(path)


def one_draw(record, config, noise_name, replicate, draw_identity):
    spec = ModelSpec(**record["fit"]["spec"])
    truth = np.array([record["fit"]["parameters"][name] for name in spec.names])
    noise = config["noise_scenarios"][noise_name]
    key = record["calibration"]+"__"+spec.identifier
    seed = replicate_seed(key, noise_name, replicate, config["seed"])
    data = simulate_experiments([load(name) for name in record["datasets"]], truth, spec, noise, seed)
    started = time.monotonic()
    revisions = []
    fitted = fit(data, spec, n_starts=config["fit_starts"], seed=seed, starting=truth, max_nfev=config["max_nfev"])
    covariance = uncertainty(fitted, data, bandwidths=config["hac_bandwidth_samples"], sigma_known=noise["sigma_C"])
    history = []
    likelihood_profile = None
    for retry in range(2):
        try:
            likelihood_profile = profile({"fit": fitted, "uncertainty": covariance}, data, 2,
                                         grid_points=config["profile_grid_points"],
                                         n_starts=config["profile_starts"]*(retry+1), seed=seed,
                                         max_nfev=config["max_nfev"]*(retry+1))
            if likelihood_profile["baseline_requires_revisit"]:
                if retry:
                    raise RuntimeError("Delay profile still improves the multistart minimum")
                revisions.append({"fit": fitted, "profile": likelihood_profile})
                candidate = min(likelihood_profile["points"], key=lambda p: p["sse"])["parameters"]
                fitted = fit(data, spec, n_starts=max(24, config["fit_starts"]), seed=seed,
                             starting=candidate, max_nfev=2*config["max_nfev"])
                covariance = uncertainty(fitted, data, bandwidths=config["hac_bandwidth_samples"], sigma_known=noise["sigma_C"])
                likelihood_profile = None
                continue
            break
        except Exception:
            history.append(traceback.format_exc())
            likelihood_profile = None
    if likelihood_profile is None:
        raise RuntimeError("Both delay-profile attempts failed: "+"\n".join(history))
    estimated = np.array([fitted["parameters"][name] for name in spec.names])
    covariances = {"iid_known_sigma": covariance["cov_log_known_sigma"],
                   "iid_estimated_sigma": covariance["cov_log_iid_unknown_sigma"]}
    covariances.update({"hac"+str(b): (covariance["cov_log_hac"] or {}).get(str(b)) for b in config["hac_bandwidth_samples"]})
    intervals = {name: local_intervals(estimated, truth, C) for name, C in covariances.items()}
    M = np.array([[-1., 0., 0.], [-1., 1., 0.], [0., 0., 1.]])
    physical_intervals = {name: local_intervals(to_physics(estimated), to_physics(truth),
                                               M@np.asarray(C)@M.T if C is not None else None) for name, C in covariances.items()}
    profile_intervals = {"known_sigma_iid": likelihood_profile["intervals_known_sigma_iid"],
                         "estimated_sigma_iid": likelihood_profile["intervals_unknown_sigma_iid"]}
    return {"identity_sha256": draw_identity, "status": "complete", "case": key, "noise_name": noise_name,
            "noise": noise, "replicate": replicate, "seed": seed, "truth_io": truth.tolist(),
            "truth_physics": to_physics(truth).tolist(), "datasets": record["datasets"],
            "estimated_io": estimated.tolist(), "estimated_physics": to_physics(estimated).tolist(),
            "fit": fitted, "uncertainty": covariance, "local_intervals_io": intervals,
            "local_intervals_physics": physical_intervals, "delay_profile": likelihood_profile,
            "delay_profile_coverage": {name: profile_interval_summary(value, truth[2]) for name, value in profile_intervals.items()},
            "profile_retry_errors": history, "revisited_minima": revisions,
            "elapsed_seconds": time.monotonic()-started,
            "notice": "Conditional synthetic coverage at one chosen true model point and the recorded inputs/times. Initial temperatures, ambient and heater calibration are exact. IID profiles are deliberately assessed under both IID and correlated true noise. This does not establish validity for real TC-Lab discrepancy."}


def execute(task):
    fit_path, noise_name, indices, config, output, draw_identity = task
    record = json.loads(Path(fit_path).read_text())
    folder = Path(output)/"draws"/(Path(fit_path).stem+"__"+noise_name)
    folder.mkdir(parents=True, exist_ok=True)
    counts = {"complete": 0, "failed": 0, "reused": 0}
    elapsed = []
    for replicate in indices:
        path = folder/(f"{replicate:04d}.json.gz")
        if path.exists():
            previous = json.loads(gzip.decompress(path.read_bytes()))
            if previous["identity_sha256"] != draw_identity:
                raise RuntimeError("Coverage draw identity changed: "+str(path))
            counts[previous["status"]] += 1
            counts["reused"] += 1
            elapsed.append(previous.get("elapsed_seconds", 0.))
            continue
        try:
            result = one_draw(record, config, noise_name, replicate, draw_identity)
        except Exception:
            result = {"identity_sha256": draw_identity, "status": "failed", "replicate": replicate,
                      "case": Path(fit_path).stem, "noise_name": noise_name, "error": traceback.format_exc(),
                      "notice": "Keep this failure in the Monte Carlo accounting; do not silently condition coverage on a successful optimizer."}
        save_draw(path, result)
        counts[result["status"]] += 1
        elapsed.append(result.get("elapsed_seconds", 0.))
    return counts, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", type=Path, default=HERE/"results/fits")
    parser.add_argument("--output", type=Path, default=HERE/"results/coverage")
    parser.add_argument("--config", type=Path, default=HERE/"coverage_config.json")
    parser.add_argument("--only", default="")
    parser.add_argument("--replicates", type=int, default=None, help="Explicit short-trial override")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    target = args.replicates or config["replicates"]
    identity = {"sources": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                            for name in ("models.py", "data.py", "inference.py", "profiles.py", "bootstrap.py", "coverage_study.py", "run_coverage.py", "run_fits.py")},
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "replicate_override": args.replicates}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks = []
        input_hashes = {}
        for case in config["cases"]:
            path = args.fits/(case+".json")
            input_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for noise_name in config["noise_scenarios"]:
                if args.only not in case+"__"+noise_name:
                    continue
                draw_identity = hashlib.sha256(json.dumps({"identity": identity, "fit_sha256": input_hashes[path.name],
                                                          "noise_name": noise_name, "target": target}, sort_keys=True).encode()).hexdigest()
                for first in range(0, target, config["chunk_size"]):
                    indices = list(range(first, min(first+config["chunk_size"], target)))
                    key = case+"__"+noise_name+f"__{indices[0]:04d}_{indices[-1]:04d}"
                    tasks.append((key, (str(path), noise_name, indices, config, str(args.output), draw_identity)))
        if not tasks:
            raise RuntimeError("No coverage scenarios selected")
        old = args.output/"manifest.json"
        if old.exists():
            previous = json.loads(old.read_text())
            if previous["identity"] != identity or previous["input_fit_sha256"] != input_hashes:
                raise RuntimeError("Coverage source/configuration/input changed; preserve earlier phase")
        manifest = {"identity": identity, "input_fit_sha256": input_hashes,
                    "started_utc": datetime.now(timezone.utc).isoformat(), "expected_chunks": len(tasks),
                    "expected_draws": sum(len(task[1][2]) for task in tasks), "completed": {}, "failed": {},
                    "status": "running", "filter": args.only, "target_replicates_per_scenario": target,
                    "n1_profile_policy": "Truth uses the baseline canonical pole representative; the full unconstrained pole-label profile union is retained. Union-component and envelope widths are separate quantities.",
                    "coverage_notice": "Report method-specific defined intervals, optimizer failures, unconditional failure bounds, binomial uncertainty and the exact simulated true models/noise. Do not generalize to all parameter points or real model mismatch."}
        save_json(args.output/"manifest.json", manifest)
        with ProcessPoolExecutor(max_workers=min(config["workers"], 4)) as pool:
            futures = {pool.submit(execute, task): key for key, task in tasks}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    counts, elapsed = future.result()
                    manifest["completed"][key] = {**counts, "elapsed_seconds_per_draw": elapsed}
                    print(f"{len(manifest['completed'])}/{len(tasks)} {key} complete={counts['complete']} failed={counts['failed']}", flush=True)
                except Exception:
                    manifest["failed"][key] = traceback.format_exc()
                    print("FAILED CHUNK "+key, flush=True)
                save_json(args.output/"manifest.json", manifest)
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "complete" if not manifest["failed"] and not any(r["failed"] for r in manifest["completed"].values()) else "complete_with_failures"
        save_json(args.output/"manifest.json", manifest)
        return manifest["status"] != "complete"


if __name__ == "__main__":
    raise SystemExit(main())
