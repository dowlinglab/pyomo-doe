"""Resumable profile likelihood jobs, one parameter per worker task."""
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

from data import load
from inference import ModelSpec
from profiles import profile
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def execute(task):
    path, parameter_index, config, output, identity, original_recovery = task
    record = json.loads(Path(path).read_text())
    spec = ModelSpec(**record["fit"]["spec"])
    key = Path(path).stem+"__"+spec.names[parameter_index]
    destination = Path(output)/(key+".json")
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing["identity"] != identity or existing["fit_sha256"] != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise RuntimeError("Profile source/configuration changed: "+key)
        return key, "reused", existing["profile"]["baseline_requires_revisit"]
    datasets = [load(name) for name in record["datasets"]]
    result = profile(record, datasets, parameter_index, config["profile_grid_points"], config["profile_starts"],
                     config["seed"], config["max_nfev"], recover_original=original_recovery)
    save_json(destination, {"identity": identity, "fit_file": Path(path).name,
                            "fit_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "profile": result})
    return key, "complete", result["baseline_requires_revisit"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", type=Path, default=HERE/"results/fits")
    parser.add_argument("--output", type=Path, default=HERE/"results/profiles")
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    parser.add_argument("--only", default="")
    parser.add_argument("--original-recovery", action="store_true", help="Retry only failed original nuisance fits with scaled complex-step Jacobians and ten times the evaluation budget")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    identity = {"sources": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ("models.py", "data.py", "inference.py", "profiles.py", "run_profiles.py", "run_fits.py")},
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "original_recovery": args.original_recovery}
    with (args.output/"run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks = []
        for path in sorted(args.fits.glob("*.json")):
            record = json.loads(path.read_text())
            spec = ModelSpec(**record["fit"]["spec"])
            for j, name in enumerate(spec.names):
                if args.only in path.stem+"__"+name:
                    tasks.append((str(path), j, config, str(args.output), identity, args.original_recovery))
        manifest = {"identity": identity, "started_utc": datetime.now(timezone.utc).isoformat(),
                    "expected_jobs": len(tasks), "completed": {}, "failed": {}, "status": "running", "filter": args.only}
        save_json(args.output/"manifest.json", manifest)
        with ProcessPoolExecutor(max_workers=min(config["workers"], 4)) as pool:
            futures = {pool.submit(execute, task): Path(task[0]).stem+"__"+str(task[1]) for task in tasks}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    key, status, revisit = future.result()
                    manifest["completed"][key] = {"status": status, "baseline_requires_revisit": revisit}
                    print(f"{len(manifest['completed'])}/{len(tasks)} {key}", flush=True)
                except Exception:
                    manifest["failed"][name] = traceback.format_exc()
                    print("FAILED "+name, flush=True)
                save_json(args.output/"manifest.json", manifest)
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "complete" if not manifest["failed"] else "complete_with_failures"
        save_json(args.output/"manifest.json", manifest)
        return bool(manifest["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
