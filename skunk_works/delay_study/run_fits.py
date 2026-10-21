"""Resumable, bounded calibration matrix; writes data only, never plots."""
from __future__ import annotations

import os
for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[thread_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

from data import load, audit
from inference import fit, uncertainty, predict, ModelSpec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def source_identity(config):
    paths = [HERE/name for name in ("data.py", "models.py", "inference.py", "run_fits.py", "config.json")]
    relative = [str(p.relative_to(ROOT)) for p in paths]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"]+relative, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}


def cases(config):
    for calibration, names in config["calibration_sets"].items():
        for family in config["families"]:
            if family == "original":
                yield calibration, names, ModelSpec(family=family, order=0, coordinates="physics")
                continue
            orders = config["orders"] if family == "chain" else [0]
            for order in orders:
                for initialization in config["initializations"]:
                    for coordinates in config["coordinates"]:
                        yield calibration, names, ModelSpec(family=family, order=order, coordinates=coordinates, initialization=initialization)


def run_case(task):
    calibration, names, spec_dict, config, destination, identity, validation_names = task
    spec = ModelSpec(**spec_dict)
    key = calibration+"__"+spec.identifier
    path = Path(destination)/"fits"/(key+".json")
    if path.exists():
        old = json.loads(path.read_text())
        if old["identity"]["source_sha256"] != identity["source_sha256"] or old["identity"]["config_sha256"] != identity["config_sha256"]:
            raise RuntimeError("Existing result has different source/configuration: "+key)
        return key, "reused", old["fit"]["rmse"]
    datasets = [load(name) for name in names]
    result = fit(datasets, spec, config["n_starts"], config["seed"], max_nfev=config["max_nfev"])
    u = uncertainty(result, datasets, config["hac_bandwidth_samples"], config["known_sigma_C"])
    parameters = [result["parameters"][name] for name in spec.names]
    predictions = []
    validation = []
    for role, experiments in (("calibration", datasets), ("validation", [load(n) for n in validation_names])):
        for data in experiments:
            output = predict(data, parameters, spec)
            errors = output[1:]-data.temperature[1:]
            if role == "validation":
                validation.append({"dataset": data.name, "n": len(errors), "rmse_C": float(np.sqrt(np.mean(errors**2))),
                                   "bias_C": float(np.mean(errors)), "max_abs_error_C": float(np.max(np.abs(errors)))})
            predictions.append(pd.DataFrame({"role": role, "experiment": data.name, "time_s": data.time,
                                             "measured_C": data.temperature, "predicted_C": output, "input_percent": data.input}))
    prediction_path = Path(destination)/"predictions"/(key+".csv.gz")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_csv(prediction_path, index=False, compression={"method": "gzip", "mtime": 0})
    save_json(path, {"identity": identity, "calibration": calibration, "datasets": names, "fit": result,
                     "uncertainty": u, "validation": validation,
                     "prediction_file": str(prediction_path.relative_to(Path(destination))),
                     "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()})
    return key, "complete", result["rmse"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    parser.add_argument("--output", type=Path, default=HERE/"results")
    parser.add_argument("--only", default="", help="Substring filter for a bounded trial; manifest records it")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity = source_identity(config)
        inventory = audit()
        calibration_hashes = {row["numeric_sha256"] for row in inventory["files"] if Path(row["file"]).name.startswith("tclab_")}
        validation_names = [Path(row["file"]).name for row in inventory["files"] if Path(row["file"]).name.startswith("validation_") and row["numeric_sha256"] not in calibration_hashes]
        selected = [(label, names, spec) for label, names, spec in cases(config) if args.only in label+"__"+spec.identifier]
        manifest = {"identity": identity, "started_utc": datetime.now(timezone.utc).isoformat(),
                    "command": sys.argv, "config": config, "filter": args.only,
                    "expected_cases": len(selected), "completed": {}, "failed": {},
                    "validation_files": validation_names, "data_audit": inventory,
                    "status": "running", "scope_notice": "Initial thermal reading conditioned upon; unweighted SSE; validation excludes calibration duplicates"}
        save_json(args.output/"manifest.json", manifest)
        tasks = [(label, names, asdict(spec), config, str(args.output), identity, validation_names) for label, names, spec in selected]
        with ProcessPoolExecutor(max_workers=min(config["workers"], 4)) as pool:
            futures = {pool.submit(run_case, task): task[0]+"__"+ModelSpec(**task[2]).identifier for task in tasks}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    _, status, rmse = future.result()
                    manifest["completed"][key] = {"status": status, "rmse_C": rmse}
                    print(f"{len(manifest['completed'])}/{len(selected)} {key} RMSE={rmse:.6g}", flush=True)
                except Exception:
                    manifest["failed"][key] = traceback.format_exc()
                    print("FAILED "+key+"\n"+manifest["failed"][key], flush=True)
                save_json(args.output/"manifest.json", manifest)
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "complete" if not manifest["failed"] else "complete_with_failures"
        save_json(args.output/"manifest.json", manifest)
        return bool(manifest["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
