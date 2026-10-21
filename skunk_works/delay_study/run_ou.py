"""Resumable OU residual-model calibration; raw mean predictions remain auditable."""
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
import pandas as pd

from data import load
from inference import ModelSpec, predict
from models import to_physics
from noise import whiten_ou
from ou_inference import fit_ou, local_uncertainty
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def execute(task):
    input_path, config, output, identity, starts_override = task
    path = Path(input_path)
    record = json.loads(path.read_text())
    spec = ModelSpec(**record["fit"]["spec"])
    destination = Path(output)/(path.stem+".json")
    fit_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["identity"] != identity or saved["fit_sha256"] != fit_sha:
            raise RuntimeError("OU result source/input identity changed: "+path.stem)
        prediction = Path(output)/saved["prediction_file"]
        if hashlib.sha256(prediction.read_bytes()).hexdigest() != saved["prediction_sha256"]:
            raise RuntimeError("OU prediction bytes changed")
        return path.stem, "reused", saved["fit"]["requires_revisit"]
    datasets = [load(name) for name in record["datasets"]]
    initial = np.array([record["fit"]["parameters"][name] for name in spec.names])
    result = fit_ou(datasets, spec, initial, n_starts=starts_override or config["n_starts"],
                    seed=config["seed"], maxiter=config["max_nfev"])
    parameters = np.array([result["parameters"][name] for name in spec.names])
    covariance = local_uncertainty(result, datasets, spec)
    predictions, validation = [], []
    for role, experiments in [("calibration", datasets), ("validation", [load(row["dataset"]) for row in record["validation"]])]:
        for data in experiments:
            predicted = predict(data, parameters, spec)
            residual = (predicted-data.temperature)[1:]
            innovation, _ = whiten_ou(residual, data.time[1:], result["correlation_length_s"])
            predictions.append(pd.DataFrame({"role": role, "experiment": data.name, "time_s": data.time,
                                             "measured_C": data.temperature, "predicted_C": predicted,
                                             "input_percent": data.input,
                                             "innovation_C": np.r_[np.nan, innovation],
                                             "standardized_innovation": np.r_[np.nan, innovation/result["sigma_stationary_ML_C"]]}))
            if role == "validation":
                validation.append({"dataset": data.name, "n": len(residual),
                                   "rmse_C": float(np.sqrt(np.mean(residual**2))),
                                   "bias_C": float(np.mean(residual)),
                                   "max_abs_error_C": float(np.max(abs(residual)))})
    prediction_path = Path(output)/"predictions"/(path.stem+".csv.gz")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = prediction_path.with_name(prediction_path.stem+".tmp.csv.gz")
    pd.concat(predictions, ignore_index=True).to_csv(temporary, index=False, compression={"method": "gzip", "mtime": 0})
    temporary.replace(prediction_path)
    n = result["n_conditional"]
    nominal_parameters = 5  # K,tau,theta, OU length, stationary variance.
    nll = result["negative_log_likelihood"]
    iid_nll = .5*n*(np.log(2*np.pi)+1+np.log(record["fit"]["sse"]/n))
    result.update(physics_parameters=dict(zip(["Ua", "Cp", "theta"], to_physics(parameters, power=datasets[0].power).tolist())),
                  rmse_raw_C=float(np.sqrt(result["raw_sse_C2"]/n)),
                  aic_nominal=float(2*nll+2*nominal_parameters),
                  aicc_nominal=float(2*nll+2*nominal_parameters+2*nominal_parameters*(nominal_parameters+1)/(n-nominal_parameters-1)),
                  bic_nominal=float(2*nll+nominal_parameters*np.log(n)),
                  iid_baseline_nll=float(iid_nll), twice_nll_improvement_over_iid=float(2*(iid_nll-nll)),
                  information_criterion_notice="Nominal parameter count includes the three mean coordinates, length and variance. Rank loss, active bounds, prehistory assumptions, model error and the IID boundary limit prevent routine regular model-selection/test interpretations.")
    save_json(destination, {"identity": identity, "fit_file": path.name, "fit_sha256": fit_sha,
                            "calibration": record["calibration"], "datasets": record["datasets"], "spec": asdict(spec),
                            "starts_override": starts_override, "fit": result, "uncertainty": covariance,
                            "validation": validation, "prediction_file": str(prediction_path.relative_to(Path(output))),
                            "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                            "prediction_notice": "Thermal mean prediction; first-sample innovation undefined because that temperature conditions the initial state. Noise covariance changes inference, not the definition of the thermal output."})
    return path.stem, "complete", result["requires_revisit"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", type=Path, default=HERE/"results/fits")
    parser.add_argument("--output", type=Path, default=HERE/"results/ou")
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    parser.add_argument("--only", default="")
    parser.add_argument("--starts", type=int, default=None, help="Explicit short-trial override")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"sources": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                            for name in ("models.py", "data.py", "inference.py", "noise.py", "ou_inference.py", "run_ou.py", "run_fits.py")},
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "starts_override": args.starts}
    with (args.output/"run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        tasks = []
        for path in sorted(args.fits.glob("*.json")):
            record = json.loads(path.read_text())
            spec = ModelSpec(**record["fit"]["spec"])
            if spec.coordinates == "io" and spec.family != "original" and args.only in path.stem:
                tasks.append((str(path), config, str(args.output), identity, args.starts))
        if not tasks:
            raise RuntimeError("No OU calibration cases selected")
        manifest = {"identity": identity, "expected_jobs": len(tasks), "started_utc": datetime.now(timezone.utc).isoformat(),
                    "completed": {}, "failed": {}, "status": "running", "filter": args.only,
                    "formulation_policy": "IO mean fitting with exact physical-coordinate transformation. The original model's zero-state identifiable transfer is represented by the cold one-stage case, subject to its distinct-pole and physical-box conditions."}
        save_json(args.output/"manifest.json", manifest)
        with ProcessPoolExecutor(max_workers=min(config["workers"], 4)) as pool:
            futures = {pool.submit(execute, task): Path(task[0]).stem for task in tasks}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    _, status, revisit = future.result()
                    manifest["completed"][key] = {"status": status, "requires_revisit": revisit}
                    print(f"{len(manifest['completed'])}/{len(tasks)} {key}", flush=True)
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
