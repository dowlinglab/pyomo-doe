"""Mean and future-observation intervals from saved fits and bootstrap draws.

No model refits. Pointwise intervals are conditional on the recorded input,
initial temperature, ambient, heater calibration, chosen model and error law.
"""
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
import pandas as pd

from data import load
from inference import ModelSpec, parameter_box, predict, sensitivities
from noise import whiten_ou
from prediction_uncertainty import (estimable_variance, normal_bands, local_covariances,
                                     gaussian_mixture_quantiles, constant_equilibrated_target)
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tasks(fits, config):
    jobs = []
    for path in sorted(Path(fits).glob("*.json")):
        record = json.loads(path.read_text())
        spec = ModelSpec(**record["fit"]["spec"])
        if spec.coordinates != "io" and spec.family != "original":
            continue
        jobs.append(("local__"+path.stem, "local", path.name, None))
        if spec.family != "original":
            jobs.append(("ou__"+path.stem, "ou", path.name, None))
            if record["calibration"] in config["bootstrap_calibrations"]:
                jobs.extend(("bootstrap__"+path.stem+"__"+method, "bootstrap", path.name, method)
                            for method in config["bootstrap_methods"])
    return jobs


def frame_for(data, role, method, mean, variance, bands, defined):
    frame = pd.DataFrame({"role": role, "experiment": data.name, "method": method, "time_s": data.time,
                          "input_percent": data.input, "measured_C": data.temperature, "mean_C": mean,
                          "mean_variance_C2": np.r_[np.nan, variance], "estimable": np.r_[False, defined],
                          "first_temperature_conditions_state": np.r_[True, np.zeros(len(data.time)-1, dtype=bool)]})
    for name, values in bands.items():
        frame[name+"_C"] = np.r_[np.nan, values]
    return frame


def descriptive_summary(frame):
    defined = frame.estimable.to_numpy() & ~frame.first_temperature_conditions_state.to_numpy()
    mean_width = (frame.mean_upper_C-frame.mean_lower_C).to_numpy()
    result = {"role": frame.role.iloc[0], "experiment": frame.experiment.iloc[0], "method": frame.method.iloc[0],
              "conditional_points": len(frame)-1, "estimable_points": int(defined.sum()),
              "mean_band_median_width_C": float(np.nanmedian(mean_width)) if defined.any() else None,
              "mean_band_maximum_width_C": float(np.nanmax(mean_width)) if defined.any() else None}
    if "prediction_lower_C" in frame:
        inside = (frame.measured_C >= frame.prediction_lower_C) & (frame.measured_C <= frame.prediction_upper_C)
        result["observed_fraction_inside_prediction_band_defined_points"] = float(inside[defined].mean()) if defined.any() else None
        result["coverage_notice"] = "Descriptive pointwise containment on this record, conditional on defined intervals. Calibration is optimistic; validation records do not establish independent repeated-experiment coverage."
    return result


def execute(job, config, fits, ou, bootstrap, output, identity):
    key, phase, filename, method = job
    output = Path(output)
    path, input_path = output/(key+".json"), Path(fits)/filename
    record = json.loads(input_path.read_text())
    inputs = {str(input_path): sha(input_path)}
    spec = ModelSpec(**record["fit"]["spec"])
    parameter_record = record["fit"]["parameters"]
    matrices, null, information_record = None, None, None
    maximum_mixture_error = 0.
    if phase == "ou":
        ou_path = Path(ou)/filename
        source = json.loads(ou_path.read_text())
        if source["fit_sha256"] != sha(input_path):
            raise RuntimeError("OU baseline fit input changed")
        inputs[str(ou_path)] = sha(ou_path)
        parameter_record = source["fit"]["parameters"]
    elif phase == "bootstrap":
        source_path = Path(bootstrap)/(input_path.stem+"__"+method+".json")
        source = json.loads(source_path.read_text())
        array_path = source_path.with_suffix(".npz")
        if source["fit_sha256"] != sha(input_path) or source["arrays_sha256"] != sha(array_path):
            raise RuntimeError("Bootstrap inputs changed")
        inputs.update({str(source_path): sha(source_path), str(array_path): sha(array_path)})
        saved = np.load(array_path)
        if not saved["completed"].all() or not saved["success"].all():
            raise RuntimeError("Bootstrap completion must be accounted before prediction")
        parameter_draws = saved["parameters"]
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["identity"] != identity or previous["input_sha256"] != inputs:
            raise RuntimeError("Preserve prediction results from earlier inputs/sources")
        if sha(output/previous["bands_file"]) != previous["bands_sha256"]:
            raise RuntimeError("Saved prediction interval bytes changed")
        return "reused"
    parameters = np.array([parameter_record[name] for name in spec.names])
    calibration = [load(name) for name in record["datasets"]]
    if phase == "local":
        J = sensitivities(parameters, calibration, spec)
        residual = np.concatenate([(predict(d, parameters, spec)-d.temperature)[1:] for d in calibration])
        if abs(residual@residual-record["fit"]["sse"]) > 1e-7:
            raise RuntimeError("Baseline mean changed since the completed fit")
        matrices, null, information_record = local_covariances(J, residual, [len(d.time)-1 for d in calibration],
                                                               config["known_sigma_C"], config["hac_bandwidth_samples"])
    elif phase == "ou":
        transformed = []
        for d in calibration:
            J, _ = whiten_ou(sensitivities(parameters, [d], spec), d.time[1:], source["fit"]["correlation_length_s"])
            transformed.append(J/source["fit"]["sigma_stationary_ML_C"])
        _, singular, Vh = np.linalg.svd(np.vstack(transformed), full_matrices=False)
        keep = singular > 1e-8*singular[0]
        null = Vh[~keep].T
        matrices = {"ou_conditional_noise": (Vh[keep].T/singular[keep]**2)@Vh[keep]}
        C = source["uncertainty"]["observed_cov_log_mean_and_length"]
        if C is not None:
            matrices["ou_observed_noise_profiled"] = np.asarray(C)[:3, :3]
        information_record = {"rank": int(keep.sum()), "observed_hessian_covariance_available": C is not None,
                              "active_bounds_log": source["fit"]["active_bounds_log"],
                              "hessian_relative_step_disagreement": source["uncertainty"]["hessian_relative_step_disagreement"]}
    frames, summaries = [], []
    experiments = [("calibration", d) for d in calibration]
    experiments.extend(("validation", load(row["dataset"])) for row in record["validation"])
    for role, data in experiments:
        mean = predict(data, parameters, spec)
        if phase == "bootstrap":
            means = np.stack([predict(data, draw, spec)[1:] for draw in parameter_draws])
            quantiles = np.quantile(means, [.025, .5, .975], axis=0)
            variance = np.var(means, axis=0, ddof=1)
            bands = {"mean_lower": quantiles[0], "mean_upper": quantiles[2], "bootstrap_mean_median": quantiles[1]}
            if method == "parametric":
                predicted, error = gaussian_mixture_quantiles(means, config["known_sigma_C"], steps=config["mixture_bisection_steps"])
                bands.update(prediction_lower=predicted[0], prediction_upper=predicted[1])
                maximum_mixture_error = max(maximum_mixture_error, error)
            frame = frame_for(data, role, "bootstrap_"+method, mean, variance, bands, np.ones(len(mean)-1, dtype=bool))
            frames.append(frame)
            summaries.append(descriptive_summary(frame))
        else:
            target_J = sensitivities(parameters, [data], spec)
            envelope = None
            if phase == "local" and spec.coordinates == "io" and "theta" in record["uncertainty"]["nonestimable_native_parameters"]:
                lower, upper = parameter_box(spec)
                grid = np.unique(np.r_[np.geomspace(lower[2], upper[2], config["null_delay_envelope_points"]), parameters[2]])
                probes = np.stack([predict(data, np.r_[parameters[:2], theta], spec) for theta in grid])
                envelope = np.min(probes, axis=0), np.max(probes, axis=0)
            for name, covariance in matrices.items():
                variance, defined = estimable_variance(target_J, covariance, null)
                if ("theta" in record["uncertainty"]["nonestimable_native_parameters"]
                        and not constant_equilibrated_target(data, spec)):
                    # A short pure-delay target can have zero local theta
                    # derivative while a distant equivalent theta changes it.
                    # Local annihilation alone does not resolve a global mode.
                    defined[:] = False
                    variance[:] = np.nan
                future_variance = None
                if name == "iid_known_sigma":
                    future_variance = config["known_sigma_C"]**2
                elif name == "iid_unknown_quotient_dof":
                    future_variance = information_record["estimated_future_iid_variance_C2"]
                elif phase == "ou":
                    future_variance = source["fit"]["sigma_stationary_ML_C"]**2
                bands = normal_bands(mean[1:], variance, future_variance)
                frame = frame_for(data, role, name, mean, variance, bands, defined)
                if envelope is not None and name == "iid_known_sigma":
                    frame["theta_exploration_lower_C"], frame["theta_exploration_upper_C"] = envelope
                frames.append(frame)
                summaries.append(descriptive_summary(frame))
    prediction_path = output/"bands"/(key+".csv.gz")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = prediction_path.with_name(prediction_path.stem+".tmp.csv.gz")
    pd.concat(frames, ignore_index=True).to_csv(temporary, index=False, compression={"method": "gzip", "mtime": 0})
    temporary.replace(prediction_path)
    save_json(path, {"identity": identity, "input_sha256": inputs, "phase": phase, "case": input_path.stem,
                     "spec": record["fit"]["spec"], "method": method, "information": information_record,
                     "bands_file": str(prediction_path.relative_to(output)), "bands_sha256": sha(prediction_path),
                     "summaries": summaries, "maximum_gaussian_mixture_cdf_residual": maximum_mixture_error,
                     "working_known_sigma_C": config["known_sigma_C"],
                     "null_target_relative_tolerance": 1e-8,
                     "global_delay_policy": "If calibration leaves theta structurally unobservable, all bands for a varying-input target are withheld, even where the local derivative vanishes. Constant equilibrated targets are algebraically theta-independent. A finite exploration envelope is reported separately without confidence meaning.",
                     "notice": "Pointwise conditional bands, not simultaneous bands or model-discrepancy uncertainty. The first temperature conditions the state and has no reported interval. A pseudoinverse is used only for target sensitivities annihilating the calibration null space. HAC supplies a mean-estimator covariance and no future error law; its observation prediction bands are withheld. OU future-error scale/length are working fitted quantities, not established sensor noise.",
                     "bootstrap_notice": "Joint IO fits only, 400 saved draws for each selected method. Mean bands use percentile trajectories. Gaussian-parametric prediction quantiles invert the empirical equal-weight mean mixture plus fresh independent Gaussian error at the stated working_known_sigma_C. Block60 receives mean bands only. No arbitrary original physical gauge member is bootstrapped; the zero-state original transfer is represented by its cold n1 quotient.",
                     "null_envelope_notice": "Where baseline delay is unidentifiable, optional theta exploration extrema vary only theta over the numerical box with fitted K/tau fixed. This finite grid is not a confidence interval, a global bound or a probability distribution."})
    return "complete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", type=Path, default=HERE/"results/fits")
    parser.add_argument("--ou", type=Path, default=HERE/"results/ou")
    parser.add_argument("--bootstrap", type=Path, default=HERE/"results/bootstrap")
    parser.add_argument("--output", type=Path, default=HERE/"results/prediction_uncertainty")
    parser.add_argument("--config", type=Path, default=HERE/"prediction_config.json")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    jobs = [job for job in tasks(args.fits, config) if args.only in job[0]]
    if not jobs:
        raise RuntimeError("No prediction jobs selected")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"source_sha256": {name: sha(HERE/name) for name in ("data.py", "models.py", "inference.py", "noise.py", "prediction_uncertainty.py", "run_prediction_uncertainty.py", "run_fits.py")},
                "config_sha256": sha(args.config)}
    with (args.output/"run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/"manifest.json").exists():
            previous = json.loads((args.output/"manifest.json").read_text())
            if previous["identity"] != identity:
                raise RuntimeError("Preserve prediction phase from earlier source/configuration")
        manifest = {"identity": identity, "status": "running", "expected_jobs": len(jobs), "filter": args.only,
                    "started_utc": datetime.now(timezone.utc).isoformat(), "completed": {}, "failed": {}}
        save_json(args.output/"manifest.json", manifest)
        with ProcessPoolExecutor(max_workers=min(config["workers"], 4)) as pool:
            futures = {pool.submit(execute, job, config, str(args.fits), str(args.ou), str(args.bootstrap), str(args.output), identity): job[0] for job in jobs}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    manifest["completed"][key] = {"status": future.result()}
                    print(f"{len(manifest['completed'])}/{len(jobs)} {key}", flush=True)
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
