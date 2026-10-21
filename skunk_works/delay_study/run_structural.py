"""Numerically check the algebraic equivalences and nonregular pole example.

These are deterministic model calculations, not additional fitted datasets.
Physical parameter ranges on an equivalence class are not confidence intervals.
"""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from data import load
from inference import ModelSpec, parameter_box, predict, sensitivities
from models import original_transfer, equivalent_original
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    manifest = {"source_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                  for name in ("models.py", "inference.py", "data.py", "run_structural.py", "run_fits.py")},
                "input_fit_sha256": {}, "status": "running",
                "notice": "Deterministic model implications conditional on ambient and initial states. Sampled physical ranges are equivalence-class ranges, not confidence intervals."}
    for calibration in ("sine", "step", "joint"):
        path = args.fits/(calibration+"__original_n0_physics_cold_linear_exact.json")
        manifest["input_fit_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    old_path = args.output/"manifest.json"
    if old_path.exists():
        previous = json.loads(old_path.read_text())
        if any(previous[key] != manifest[key] for key in ("source_sha256", "input_fit_sha256")):
            raise RuntimeError("Structural result source/input identity changed; use a new output folder")
        if previous["status"] == "complete":
            for name, sha in previous["output_sha256"].items():
                if hashlib.sha256((args.output/name).read_bytes()).hexdigest() != sha:
                    raise RuntimeError("Structural output changed: "+name)
            print("Reused verified structural outputs", flush=True)
            return
    save_json(args.output/"manifest.json", manifest)
    family_rows, families = [], []
    for calibration in ("sine", "step", "joint"):
        path = args.fits/(calibration+"__original_n0_physics_cold_linear_exact.json")
        record = json.loads(path.read_text())
        spec = ModelSpec(**record["fit"]["spec"])
        p = np.array([record["fit"]["parameters"][name] for name in spec.names])
        K, a, b = original_transfer(p)
        fast, slow = sorted(np.roots([1., -a, b]).real)
        if a*a <= 4*b or fast <= 0:
            raise RuntimeError("Expected distinct positive original thermal poles")
        lower, upper = parameter_box(spec)
        # Include analytic extrema of CpS and Ub in addition to a dense r grid.
        grid = np.unique(np.r_[np.geomspace(fast*(1+1e-7), slow*(1-1e-7), 401), np.sqrt(b), 2*b/a])
        data = [load(name) for name in record["datasets"]]
        reference = np.concatenate([predict(d, p, spec) for d in data])
        J = sensitivities(p, data, spec)
        r0 = p[3]/p[1]
        q = (-r0+b/r0)/(a-r0-b/r0)
        tangent = np.array([0., q-1, -1., q])
        tangent /= np.linalg.norm(tangent)
        rows = []
        max_error = 0.
        for r in grid:
            member = equivalent_original(K, a, b, r)
            feasible = bool(np.all(member >= lower) and np.all(member <= upper))
            output = np.concatenate([predict(d, member, spec) for d in data])
            error = float(np.max(abs(output-reference)))
            max_error = max(max_error, error)
            row = {"calibration": calibration, "r_s": float(r), "within_original_fit_box": feasible,
                   **dict(zip(spec.names, member.tolist())), "maximum_prediction_difference_C": error}
            rows.append(row)
        family_rows.extend(rows)
        feasible_rows = [row for row in rows if row["within_original_fit_box"]]
        families.append({"calibration": calibration, "K": float(K), "a_s": float(a), "b_s2": float(b),
                         "fast_pole_s": float(fast), "slow_pole_s": float(slow),
                         "sampled_members": len(rows), "sampled_members_in_fit_box": len(feasible_rows),
                         "maximum_prediction_difference_C": max_error,
                         "unit_log_null_tangent": tangent.tolist(),
                         "maximum_J_times_unit_null_tangent_C": float(np.max(abs(J@tangent))),
                         "sampled_ranges_in_fit_box": {name: [float(min(row[name] for row in feasible_rows)),
                                                                  float(max(row[name] for row in feasible_rows))] for name in spec.names},
                         "analytic_positive_family_ranges": {"CpH": [float(p[0]*fast), float(p[0]*slow)],
                                                              "CpS": [0., float(p[0]*(a-2*np.sqrt(b)))],
                                                              "Ub": [0., float(p[0]*(a*a/(4*b)-1))]},
                         "endpoint_notice": "CpH endpoints and lower CpS/Ub endpoints are open in the strictly positive family. Rectangular fit bounds further restrict these ranges."})
    write_csv(args.output/"equivalence_curves.csv", family_rows)
    save_json(args.output/"families.json", families)

    # Keep impulse-time mean fixed while perturbing a coincident pole.
    source = load("sine")
    repeated_rows = []
    for order in (1, 2, 3, 6, 12):
        spec = ModelSpec(order=order)
        center = np.array([.78, 10., 10.*order])
        reference = predict(source, center, spec)
        for delta in np.r_[-np.geomspace(1e-3, 3., 35)[::-1], 0., np.geomspace(1e-3, 3., 35)]:
            p = center+np.array([0., delta, -delta])
            difference = predict(source, p, spec)-reference
            repeated_rows.append({"order": order, "delta_s": float(delta), "tau_s": float(p[1]),
                                  "theta_s": float(p[2]), "fixed_mean_s": float(p[1]+p[2]),
                                  "squared_prediction_distance_C2": float(difference[1:]@difference[1:]),
                                  "maximum_prediction_difference_C": float(np.max(abs(difference)))})
    write_csv(args.output/"repeated_poles.csv", repeated_rows)

    # A common parameter point isolates chain approximation from refitting.
    parameters = np.array([.78, 150., 18.])
    limit = predict(source, parameters, ModelSpec(family="pure_delay", order=0))
    limit_rows = []
    for order in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128):
        output = predict(source, parameters, ModelSpec(order=order))
        difference = output-limit
        limit_rows.append({"order": order, "delay_kernel_mean_s": float(parameters[2]),
                           "delay_kernel_variance_s2": float(parameters[2]**2/order),
                           "rmse_from_pure_delay_C": float(np.sqrt(np.mean(difference[1:]**2))),
                           "maximum_difference_from_pure_delay_C": float(np.max(abs(difference)))})
    write_csv(args.output/"chain_limit.csv", limit_rows)
    manifest["output_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(args.output.iterdir()) if p.suffix in {".json", ".csv"} and p.name not in {"manifest.json", "source_snapshot.json"}}
    manifest["status"] = "complete"
    save_json(args.output/"manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", type=Path, default=HERE/"results/fits")
    parser.add_argument("--output", type=Path, default=HERE/"results/structural")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        analyze(args)


if __name__ == "__main__":
    main()
