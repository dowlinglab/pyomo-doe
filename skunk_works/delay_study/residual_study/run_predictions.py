"""Generate ten retrospective out-of-calibration prediction comparisons per pair."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np

from audit_mean import reference
from data import load
from prediction import bands, coverage, empirical_innovations, future_mean, conditional_ambient_reconstruction, quantile_preflight
from run_mean import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--gaussian", type=Path, required=True)
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-draws", type=int, default=360)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fit_path = args.reference/"fit.json"
    fit = json.loads(fit_path.read_text())
    split = json.loads((args.reference/"split.json").read_text())
    ordinary = json.loads((args.reference/"ols.json").read_text())
    fit_sha = hashlib.sha256(fit_path.read_bytes()).hexdigest()
    checkpoints = {"gaussian": args.gaussian, "innovation_blocks": args.blocks}
    usable = {}
    binding = {"reference_fit_sha256": fit_sha, "ordinary_sha256": hashlib.sha256((args.reference/"ols.json").read_bytes()).hexdigest()}
    counts = {}
    for mode, directory in checkpoints.items():
        path = directory/"checkpoint.json"
        checkpoint = json.loads(path.read_text())
        if checkpoint["status"] != "complete" or checkpoint["identity"]["reference_fit_sha256"] != fit_sha or checkpoint["identity"]["measurement"] != mode:
            raise ValueError("Prediction bootstrap does not match the terminal reference fit")
        rows = [d for d in checkpoint["draws"] if d["status"] == "complete"]
        if len(rows) < args.minimum_draws:
            raise RuntimeError("Too few usable bootstrap refits for declared prediction analysis")
        usable[mode] = rows
        counts[mode] = {"attempted": len(checkpoint["draws"]), "usable": len(rows),
                        "failed_or_flagged": len(checkpoint["draws"])-len(rows),
                        "block_length": checkpoint["identity"].get("block_length", 60)}
        binding[mode+"_checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(split["evaluation"]) != 10 or set(split["evaluation"]) & set(split["training"]):
        raise AssertionError("Prediction split is not the declared ten-record evaluation")
    training = [load(n) for n in split["training"]]
    empirical = empirical_innovations(training, fit)
    write_json(args.output/"quantile_preflight.json", quantile_preflight())
    source = json.loads((args.reference/"source_snapshot.json").read_text())
    summaries = []
    numeric = {}
    audits = []
    for j, name in enumerate(split["evaluation"]):
        data = load(name)
        path = HERE.parents[2]/data.source
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"][data.source]:
            raise AssertionError("Evaluation raw-data source changed")
        point = future_mean(data, fit, fit["best"])
        independently = reference(data, fit["best"]["coefficients"][0], fit["best"]["tau"], fit["best"]["theta"], [0., 0.], fit["hold"])
        discrepancy = float(np.max(np.abs(point-independently)))
        if discrepancy > 1e-7:
            raise AssertionError("Prospective mean failed independent reconstruction")
        ordinary_mean = future_mean(data, ordinary, ordinary["best"])
        conditional, reconstruction = conditional_ambient_reconstruction(data, fit)
        residual = data.temperature[1:]-point[1:]
        row = {"record": name, "n": len(residual), "RMSE_C": float(np.sqrt(np.mean(residual**2))),
               "bias_C": float(residual.mean()),
               "ordinary_mean_RMSE_C": float(np.sqrt(np.mean((data.temperature[1:]-ordinary_mean[1:])**2))),
               "conditional_ambient_reconstruction": reconstruction, "bands": {}}
        numeric.update({"time_"+str(j): data.time, "observed_"+str(j): data.temperature,
                        "point_"+str(j): point, "ordinary_point_"+str(j): ordinary_mean,
                        "conditional_ambient_point_"+str(j): conditional, "input_"+str(j): data.input})
        for mode in ("gaussian", "innovation_blocks"):
            role = "step" if "step_50" in name else "sine"
            ranges, audit = bands(data, fit, usable[mode], mode, empirical[role],
                                  counts[mode]["block_length"], seed=123261024+10007*j)
            entries = {}
            for family in ("mean", "observation", "ambient_mean", "ambient_observation"):
                if family in {"mean", "ambient_mean"}:
                    entries[family] = {"mean_width_C": float(np.mean(ranges[family+"_upper"]-ranges[family+"_lower"])),
                                       "notice": "Thermal mean truth is unobserved; measured-temperature fractions are not confidence coverage for the mean."}
                else:
                    entries[family] = coverage(data.temperature[1:], ranges[family+"_lower"], ranges[family+"_upper"])
            row["bands"][mode] = entries
            for key, values in ranges.items():
                numeric[mode+"_"+key+"_"+str(j)] = np.r_[data.temperature[0], values]
            audits.append({"record": name, "method": mode, "forward_max_error_C": discrepancy, **audit})
        summaries.append(row)
        print(fit.get("pair"), j+1, name, row["RMSE_C"], flush=True)
    np.savez_compressed(args.output/"predictions.npz", **numeric)
    write_json(args.output/"summary.json", {"pair": fit.get("pair"), "binding": binding,
               "bootstrap_counts": counts, "records": summaries,
               "conditioning": "Recorded commands and fixed first reading; zero prehistory. Primary future ambient correction is zero. +/-2C trajectories enter as a set, not a probability prior.",
               "information_reuse": split["information_reuse_notice"],
               "interval_notice": "Pointwise bootstrap mean ranges and working future-observation ranges. Gaussian observation quantiles integrate marginal Gaussian components deterministically; innovation-block observation ranges use eight correlated paths per parameter draw. Ambient bands are envelopes, not unconditional 95% probabilities.",
               "minimum_draws_override": args.minimum_draws})
    write_json(args.output/"audit.json", {"status": "passed for declared numerical checks", "binding": binding,
               "prediction_sha256": hashlib.sha256((args.output/"predictions.npz").read_bytes()).hexdigest(),
               "checks": audits, "scientific_adequacy": "not established; working-model and retrospective qualifications retained"})


if __name__ == "__main__":
    main()
