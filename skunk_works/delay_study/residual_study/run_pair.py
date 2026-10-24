"""Restricted one-sine/one-step fit after the Tier 0 structure decision gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np

from covariance_fit import alternating_fit, joint_fit
from data import load
from fit_diagnostics import evaluate
from mean import fit_mean
from run_mean import write_json


def pair_records(pair):
    prefix = {30: "sin_50_50", 120: "sin_2_50", 300: "sin_5_50"}[pair["sine_period_s"]]
    return ["validation_experiment_env_2_"+prefix+"_run_"+str(pair["sine_run"])+".csv",
            "validation_experiment_env_2_step_50_run_"+str(pair["step_run"])+".csv"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=["P"+str(i) for i in range(1, 7)], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    job = {"status": "running", "pair": args.pair, "pid": os.getpid(), "started_unix": started}
    write_json(args.output/"job.json", job)
    try:
        selection_path = HERE/"structure_selection.json"
        selection = json.loads(selection_path.read_text())
        if selection["status"] != "frozen_working_structure" or selection["model_order"] != 1:
            raise RuntimeError("Tier 0 structure decision is not frozen for the n1 study")
        config = json.loads((HERE/"config.json").read_text())
        pair = next(p for p in config["pairs"] if p["id"] == args.pair)
        records = pair_records(pair)
        if not set(records).issubset(config["records"]) or len(set(records)) != 2:
            raise AssertionError("Invalid pair membership")
        evaluation = [name for name in config["records"] if name not in records]
        if len(evaluation) != 10:
            raise AssertionError("Expected ten excluded evaluation records")
        config.update(records=records, primary_hold=selection["hold"],
                      ambient_offset_bounds_C=selection["ambient_offset_bounds_C"])
        mode, kernel = selection["ambient_mode"], selection["kernel"]
        write_json(args.output/"effective_config.json", config)
        write_json(args.output/"split.json", {"pair": pair, "training": records, "evaluation": evaluation,
                   "structure_selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                   "information_reuse_notice": "Tier 0 inspected every ENV2 record. Evaluation is retrospective conditional on its structural choice; numerical thermal/ambient/noise fitting here uses only the two training records. Evaluation temperatures are not loaded in this driver.",
                   "numeric_policy": "Fresh pair OLS/GLS/joint fits; Tier 0 noise values are neither fixed nor supplied as prior information"})
        data = [load(name) for name in records]
        write_json(args.output/"raw_bindings.json", {d.source: hashlib.sha256((HERE.parents[2]/d.source).read_bytes()).hexdigest() for d in data})
        def progress(row):
            write_json(args.output/"progress.json", row)
            print(args.pair, row.get("start"), row.get("negative_log_likelihood", row.get("SSE", row.get("exception"))), flush=True)
        baseline = fit_mean(data, config, mode, config["primary_hold"], progress)
        write_json(args.output/"ols.json", baseline)
        noise_seed = [] if kernel == "iid" else [10.]+([.2] if kernel.endswith("_nugget") else [])
        alternating = alternating_fit(data, config, mode, config["primary_hold"], kernel, baseline, noise_seed)
        write_json(args.output/"alternating.json", alternating)
        result = joint_fit(data, config, mode, config["primary_hold"], kernel, baseline, progress, alternating)
        diagnostic, arrays = evaluate(data, result, config)
        result.update(diagnostic)
        result.update(pair=args.pair, tier=pair["tier"],
                      alternating_minus_joint_nll=alternating["history"][-1]["negative_log_likelihood"]-result["best"]["negative_log_likelihood"],
                      structure_limitations=selection["limitations"])
        write_json(args.output/"fit.json", result)
        np.savez_compressed(args.output/"trajectories.npz", **arrays)
        job.update(status="complete", elapsed_s=time.time()-started)
        write_json(args.output/"job.json", job)
    except Exception as exc:
        job.update(status="failed", elapsed_s=time.time()-started, exception=repr(exc), traceback=traceback.format_exc())
        write_json(args.output/"job.json", job)
        raise


if __name__ == "__main__":
    main()
