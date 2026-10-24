"""Declared one-factor checks before freezing the Tier 0 working structure."""
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

from audit_mean import audit
from covariance_fit import alternating_fit, joint_fit
from data import load
from fit_diagnostics import evaluate
from mean import fit_mean
from run_mean import write_json
from study_cases import CASES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    begun = time.time()
    job = {"case": args.case, "pid": os.getpid(), "started_unix": begun, "status": "running"}
    write_json(args.output/"job.json", job)
    try:
        config = json.loads((HERE/"config.json").read_text())
        config.update(CASES[args.case])
        if args.starts is not None:
            config["multistarts"] = args.starts
        if "record_filter" in config:
            config["records"] = [name for name in config["records"] if config["record_filter"] in name]
        config["sensitivity_notice"] = "Agent-declared diagnostic only; primary n1 left-held +/-2C model unchanged. Startup shifts are fixed scenarios, not a probability law or independent per-run worst-case envelope. Protocol fits are exploratory, not additional calibration pairs."
        write_json(args.output/"effective_config.json", config)
        write_json(args.output/"forward_audit.json", audit())
        records = [load(name) for name in config["records"]]
        bindings = {d.source: hashlib.sha256((HERE.parents[2]/d.source).read_bytes()).hexdigest() for d in records}
        write_json(args.output/"raw_bindings.json", bindings)
        def progress(row):
            write_json(args.output/"progress.json", row)
            print(args.case, row.get("start"), row.get("negative_log_likelihood", row.get("SSE", row.get("exception"))), flush=True)
        baseline = fit_mean(records, config, "constant", config["primary_hold"], progress)
        write_json(args.output/"ols.json", baseline)
        alternating = alternating_fit(records, config, "constant", config["primary_hold"], "ou_nugget", baseline, [10., .2])
        write_json(args.output/"alternating.json", alternating)
        result = joint_fit(records, config, "constant", config["primary_hold"], "ou_nugget", baseline, progress, alternating)
        diagnostics, arrays = evaluate(records, result, config)
        result.update(diagnostics)
        result["alternating_minus_joint_nll"] = alternating["history"][-1]["negative_log_likelihood"]-result["best"]["negative_log_likelihood"]
        result["case"] = args.case
        result["sensitivity_notice"] = config["sensitivity_notice"]
        write_json(args.output/"fit.json", result)
        np.savez_compressed(args.output/"trajectories.npz", **arrays)
        job.update(status="complete", elapsed_s=time.time()-begun)
        write_json(args.output/"job.json", job)
    except Exception as exc:
        job.update(status="failed", elapsed_s=time.time()-begun, exception=repr(exc), traceback=traceback.format_exc())
        write_json(args.output/"job.json", job)
        raise


if __name__ == "__main__":
    main()
