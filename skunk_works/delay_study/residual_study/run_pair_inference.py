"""Compute local information or a nuisance-refitted thermal likelihood profile."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from data import load
from likelihood_profiles import curve
from run_mean import write_json
from teaching_inference import local_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameter", choices=["K", "tau", "theta"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fit_path = args.reference/"fit.json"
    fit = json.loads(fit_path.read_text())
    config = json.loads((args.reference/"effective_config.json").read_text())
    records = [load(name) for name in fit["record_names"]]
    binding = {"reference_fit_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
               "reference_config_sha256": hashlib.sha256((args.reference/"effective_config.json").read_bytes()).hexdigest()}
    if args.parameter:
        def progress(row, rows):
            write_json(args.output/"progress.json", {"binding": binding, "parameter": args.parameter,
                       "points": rows, "status": "running"})
            print(args.parameter, row["value"], row["twice_nll_difference"], row["status"], flush=True)
        result = curve(records, config, fit, args.parameter, progress)
        result["binding"] = binding
        write_json(args.output/"profile.json", result)
    else:
        ordinary = json.loads((args.reference/"ols.json").read_text())
        result = local_analysis(records, fit, ordinary)
        result["binding"] = dict(binding, ordinary_fit_sha256=hashlib.sha256((args.reference/"ols.json").read_bytes()).hexdigest())
        write_json(args.output/"information.json", result)


if __name__ == "__main__":
    main()
