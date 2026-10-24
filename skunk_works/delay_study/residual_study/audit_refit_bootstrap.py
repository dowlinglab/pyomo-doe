"""Regenerate selected synthetic datasets, dense-audit them and refit 24 starts."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from covariance_fit import joint_fit
from fit_diagnostics import evaluate
import refit_bootstrap
from run_mean import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="0,11")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.input/"checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    identity = checkpoint["identity"]
    if hashlib.sha256((args.reference/"fit.json").read_bytes()).hexdigest() != identity["reference_fit_sha256"]:
        raise ValueError("Reference fit changed")
    refit_bootstrap.initialize(args.reference, identity["starts_per_refit"], identity["measurement"], identity["seed_base"])
    S = refit_bootstrap.STATE
    rows = []
    for index in [int(i) for i in args.indices.split(",")]:
        saved = next(d for d in checkpoint["draws"] if d["index"] == index)
        if saved["status"] not in {"complete", "requires_revisit"}:
            raise ValueError("Cannot audit a missing/failed fit as a success")
        seed, records = refit_bootstrap.synthetic(index)
        if seed != saved["seed"]:
            raise AssertionError("Saved seed mismatch")
        config = copy.deepcopy(S["config"])
        diagnostic, _ = evaluate(records, saved["fit"], config)
        recomputed = refit_bootstrap.statistics(records, saved["fit"])
        differences = [abs(recomputed[k]-saved["statistics"][k]) for k in
                       ("global_max_abs_acf", "global_max_abs_input_correlation", "global_max_Q30")
                       if recomputed[k] is not None]
        if max(differences) > 1e-7:
            raise AssertionError("Diagnostic statistics failed regeneration")
        config["multistarts"] = 24
        reference = S["reference"]
        full = joint_fit(records, config, reference["mode"], reference["hold"], reference["kernel"],
                         refit_bootstrap.baseline_seeds(records), noise_start=reference["best"]["noise_parameters"])
        write_json(args.output/("full_refit_"+str(index)+".json"), full)
        full_diagnostic, _ = evaluate(records, full, config)
        improvement = saved["fit"]["best"]["negative_log_likelihood"]-full["best"]["negative_log_likelihood"]
        rows.append({"index": index, "seed": seed, "dense_audit_short_fit": diagnostic["audit"],
                     "dense_audit_full_fit": full_diagnostic["audit"],
                     "max_statistic_regeneration_error": max(differences),
                     "full_24_start_nll_improvement": improvement,
                     "full_stationarity_warning": full["stationarity_warning"],
                     "requires_revisit": abs(improvement) > 1e-5 or full["stationarity_warning"]})
        print(identity["measurement"], "audited draw", index, "24-start improvement", improvement, flush=True)
    receipt = {"status": "requires_revisit" if any(r["requires_revisit"] for r in rows) else "passed",
               "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
               "scope": "Declared selected draws only: exact seed regeneration, independent mean/dense likelihood checks and 24-start refitting; not proof that every shorter refit is globally optimal",
               "draws": rows}
    write_json(args.output/"audit.json", receipt)
    if receipt["status"] != "passed":
        raise RuntimeError("Bootstrap refit check needs investigation; preserve outputs")


if __name__ == "__main__":
    main()
