"""Check each thermal profiling constraint at the saved MLE with dense likelihood."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np

from data import load
from fit_diagnostics import evaluate
from likelihood_profiles import profile_point
from run_mean import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ref = json.loads((args.reference/"fit.json").read_text())
    config = json.loads((args.reference/"effective_config.json").read_text())
    data = [load(n) for n in ref["record_names"]]
    rows = []
    for name in ("K", "tau", "theta"):
        value = ref["best"]["coefficients"][0] if name == "K" else ref["best"][name]
        point = profile_point(data, config, ref, name, value)
        if point["status"] != "complete":
            raise AssertionError(("MLE profile did not converge", name, point))
        delta = point["best"]["negative_log_likelihood"]-ref["best"]["negative_log_likelihood"]
        if abs(delta) > 1e-5:
            raise AssertionError(("MLE profile objective mismatch", name, delta))
        b = point["best"]
        reconstructed = dict(ref, best={"tau": float(np.exp(b["coordinates"][0])),
                                      "theta": float(np.exp(b["coordinates"][1])),
                                      "coefficients": b["coefficients"], "sigma_total_ML_C": b["sigma_total_ML_C"],
                                      "noise_parameters": [float(np.exp(b["coordinates"][2])), b["coordinates"][3]],
                                      "negative_log_likelihood": b["negative_log_likelihood"]})
        diagnosis, _ = evaluate(data, reconstructed, config)
        rows.append({"parameter": name, "fixed_value": value, "nll_difference": delta,
                     "profile": point, "independent_selected_fit_audit": diagnosis["audit"]})
    write_json(args.output/"audit.json", {"status": "passed", "checks": rows,
               "scope": "All three profiling constraints recover the saved MLE and independently reconstructed likelihood. Does not verify the entire grid, crossing locations or confidence coverage."})


if __name__ == "__main__":
    main()
