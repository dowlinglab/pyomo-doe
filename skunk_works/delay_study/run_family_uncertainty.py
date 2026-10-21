"""Propagate saved n1 bootstrap draws to identifiable physical-family limits.

This does not select arbitrary gauge coordinates or report them as estimates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from family_uncertainty import boundary_functions, FUNCTIONS
from run_fits import save_json
from summarize_bootstrap import endpoint_mc_interval

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=Path, default=HERE/"results/bootstrap")
    parser.add_argument("--fits", type=Path, default=HERE/"results/fits")
    parser.add_argument("--output", type=Path, default=HERE/"results/family_uncertainty")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.bootstrap/"manifest.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("Saved bootstrap phase is not complete")
    sources, rows, arrays = {}, [], {}
    for path in sorted(args.bootstrap.glob("*__chain_n1_io_cold_linear_exact__*.json")):
        record = json.loads(path.read_text())
        fit_path, arrays_path = args.fits/record["fit_file"], path.with_suffix(".npz")
        if sha(fit_path) != record["fit_sha256"] or sha(arrays_path) != record["arrays_sha256"]:
            raise RuntimeError("Saved bootstrap inputs changed")
        fit = json.loads(fit_path.read_text())
        z = np.load(arrays_path)
        if not z["completed"].all() or not z["success"].all() or len(z["parameters"]) != record["target_draws"]:
            raise RuntimeError("Bootstrap failure must remain explicit")
        values = boundary_functions(z["parameters"])
        point = boundary_functions([fit["fit"]["parameters"][name] for name in ("K", "tau", "theta")])[0]
        arrays[path.stem] = values
        for j, name in enumerate(FUNCTIONS):
            quantiles = np.quantile(values[:, j], [.025, .5, .975])
            row = {"case": fit_path.stem, "calibration": fit["calibration"], "method": record["method"],
                   "function": name, "draws": len(values), "point": point[j],
                   "lower": quantiles[0], "median": quantiles[1], "upper": quantiles[2],
                   "sample_std": np.std(values[:, j], ddof=1),
                   "status": "conditional_percentiles_of_identifiable_family_function"}
            for endpoint, probability in (("lower", .025), ("upper", .975)):
                row.update({endpoint+"_endpoint_mc_"+key: value for key, value in endpoint_mc_interval(values[:, j], probability).items()})
            rows.append(row)
        for source in (path, fit_path, arrays_path):
            sources[str(source)] = sha(source)
    if len(arrays) != 15:
        raise RuntimeError("Expected three cold n1 calibrations times five resampling methods")
    identity = {name: sha(HERE/name) for name in ("family_uncertainty.py", "run_family_uncertainty.py", "summarize_bootstrap.py", "run_fits.py")}
    if (args.output/"manifest.json").exists():
        previous = json.loads((args.output/"manifest.json").read_text())
        if previous["source_sha256"] != identity or previous["input_sha256"] != sources:
            raise RuntimeError("Preserve family-function results from the earlier producer/inputs")
        if any(sha(args.output/name) != value for name, value in previous["output_sha256"].items()):
            raise RuntimeError("Saved family-function outputs changed")
        print("Reused the verified family-function phase")
        return
    np.savez_compressed(args.output/"boundary_draws.npz", **arrays)
    pd.DataFrame(rows).to_csv(args.output/"summary.csv", index=False)
    save_json(args.output/"manifest.json", {"status": "complete", "input_sha256": sources, "source_jobs": len(arrays),
                                            "reused_bootstrap_draws": sum(len(value) for value in arrays.values()),
                                            "new_model_refits": 0, "functions_in_column_order": FUNCTIONS,
                                            "source_sha256": identity,
                                            "output_sha256": {name: sha(args.output/name) for name in ("boundary_draws.npz", "summary.csv")},
                                            "notice": "Uncertainty of identifiable transfer coefficients and analytic positive-family boundary functions. These are not marginal confidence intervals for CpH, CpS or Ub. Boundary functions describe the full strictly positive original family before numerical fit boxes; CpH endpoints and zero lower CpS/Ub endpoints are open. No arbitrary family member was selected. Percentiles are conditional on the recorded residual resampling/model/calibration assumptions."})
    print(json.dumps({"source_jobs": len(arrays), "reused_draws": sum(len(value) for value in arrays.values()), "rows": len(rows)}))


if __name__ == "__main__":
    main()
