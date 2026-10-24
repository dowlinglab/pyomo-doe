"""Reconstruct saved fits independently; audit values and residual diagnostics."""
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
from run_mean import write_json


def acf(values, nlags=30):
    x = values-values.mean()
    denominator = x@x
    return [float(x[k:]@x[:-k]/denominator) for k in range(1, min(nlags+1, len(x)))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.input/"source_snapshot.json").read_text())
    config = json.loads((args.input/"effective_config.json").read_text())
    records = [load(name) for name in config["records"]]
    bound_checks = []
    for d in records:
        relative = d.source
        raw = HERE.parents[2]/relative
        if hashlib.sha256(raw.read_bytes()).hexdigest() != source["sha256"][relative]:
            raise AssertionError("Auditor raw-data snapshot differs from producer")
        bound_checks.append(relative)
    summaries = {}
    checks = []
    for mode in ("constant", "linear"):
        path = args.input/(mode+"_fit.json")
        fit = json.loads(path.read_text())
        best = fit["best"]
        coefficients = np.asarray(best["coefficients"])
        width = 1 if mode == "constant" else 2
        trajectories = np.load(args.input/(mode+"_trajectories.npz"))
        total = 0.
        rows = []
        errors = []
        for j, d in enumerate(records):
            if not np.array_equal(trajectories["observed_"+str(j)], d.temperature):
                raise AssertionError("Observed values differ from raw CSV")
            nuisance = coefficients[1+j*width:1+(j+1)*width]
            endpoints = np.repeat(nuisance, 2) if width == 1 else nuisance
            independent = reference(d, coefficients[0], best["tau"], best["theta"], endpoints, fit["hold"])
            saved = trajectories["predicted_"+str(j)]
            error = float(np.max(np.abs(independent-saved)))
            if error > 1e-7:
                raise AssertionError((d.name, "Independent trajectory disagreement", error))
            residual = d.temperature[1:]-independent[1:]
            sse = float(residual@residual)
            total += sse
            errors.append(error)
            rows.append({"name": d.name, "RMSE_C": float(np.sqrt(sse/len(residual))),
                         "bias_C": float(residual.mean()), "residual_acf_1_to_30": acf(residual),
                         "ambient_endpoints_C": endpoints.tolist(),
                         "ambient_bound_hits": int(np.sum(np.abs(endpoints) >= 2-1e-7))})
        if abs(total-best["SSE"]) > 1e-6:
            raise AssertionError("Saved objective failed independent reconstruction")
        lo, hi = config["ambient_offset_bounds_C"]
        if np.any(coefficients[1:] < lo-1e-8) or np.any(coefficients[1:] > hi+1e-8):
            raise AssertionError("Ambient bounds violated")
        if not best["success"]:
            raise AssertionError("Selected mean fit did not converge")
        checks.append({"mode": mode, "SSE_difference": total-best["SSE"],
                       "max_forward_difference_C": max(errors),
                       "fit_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "trajectories_sha256": hashlib.sha256((args.input/(mode+"_trajectories.npz")).read_bytes()).hexdigest()})
        summaries[mode] = {"SSE": total, "RMSE_C": float(np.sqrt(total/fit["nobs"])),
                           "K": float(coefficients[0]), "tau_s": best["tau"], "theta_s": best["theta"],
                           "runs_at_ambient_bound": sum(row["ambient_bound_hits"] > 0 for row in rows),
                           "failed_starts": sum(not s["success"] for s in fit["starts"]),
                           "per_run": rows}
    write_json(args.output/"audit.json", {"status": "passed", "producer_commit": source["commit"],
               "scope": "Independent matrix-exponential trajectory and objective reconstruction, raw-data and ambient-bound checks; not a global-optimum certificate or uncertainty audit",
               "raw_files_checked": bound_checks, "checks": checks})
    write_json(args.output/"diagnostics.json", {"description": "Unweighted pooled baseline only; no covariance adjustment or uncertainty claims yet",
               "summaries": summaries,
               "linear_SSE_reduction_fraction": 1-summaries["linear"]["SSE"]/summaries["constant"]["SSE"]})
    print(json.dumps({"audit": "passed", "pooled_RMSE_C": {k: v["RMSE_C"] for k, v in summaries.items()},
                      "runs_at_ambient_bound": {k: v["runs_at_ambient_bound"] for k, v in summaries.items()}}))


if __name__ == "__main__":
    main()
