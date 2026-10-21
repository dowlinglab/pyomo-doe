"""Audit completed bootstrap records and derive tables without refitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import binom

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint_mc_interval(values, probability):
    ordered = np.sort(values)
    n = len(ordered)
    lower = int(binom.ppf(.025, n, probability))
    upper = int(binom.ppf(.975, n, probability))+1
    return {"order_lower_1based": lower, "order_upper_1based": upper,
            "value_lower": float(ordered[lower-1]) if lower else None,
            "value_upper": float(ordered[upper-1]) if upper <= n else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE/"results/bootstrap")
    parser.add_argument("--fits", type=Path, default=HERE/"results/fits")
    args = parser.parse_args()
    manifest = json.loads((args.input/"manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["failed"]:
        raise RuntimeError("Bootstrap phase is not complete")
    rows, sources = [], {}
    total, undefined = 0, 0
    max_quantile_error, max_std_error, max_sse_error = 0., 0., 0.
    for path in sorted(args.input.glob("*__*.json")):
        result = json.loads(path.read_text())
        if result["identity"] != manifest["identity"] or result["status"] != "complete":
            raise RuntimeError("Inconsistent bootstrap case state: "+path.name)
        original = args.fits/result["fit_file"]
        if sha(original) != result["fit_sha256"]:
            raise RuntimeError("Bootstrap source fit changed")
        fit = json.loads(original.read_text())
        arrays_path = path.with_suffix(".npz")
        if sha(arrays_path) != result["arrays_sha256"]:
            raise RuntimeError("Bootstrap arrays changed")
        arrays = np.load(arrays_path)
        n = result["target_draws"]
        if arrays["parameters"].shape != (n, 3) or not arrays["completed"].all() or not arrays["success"].all():
            raise RuntimeError("Bootstrap completion/shape mismatch")
        values = arrays["parameters"]
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise RuntimeError("Nonfinite/nonpositive successful parameter draw")
        if len(result["attempts"]) != n or result["successful_draws"] != n:
            raise RuntimeError("Missing optimizer-attempt accounting")
        for i in range(n):
            attempts = [a for a in result["attempts"][str(i)]["attempts"] if a["status"] == "converged"]
            if not attempts:
                raise RuntimeError("Success flag without a converged attempt")
            converged = [s["sse"] for s in attempts[-1]["starts"] if s["success"]]
            error = abs(min(converged)-arrays["sse"][i])
            max_sse_error = max(max_sse_error, error)
            if error > 1e-7:
                raise RuntimeError("Saved draw objective differs from selected multistart objective")
        # Independent algebraic transformation, rather than importing the producer.
        physical = np.column_stack([.032/values[:, 0], .032*values[:, 1]/values[:, 0], values[:, 2]])
        nonestimable = fit["uncertainty"]["nonestimable_native_parameters"]
        for coordinates, samples, names in [("io", values, ["K", "tau", "theta"]),
                                            ("physics", physical, ["Ua", "Cp", "theta"])]:
            for j, name in enumerate(names):
                stored = result["summary"][coordinates][name]
                row = {"case": original.stem, "calibration": fit["calibration"],
                       "family": fit["fit"]["spec"]["family"], "order": fit["fit"]["spec"]["order"],
                       "initialization": fit["fit"]["spec"]["initialization"],
                       "method": result["method"], "coordinates": coordinates,
                       "parameter": name, "draws": n, "status": stored["status"],
                       "lower": stored["lower"], "median": stored["median"], "upper": stored["upper"],
                       "std": stored.get("std")}
                if name == "theta" and "theta" in nonestimable:
                    if any(stored[key] is not None for key in ["lower", "median", "upper"]):
                        raise RuntimeError("A structural null direction was assigned a bootstrap interval")
                    undefined += 1
                else:
                    expected = np.quantile(samples[:, j], [.025, .5, .975])
                    error = float(np.max(abs(expected-[stored["lower"], stored["median"], stored["upper"]])))
                    max_quantile_error = max(max_quantile_error, error)
                    std_error = abs(float(np.std(samples[:, j], ddof=1))-stored["std"])
                    max_std_error = max(max_std_error, std_error)
                    if error > 1e-10 or std_error > 1e-10:
                        raise RuntimeError("Bootstrap summary/transform disagrees with saved arrays")
                    for label, p in [("lower_endpoint", .025), ("upper_endpoint", .975)]:
                        interval = endpoint_mc_interval(samples[:, j], p)
                        row.update({label+"_mc_"+key: value for key, value in interval.items()})
                rows.append(row)
        total += n
        sources[path.name] = sha(path)
        sources[arrays_path.name] = sha(arrays_path)
    if len(sources)//2 != manifest["expected_jobs"] or total != sum(r["target"] for r in manifest["completed"].values()):
        raise RuntimeError("Bootstrap job/draw count mismatch")
    table = args.input/"summary.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    snapshot = json.loads((args.input/"source_snapshot.json").read_text())
    receipt = {"status": "completed_draw_integrity_and_coordinate_summary_checks_passed",
               "producer": "summarize_bootstrap.py", "producer_sha256": sha(Path(__file__)),
               "producer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
               "numerical_source_commit": snapshot["commit"], "numerical_manifest_sha256": sha(args.input/"manifest.json"),
               "jobs_checked": len(sources)//2, "draws_checked": total, "summary_rows": len(rows),
               "undefined_null_parameter_rows": undefined, "maximum_summary_quantile_error": max_quantile_error,
               "maximum_summary_std_error": max_std_error, "maximum_draw_sse_error": max_sse_error,
               "inputs_sha256": sources, "summary_sha256": sha(table),
               "endpoint_mc_notice": "Pointwise 95% order-statistic bounds for each bootstrap-distribution endpoint, using binomial counts. This quantifies finite-resampling error, not additional physical-parameter confidence or simultaneous endpoint coverage.",
               "limits": "Integrity and independent coordinate algebra only. Does not establish global minima, noise stationarity, nominal coverage, physical mechanism or human scientific review."}
    (args.input/"summary_audit.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(f"Audited {len(sources)//2} jobs and {total} draws; {undefined} null-coordinate rows correctly undefined.")


if __name__ == "__main__":
    main()
