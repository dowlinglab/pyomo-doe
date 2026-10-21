"""Derive comparison tables from saved fits; never rerun model estimation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_fits import save_json

HERE = Path(__file__).resolve().parent


def main():
    destination = HERE/"results"
    rows, identities = [], []
    for path in sorted((destination/"fits").glob("*.json")):
        d = json.loads(path.read_text())
        fit, spec, u = d["fit"], d["fit"]["spec"], d["uncertainty"]
        p = fit["canonical_io"] or [np.nan]*3
        rows.append({"case": path.stem, "calibration": d["calibration"], **spec,
                     "sse": fit["sse"], "n": fit["n"], "rmse_C": fit["rmse"],
                     "K": p[0], "tau_s": p[1], "theta_s": p[2],
                     "rank": u["rank_relative_1e-8"], "condition_J_log": u["condition_J_log"],
                     "validation_mean_rmse_C": float(np.mean([v["rmse_C"] for v in d["validation"]])),
                     "successful_starts": sum(x["success"] for x in fit["starts"]),
                     "total_starts": len(fit["starts"]),
                     "aicc_iid_conditional": fit["aicc_iid_conditional"],
                     "bic_iid_conditional": fit["bic_iid_conditional"]})
        identities.append({"file": str(path.relative_to(destination)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    summary = pd.DataFrame(rows)
    summary.to_csv(destination/"fit_summary.csv", index=False)
    comparisons = []
    grouping = ["calibration", "family", "order", "initialization", "hold", "method"]
    for key, group in summary[summary.family != "original"].groupby(grouping):
        io = group[group.coordinates == "io"].iloc[0]
        physical = group[group.coordinates == "physics"].iloc[0]
        values_io = np.asarray(io[["K", "tau_s", "theta_s"]], dtype=float)
        values_physics = np.asarray(physical[["K", "tau_s", "theta_s"]], dtype=float)
        comparisons.append({**dict(zip(grouping, key)), "sse_abs_delta": abs(io.sse-physical.sse),
                            "canonical_relative_delta_max": float(np.max(np.abs(values_io-values_physics)/values_io)),
                            "validation_mean_rmse_abs_delta_C": abs(io.validation_mean_rmse_C-physical.validation_mean_rmse_C),
                            "full_local_rank": bool(io["rank"] == physical["rank"] == 3),
                            "interpretation": "compare_common_interior_solution" if io["rank"] == physical["rank"] == 3 else "unidentified_delay_can_differ_between_equal_training_minima"})
    pd.DataFrame(comparisons).to_csv(destination/"coordinate_comparison.csv", index=False)
    previous = pd.read_csv(HERE/"baseline/initial_sweep_summary.csv").set_index("case")
    current = summary.set_index("case")
    common = previous.index.intersection(current.index)
    deltas = np.abs(previous.loc[common, "sse"]-current.loc[common, "sse"])
    manifest = json.loads((destination/"manifest.json").read_text())
    assert len(summary) == manifest["expected_cases"] and not manifest["failed"]
    assert deltas.max() < 1e-9, "Covariance fix unexpectedly changed fitting"
    save_json(destination/"comparison_receipt.json", {
        "source_manifest_sha256": hashlib.sha256((destination/"manifest.json").read_bytes()).hexdigest(),
        "producer": "summarize_fits.py", "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": identities, "fit_count": len(summary), "earlier_fit_comparisons": len(common),
        "max_sse_change_from_covariance_fix": float(deltas.max()),
        "outputs": {name: hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in ("fit_summary.csv", "coordinate_comparison.csv")},
        "limits": "Aggregation of recorded fits; does not establish noise-model correctness or human scientific review"})
    print(f"Summarized {len(summary)} cases; {len(common)} fitting deltas <= {deltas.max():.3g}.")


if __name__ == "__main__":
    main()
