"""Summarize completed six-pair bootstrap cells without hiding failed draws."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Need at least two finite usable draws")
    ordered = np.sort(x)
    quantiles = np.quantile(x, [.025, .5, .975])
    # Order-statistic ranges describe simulation error in conditional quantiles.
    # They say nothing about physical-model adequacy or frequentist coverage.
    mc = []
    for probability in (.025, .975):
        ranks = binom.ppf([.025, .975], len(x), probability).astype(int)
        lo = max(int(ranks[0])-1, 0)
        hi = min(int(ranks[1]), len(x)-1)
        mc.append([float(ordered[lo]), float(ordered[hi])])
    half = len(x)//2
    first = np.quantile(x[:half], [.025, .975])
    second = np.quantile(x[half:], [.025, .975])
    width = max(float(quantiles[2]-quantiles[0]), 1e-30)
    return {"quantiles_025_50_975": quantiles.tolist(),
            "standard_deviation": float(np.std(x, ddof=1)),
            "mean": float(np.mean(x)),
            "min_max": [float(np.min(x)), float(np.max(x))],
            "quantile_MC_order_statistic_ranges": mc,
            "half_batch_endpoint_disagreement_over_full_width": float(np.max(np.abs(first-second))/width)}


def parameters(best):
    K, tau, theta = best["coefficients"][0], best["tau"], best["theta"]
    length, eta = best["noise_parameters"]
    sigma = best["sigma_total_ML_C"]
    row = {"K": K, "tau": tau, "theta": theta, "slow_pole_s": max(tau, theta),
           "fast_pole_s": min(tau, theta), "pole_sum_s": tau+theta,
           "pole_product_s2": tau*theta, "ell_s": length, "eta_variance_fraction": eta,
           "sigma_total_C": sigma, "sigma_slow_C": sigma*np.sqrt(1-eta),
           "sigma_measurement_C": sigma*np.sqrt(eta)}
    row.update({"ambient_"+str(i+1)+"_C": a for i, a in enumerate(best["coefficients"][1:])})
    return row


def summarize_cell(folder, reference, expected):
    checkpoint_path = folder/"checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    if checkpoint["status"] != "complete":
        raise ValueError("Cell is still live or incomplete: "+str(folder))
    fit_path = reference/"fit.json"
    fit = json.loads(fit_path.read_text())
    identity = checkpoint["identity"]
    if identity["reference_fit_sha256"] != digest(fit_path) or identity["draws"] != expected:
        raise ValueError("Reference/count binding mismatch")
    rows = sorted(checkpoint["draws"], key=lambda r: r["index"])
    if [r["index"] for r in rows] != list(range(expected)):
        raise ValueError("Missing or duplicate draw identity")
    counts = {state: sum(r["status"] == state for r in rows)
              for state in ("complete", "failed", "requires_revisit")}
    if sum(counts.values()) != expected:
        raise ValueError("Unknown draw state")
    good = [r for r in rows if r["status"] == "complete"]
    samples = [parameters(r["fit"]["best"]) for r in good]
    if not samples:
        raise ValueError("No usable draws; preserve and investigate")
    point = parameters(fit["best"])
    summaries = {name: distribution([r[name] for r in samples]) for name in point}
    modes = {}
    for label, predicate in [("tau_ge_theta", lambda b: b["tau"] >= b["theta"]),
                             ("tau_lt_theta", lambda b: b["tau"] < b["theta"])]:
        selected = [r for r in samples if predicate(r)]
        modes[label] = {"count": len(selected), "fraction_of_usable": len(selected)/len(good),
                        "thermal_quantiles_025_50_975": {name: np.quantile([r[name] for r in selected], [.025, .5, .975]).tolist()
                         for name in ("K", "tau", "theta")} if selected else {}}
    config = json.loads((reference/"effective_config.json").read_text())
    bounds = dict(config["thermal_bounds"], ell_s=[.05, 3000.], eta_variance_fraction=[0., 1.])
    bounds.update({name: config["ambient_offset_bounds_C"] for name in point if name.startswith("ambient_")})
    hit_counts = {name: sum(abs(r[name]-limits[0]) <= 1e-6*max(1., abs(limits[0])) or
                            abs(r[name]-limits[1]) <= 1e-6*max(1., abs(limits[1])) for r in samples)
                  for name, limits in bounds.items()}
    return {"requested": expected, "status_counts": counts, "usable": len(good),
            "selected_draw_policy": "Only complete stationary refits enter quantiles; failures and revisit counts stay in the requested denominator.",
            "point": point, "distributions": summaries, "pole_label_modes": modes,
            "bound_hit_counts_among_usable": hit_counts,
            "issues": [r["index"] for r in rows if r["status"] != "complete"],
            "binding": {"reference_fit_sha256": digest(fit_path), "checkpoint_sha256": digest(checkpoint_path),
                        "source_sha256": identity["source_sha256"], "mechanism": identity["measurement"],
                        "block_length_samples": identity["block_length"]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads((args.matrix/"manifest.json").read_text())
    if matrix["status"] not in {"complete", "complete_with_issues"} or len(matrix["jobs"]) != 24:
        raise ValueError("Need the complete declared six-pair uncertainty matrix")
    output = {"status": "conditional uncertainty summary", "pairs": {},
              "notice": "Percentile ranges are conditional working-model/bootstrap summaries, not posterior credible intervals or demonstrated coverage. Block resampling breaks command alignment and cannot correct deterministic mean bias. Pole labels are not physical assignments. Monte Carlo ranges only assess finite simulation precision; pairs/evaluations overlap."}
    for pair in ["P"+str(i) for i in range(1, 7)]:
        output["pairs"][pair] = {}
        for method, expected in [("gaussian", 400), ("blocks60", 400), ("blocks30", 100), ("blocks120", 100)]:
            name = pair+"_"+method
            if matrix["jobs"][name]["status"] not in {"complete", "requires_revisit"}:
                raise ValueError("A cell failed; do not silently omit "+name)
            output["pairs"][pair][method] = summarize_cell(args.matrix/name, args.references/pair, expected)
    output["binding"] = {"matrix_manifest_sha256": digest(args.matrix/"manifest.json")}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"uncertainty.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
