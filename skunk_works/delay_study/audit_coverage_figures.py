"""Recompute the 84 displayed delay rows from terminal saved draw intervals.

Checks source/plot/CSV identities, counts, Wilson arithmetic, disconnected-set
widths and grid sensitivity independently of the summary producer's helpers.
Does not refit draws or certify finite-grid profile accuracy or visual layout.
"""
import argparse
import gzip
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from audit_prediction_uncertainty import compare_values
from plot_coverage import FIELDS, METHODS, MODELS, NOISES, load_delay_summary, sha
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def score_interval(covered, defined):
    if defined == 0:
        return None, None
    z = NormalDist().inv_cdf(.975)
    roots = np.sort(np.roots([defined+z*z, -(2*covered+z*z), covered*covered/defined]))
    return (0. if covered == 0 else float(roots[0]),
            1. if covered == defined else float(roots[1]))


def quadratic_value(points, statistic, query):
    x = np.array([p["log_value"] for p in points])
    y = np.array([p[statistic] for p in points])
    if not x[0] <= query <= x[-1]:
        return None
    exact = np.flatnonzero(abs(x-query) < 1e-12)
    if len(exact):
        return float(y[exact[0]])
    keep = np.r_[True, np.diff(x) > 1e-10]
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return None
    insertion = int(np.count_nonzero(x < query))
    first = int(np.clip(insertion-1, 0, len(x)-3))
    # Solve the three local interpolation equations instead of reusing the
    # summary producer's Lagrange formula. Centering preserves conditioning.
    shifted = x[first:first+3]-query
    coefficients = np.linalg.solve(np.column_stack([np.ones(3), shifted, shifted**2]), y[first:first+3])
    return float(coefficients[0])


def direct_statistics(draws, method, requested):
    if method not in METHODS or len(draws) > requested or any(d["status"] not in {"complete", "failed"} for d in draws):
        raise RuntimeError("Invalid saved draw collection/method")
    completed = [d for d in draws if d["status"] == "complete"]
    covered, widths, envelopes, component_counts, open_sets, changes = [], [], [], [], [], []
    profile = method.startswith("profile_")
    for draw in completed:
        truth = draw["truth_io"][2]
        if profile:
            suffix = "known" if method == "profile_known_sigma_iid" else "unknown"
            saved = draw["delay_profile"]
            components = saved["intervals_"+suffix+"_sigma_iid"]
            if any(not 0 < c["lower"] <= c["upper"] or not np.isfinite([c["lower"], c["upper"]]).all() for c in components):
                raise RuntimeError("Invalid saved profile component")
            if any(a["upper"] > b["lower"] for a, b in zip(components, components[1:])):
                raise RuntimeError("Overlapping or unsorted saved profile components")
            hit = any(c["lower"] <= truth <= c["upper"] for c in components)
            width = sum(c["upper"]-c["lower"] for c in components)
            envelopes.append(components[-1]["upper"]-components[0]["lower"] if components else 0.)
            component_counts.append(len(components))
            open_sets.append(any(c[f] == "open_at_exploration_bound" for c in components for f in ["lower_status", "upper_status"]))
            threshold = NormalDist().inv_cdf(.975)**2
            if abs(saved["chi2_95_df1"]-threshold) > 1e-10:
                raise RuntimeError("Profile reference threshold differs from chi-squared df1")
            value = quadratic_value(saved["points"], "lr_"+suffix+"_sigma_iid", np.log(truth))
            if value is not None:
                changes.append((value <= threshold) != hit)
        else:
            interval = draw["local_intervals_io"][method]
            if interval["covered"] is None:
                if interval["lower"] is not None or interval["upper"] is not None:
                    raise RuntimeError("Undefined interval has finite endpoints")
                continue
            lower, upper = interval["lower"][2], interval["upper"][2]
            if not 0 < lower <= upper or not np.isfinite([lower, upper]).all():
                raise RuntimeError("Invalid saved local endpoints")
            hit, width = lower <= truth <= upper, upper-lower
            if hit != interval["covered"][2]:
                raise RuntimeError("Saved local membership disagrees with endpoints")
        covered.append(hit)
        widths.append(width)
    n, k = len(covered), sum(covered)
    lower, upper = score_interval(k, n)
    result = {"requested": requested, "defined": n, "covered": k, "undefined_or_failed": requested-n,
              "coverage_when_defined": k/n if n else None, "wilson_when_defined_lower": lower,
              "wilson_when_defined_upper": upper, "binomial_mcse_when_defined": np.sqrt((k/n)*(1-k/n)/n) if n else None,
              "unconditional_coverage_lower": k/requested, "unconditional_coverage_upper": (k+requested-n)/requested,
              "completed_draws": len(completed), "failed_draws": len(draws)-len(completed),
              "missing_draws": requested-len(draws), "undefined_on_completed_draws": len(completed)-n}
    if widths:
        result.update(mean_width=float(np.mean(widths)), median_width=float(np.median(widths)),
                      width_q025=float(np.quantile(widths, .025)), width_q975=float(np.quantile(widths, .975)))
    else:
        result.update(dict.fromkeys(["mean_width", "median_width", "width_q025", "width_q975"]))
    result.update(dict.fromkeys(["mean_envelope_width", "mean_component_count", "open_exploration_sets",
                                "quadratic_interpolation_defined", "quadratic_interpolation_coverage_class_changes"]))
    if profile:
        result.update(mean_envelope_width=float(np.mean(envelopes)) if envelopes else None,
                      mean_component_count=float(np.mean(component_counts)) if component_counts else None,
                      open_exploration_sets=sum(open_sets), quadratic_interpolation_defined=len(changes),
                      quadratic_interpolation_coverage_class_changes=sum(changes))
    return result


def audit_values(actual, expected, errors):
    for field, value in expected.items():
        if value is None:
            if actual[field] is not None and not pd.isna(actual[field]):
                raise RuntimeError("Undefined coverage value replaced: "+field)
        elif actual[field] is None:
            raise RuntimeError("Defined coverage value missing: "+field)
        else:
            compare_values(actual[field], value, field, errors, atol=1e-10, rtol=1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=Path, default=HERE/"results/coverage")
    parser.add_argument("--summary", type=Path, default=HERE/"results/coverage_summary")
    parser.add_argument("--config", type=Path, default=HERE/"coverage_config.json")
    parser.add_argument("--figures", type=Path, default=HERE/"manuscript/figures")
    args = parser.parse_args()
    selected, checked_inputs = load_delay_summary(args.summary, args.coverage, args.config)
    manifest_path = args.figures/"coverage_figure_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data_path, csv_path = args.figures/"coverage_delay_data.json", args.figures/"coverage_delay_data.csv"
    if (manifest["input_sha256"] != checked_inputs or manifest["plot_data_sha256"] != sha(data_path)
            or manifest["csv_sha256"] != sha(csv_path)
            or manifest["figures"] != ["coverage_delay_coverage", "coverage_delay_width"]):
        raise RuntimeError("Coverage figure source/selection identity changed")
    rows = json.loads(data_path.read_text())["rows"]
    csv = pd.read_csv(csv_path)
    keys = [(r["case"], r["true_noise"], r["method"]) for r in selected]
    if (len(rows) != 84 or [(r["case"], r["true_noise"], r["method"]) for r in rows] != keys
            or len(csv) != 84 or list(zip(csv.case, csv.true_noise, csv.method)) != keys or list(csv.columns) != FIELDS):
        raise RuntimeError("Coverage figure data/CSV lost or duplicated a selected row")
    errors, numeric_values = {}, 0
    source_audit = json.loads((args.summary/"audit.json").read_text())
    for case, _, _, _ in MODELS:
        for noise, _ in NOISES:
            draws = []
            for replicate in range(300):
                path = args.coverage/"draws"/(case+"__"+noise)/(f"{replicate:04d}.json.gz")
                if path.exists():
                    if source_audit["input_sha256"].get(str(path)) != sha(path):
                        raise RuntimeError("Saved draw differs from the source summary audit")
                    draws.append(json.loads(gzip.decompress(path.read_bytes())))
            for method in METHODS:
                key = (case, noise, method)
                i = keys.index(key)
                expected = direct_statistics(draws, method, 300)
                for actual in [selected[i], rows[i], csv.iloc[i].to_dict()]:
                    audit_values(actual, expected, errors)
                    numeric_values += len(expected)
    report = {"status": "passed", "delay_rows": 84, "numeric_values_checked": numeric_values,
              "maximum_errors": errors, "figure_manifest_sha256": sha(manifest_path),
              "plot_data_sha256": sha(data_path), "csv_sha256": sha(csv_path), "source_summary_inputs": checked_inputs,
              "auditor_sources_sha256": {n: sha(HERE/n) for n in ["audit_coverage_figures.py", "plot_coverage.py", "audit_prediction_uncertainty.py"]},
              "scope": "Every displayed delay row recomputed from saved terminal interval endpoints: counts, independent score-root Wilson arithmetic, conditional/unconditional failure denominators, component-sum and envelope widths, exploration flags and independently solved quadratic interpolation diagnostic. No refitting, covariance audit, endpoint-accuracy certificate or rendered-pixel/layout review."}
    save_json(args.figures/"coverage_numeric_audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"source_summary_inputs", "scope"}}))


if __name__ == "__main__":
    main()
