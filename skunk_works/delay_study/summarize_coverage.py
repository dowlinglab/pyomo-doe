"""Failure-aware Monte Carlo summaries, only after the prespecified run ends.

All draws remain in the requested denominator. Defined-method coverage and its
Wilson interval are conditional on a method returning an interval. Worst/best
unconditional bounds retain undefined intervals and failed or missing draws.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from bootstrap import replicate_seed
from coverage_study import wilson_interval
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coverage_counts(covered, defined, requested):
    if not 0 <= covered <= defined <= requested or requested <= 0:
        raise ValueError("Invalid method coverage counts")
    unknown = requested-defined
    conditional_wilson = wilson_interval(covered, defined)
    low_wilson = wilson_interval(covered, requested)
    high_wilson = wilson_interval(covered+unknown, requested)
    p = covered/defined if defined else None
    return {"requested": requested, "defined": defined, "covered": covered,
            "undefined_or_failed": unknown, "coverage_when_defined": p,
            "wilson_when_defined_lower": None if conditional_wilson is None else conditional_wilson[0],
            "wilson_when_defined_upper": None if conditional_wilson is None else conditional_wilson[1],
            "binomial_mcse_when_defined": None if p is None else float(np.sqrt(p*(1-p)/defined)),
            "unconditional_coverage_lower": covered/requested,
            "unconditional_coverage_upper": (covered+unknown)/requested,
            "conservative_wilson_lower": low_wilson[0], "conservative_wilson_upper": high_wilson[1]}


def quadratic_grid_statistic(log_grid, values, query):
    """Three-point interpolation sensitivity; no convexity/global certificate."""
    x, y = np.asarray(log_grid), np.asarray(values)
    if query < x[0] or query > x[-1]:
        return None
    exact = np.flatnonzero(abs(x-query) < 1e-12)
    if len(exact):
        return float(y[exact[0]])
    # Drop only numerically duplicated abscissae for this secondary diagnostic;
    # the original saved points and primary profile intervals are untouched.
    keep = np.r_[True, np.diff(x) > 1e-10]
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return None
    right = int(np.searchsorted(x, query))
    first = max(0, min(len(x)-3, right-1))
    x, y = x[first:first+3]-query, y[first:first+3]
    result = sum(y[j]*np.prod([-x[k]/(x[j]-x[k]) for k in range(3) if k != j]) for j in range(3))
    return float(result) if np.isfinite(result) else None


def descriptive(values):
    values = np.asarray(values)
    if not len(values):
        return {"mean_width": None, "median_width": None, "width_q025": None, "width_q975": None}
    return {"mean_width": float(values.mean()), "median_width": float(np.median(values)),
            "width_q025": float(np.quantile(values, .025)), "width_q975": float(np.quantile(values, .975))}


def verify_chunk_counts(draws, case, noise, requested, chunk_size, manifest):
    """A missing file is explainable only by a recorded failed worker chunk."""
    statuses = {draw["replicate"]: draw["status"] for draw in draws}
    if len(statuses) != len(draws) or any(status not in ("complete", "failed") for status in statuses.values()):
        raise RuntimeError("Duplicate replicate or unrecognized draw status")
    for first in range(0, requested, chunk_size):
        indices = list(range(first, min(requested, first+chunk_size)))
        key = case+"__"+noise+f"__{indices[0]:04d}_{indices[-1]:04d}"
        if key in manifest["completed"]:
            saved = manifest["completed"][key]
            if any(index not in statuses for index in indices) or any(
                    sum(statuses.get(index) == status for index in indices) != saved[status]
                    for status in ("complete", "failed")):
                raise RuntimeError("Completed chunk's saved draw files disagree with its manifest")
        elif key not in manifest["failed"]:
            raise RuntimeError("Missing chunk has no recorded terminal failure")


def summarize_scenario(draws, case, noise, requested):
    complete = [draw for draw in draws if draw["status"] == "complete"]
    failed = sum(draw["status"] == "failed" for draw in draws)
    missing = requested-len(draws)
    rows = []
    methods = ["iid_known_sigma", "iid_estimated_sigma", "hac10", "hac30", "hac60"]
    for coordinates, names in [("io", ["K", "tau", "theta"]), ("physics", ["Ua", "Cp", "theta"])]:
        estimates = np.asarray([draw["estimated_"+coordinates] for draw in complete])
        truth = np.asarray(complete[0]["truth_"+coordinates]) if complete else None
        for method in methods:
            for j, parameter in enumerate(names):
                coverage, widths, delta_sd = [], [], []
                for draw in complete:
                    interval = draw["local_intervals_"+coordinates][method]
                    if interval["covered"] is None:
                        continue
                    lower, upper, actual = interval["lower"][j], interval["upper"][j], draw["truth_"+coordinates][j]
                    covered = lower <= actual <= upper
                    if covered != interval["covered"][j] or lower <= 0 or upper < lower:
                        raise RuntimeError("Saved local interval coverage is inconsistent")
                    coverage.append(covered)
                    widths.append(upper-lower)
                    log_sd = (np.log(upper)-np.log(lower))/(2*norm.ppf(.975))
                    delta_sd.append(draw["estimated_"+coordinates][j]*log_sd)
                count = coverage_counts(sum(coverage), len(coverage), requested)
                bias = float(estimates[:, j].mean()-truth[j]) if len(complete) else None
                empirical_sd = float(estimates[:, j].std(ddof=1)) if len(complete) > 1 else None
                rms_sd = float(np.sqrt(np.mean(np.square(delta_sd)))) if delta_sd else None
                rows.append({"case": case, "true_noise": noise, "coordinates": coordinates, "parameter": parameter,
                             "method": method, **count, **descriptive(widths), "completed_draws": len(complete),
                             "failed_draws": failed, "missing_draws": missing,
                             "undefined_on_completed_draws": len(complete)-len(coverage),
                             "truth": None if truth is None else float(truth[j]),
                             "bias_when_draw_complete": bias, "empirical_sd_when_draw_complete": empirical_sd,
                             "rms_reported_delta_sd_when_defined": rms_sd,
                             "rms_sd_over_empirical_sd": None if not empirical_sd or rms_sd is None else rms_sd/empirical_sd})
    for name in ("known_sigma_iid", "estimated_sigma_iid"):
        widths, envelope, coverage, counts, open_sets, changes = [], [], [], [], [], []
        for draw in complete:
            profile = draw["delay_profile"]
            components = profile["intervals_"+("known" if name == "known_sigma_iid" else "unknown")+"_sigma_iid"]
            truth = draw["truth_io"][2]
            covered = any(item["lower"] <= truth <= item["upper"] for item in components)
            saved = draw["delay_profile_coverage"][name]
            if covered != saved["covered"]:
                raise RuntimeError("Saved profile coverage is inconsistent")
            coverage.append(covered)
            widths.append(sum(item["upper"]-item["lower"] for item in components))
            envelope.append(max(item["upper"] for item in components)-min(item["lower"] for item in components) if components else 0.)
            counts.append(len(components))
            open_sets.append(any("open_at_exploration_bound" in [item["lower_status"], item["upper_status"]] for item in components))
            statistic = "lr_"+("known" if name == "known_sigma_iid" else "unknown")+"_sigma_iid"
            quadratic = quadratic_grid_statistic([point["log_value"] for point in profile["points"]],
                                                  [point[statistic] for point in profile["points"]], np.log(truth))
            if quadratic is not None:
                changes.append((quadratic <= profile["chi2_95_df1"]) != covered)
        rows.append({"case": case, "true_noise": noise, "coordinates": "io", "parameter": "theta",
                     "method": "profile_"+name, **coverage_counts(sum(coverage), len(coverage), requested),
                     **descriptive(widths), "completed_draws": len(complete), "failed_draws": failed, "missing_draws": missing,
                     "undefined_on_completed_draws": 0, "mean_envelope_width": float(np.mean(envelope)) if envelope else None,
                     "mean_component_count": float(np.mean(counts)) if counts else None,
                     "open_exploration_sets": sum(open_sets), "quadratic_interpolation_defined": len(changes),
                     "quadratic_interpolation_coverage_class_changes": sum(changes),
                     "width_notice": "Primary width sums disconnected components; envelope width includes excluded gaps. The three-point quadratic diagnostic only tests grid-interpolation sensitivity and is not a refined nuisance profile or a guarantee of endpoint accuracy."})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=Path, default=HERE/"results/coverage")
    parser.add_argument("--fits", type=Path, default=HERE/"results/fits")
    parser.add_argument("--config", type=Path, default=HERE/"coverage_config.json")
    parser.add_argument("--output", type=Path, default=HERE/"results/coverage_summary")
    args = parser.parse_args()
    manifest_path = args.coverage/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(args.config.read_text())
    if manifest["status"] not in ("complete", "complete_with_failures") or len(manifest["completed"])+len(manifest["failed"]) != manifest["expected_chunks"]:
        raise RuntimeError("Do not summarize a still-running or interrupted coverage run")
    if sha(args.config) != manifest["identity"]["config_sha256"] or manifest["identity"]["replicate_override"] is not None or manifest["filter"]:
        raise RuntimeError("Production summary requires the complete prespecified configuration, without a trial override")
    if manifest["expected_draws"] != len(config["cases"])*len(config["noise_scenarios"])*config["replicates"]:
        raise RuntimeError("Coverage run differs from the declared comparison matrix")
    requested, inputs, rows, failures = config["replicates"], {str(manifest_path): sha(manifest_path), str(args.config): sha(args.config)}, [], []
    maxima = {"known_lr_arithmetic": 0., "unknown_lr_arithmetic": 0., "physical_transform_relative": 0.,
              "fit_sse_minus_best_converged_absolute_C2": 0.}
    attempts = {"fit_total": 0, "fit_failed": 0, "profile_total": 0, "profile_failed": 0,
                "profile_revisited_minima_draws": 0, "profile_retry_errors_draws": 0}
    for case in config["cases"]:
        fit_path = args.fits/(case+".json")
        if sha(fit_path) != manifest["input_fit_sha256"][fit_path.name]:
            raise RuntimeError("Coverage generating fit changed")
        inputs[str(fit_path)] = sha(fit_path)
        fit = json.loads(fit_path.read_text())["fit"]
        truth = np.array([fit["parameters"][name] for name in ("K", "tau", "theta")])
        for noise_name, noise in config["noise_scenarios"].items():
            identity = hashlib.sha256(json.dumps({"identity": manifest["identity"], "fit_sha256": sha(fit_path),
                                                 "noise_name": noise_name, "target": requested}, sort_keys=True).encode()).hexdigest()
            folder = args.coverage/"draws"/(case+"__"+noise_name)
            expected_names = {f"{replicate:04d}.json.gz" for replicate in range(requested)}
            if any(path.name not in expected_names for path in folder.glob("*.json.gz")):
                raise RuntimeError("Unexpected coverage draw outside the prespecified replicate range")
            draws, seeds = [], set()
            for replicate in range(requested):
                path = folder/(f"{replicate:04d}.json.gz")
                if not path.exists():
                    failures.append({"case": case, "true_noise": noise_name, "replicate": replicate, "status": "missing_after_terminal_run"})
                    continue
                inputs[str(path)] = sha(path)
                draw = json.loads(gzip.decompress(path.read_bytes()))
                if draw["identity_sha256"] != identity or draw["case"] != case or draw["noise_name"] != noise_name or draw["replicate"] != replicate:
                    raise RuntimeError("Coverage draw identity mismatch")
                draws.append(draw)
                if draw["status"] != "complete":
                    failures.append({"case": case, "true_noise": noise_name, "replicate": replicate,
                                     "status": draw["status"], "error": draw.get("error")})
                    continue
                expected_seed = replicate_seed(case, noise_name, replicate, config["seed"])
                if draw["seed"] != expected_seed or expected_seed in seeds or draw["noise"] != noise or not np.array_equal(draw["truth_io"], truth):
                    raise RuntimeError("Coverage truth/noise/seed mismatch")
                seeds.add(expected_seed)
                if draw["delay_profile"]["baseline_requires_revisit"]:
                    raise RuntimeError("Coverage profile has an unresolved lower minimum")
                fit_attempts = draw["fit"]["starts"]
                attempts["fit_total"] += len(fit_attempts)
                attempts["fit_failed"] += sum(not attempt["success"] for attempt in fit_attempts)
                attempts["profile_revisited_minima_draws"] += bool(draw["revisited_minima"])
                attempts["profile_retry_errors_draws"] += bool(draw["profile_retry_errors"])
                converged = [attempt["sse"] for attempt in fit_attempts if attempt["success"]]
                if not converged:
                    raise RuntimeError("Completed coverage fit has no converged start")
                maxima["fit_sse_minus_best_converged_absolute_C2"] = max(maxima["fit_sse_minus_best_converged_absolute_C2"],
                                                                         abs(draw["fit"]["sse"]-min(converged)))
                for key in ("truth", "estimated"):
                    K, tau, theta = draw[key+"_io"]
                    expected = np.array([.032/K, .032*tau/K, theta])
                    maxima["physical_transform_relative"] = max(maxima["physical_transform_relative"], float(np.max(abs(np.asarray(draw[key+"_physics"])/expected-1))))
                base, n, sigma = draw["fit"]["sse"], draw["fit"]["n"], noise["sigma_C"]
                for point in draw["delay_profile"]["points"]:
                    attempts["profile_total"] += len(point["attempts"])
                    attempts["profile_failed"] += sum(not attempt["success"] for attempt in point["attempts"])
                    maxima["known_lr_arithmetic"] = max(maxima["known_lr_arithmetic"], abs((point["sse"]-base)/sigma**2-point["lr_known_sigma_iid"]))
                    maxima["unknown_lr_arithmetic"] = max(maxima["unknown_lr_arithmetic"], abs(n*np.log(point["sse"]/base)-point["lr_unknown_sigma_iid"]))
            verify_chunk_counts(draws, case, noise_name, requested, config["chunk_size"], manifest)
            rows.extend(summarize_scenario(draws, case, noise_name, requested))
    if any(value > 1e-8 for value in maxima.values()):
        raise RuntimeError("Coverage arithmetic audit failed: "+json.dumps(maxima))
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output/"coverage_summary.csv", index=False)
    save_json(args.output/"audit.json", {"status": "passed", "requested_draws": manifest["expected_draws"],
                                        "summary_sha256": sha(args.output/"coverage_summary.csv"),
                                        "failed_or_missing_draws": len(failures), "failures": failures,
                                        "summary_rows": len(rows), "maximum_arithmetic_errors": maxima,
                                        "optimizer_attempts_on_completed_draws": attempts,
                                        "input_sha256": inputs, "producer_sha256": sha(HERE/"summarize_coverage.py"),
                                        "helper_sha256": {name: sha(HERE/name) for name in ("bootstrap.py", "coverage_study.py", "run_fits.py")},
                                        "notice": "Method-defined denominators, failed/missing/undefined outcomes, unconditional worst/best bounds and binomial Monte Carlo uncertainty are explicit. These simulations condition on four true model points, the recorded input/times, known initial temperature, ambient and heater calibration. They do not validate real-data mean discrepancy or infer universal coverage. Profile results retain finite-grid interpolation; its quadratic sensitivity is not a global or numerical-refinement certificate."})
    print(json.dumps({"status": "passed", "summary_rows": len(rows), "failed_or_missing_draws": len(failures), "maximum_arithmetic_errors": maxima}))


if __name__ == "__main__":
    main()
