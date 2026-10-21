"""Select all completed profiles and audit numerical/structural interpretation.

The analytic family check deliberately challenges interpolated endpoints. A
finite grid is not an exact likelihood set, even when its nuisance fits converge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from run_fits import save_json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contains(components, value):
    return any(component["lower"] <= value <= component["upper"] for component in components)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE/"results")
    args = parser.parse_args()
    root = args.results
    first = json.loads((root/"profiles/manifest.json").read_text())
    retry = json.loads((root/"profile_retries/manifest.json").read_text())
    if len(first["completed"])+len(first["failed"]) != first["expected_jobs"] or retry["status"] != "complete":
        raise RuntimeError("Profile manifests incomplete")
    family_path = root/"structural/equivalence_curves.csv"
    family = pd.read_csv(family_path)
    family = family[family.within_original_fit_box]
    inputs = {str(family_path): sha(family_path)}
    rows, index, family_issues = [], {}, []
    maxima = {"known_lr_absolute_error": 0., "unknown_lr_absolute_error": 0.,
              "selected_sse_minus_best_converged_absolute_error_C2": 0., "fixed_parameter_relative_error": 0.}
    for fit_path in sorted((root/"fits").glob("*.json")):
        record = json.loads(fit_path.read_text())
        fit, uncertainty = record["fit"], record["uncertainty"]
        spec = fit["spec"]
        inputs[str(fit_path)] = sha(fit_path)
        for j, name in enumerate(uncertainty["parameters"]):
            filename = fit_path.stem+"__"+name+".json"
            job_key = fit_path.stem+"__"+str(j)
            retried = (root/"profile_retries"/filename).exists()
            if retried and job_key not in first["failed"]:
                raise RuntimeError("An unrequested profile replacement appeared: "+filename)
            path = root/("profile_retries" if retried else "profiles")/filename
            saved = json.loads(path.read_text())
            if saved["fit_sha256"] != sha(fit_path):
                raise RuntimeError("Profile baseline input changed")
            p = saved["profile"]
            if p["parameter_index"] != j or p["profiled_parameter"] != name or p["baseline_sse"] != fit["sse"]:
                raise RuntimeError("Profile identity mismatch")
            if p["baseline_requires_revisit"]:
                raise RuntimeError("A lower baseline minimum must be resolved first")
            grid = np.array([point["parameter_value"] for point in p["points"]])
            log_grid = np.array([point["log_value"] for point in p["points"]])
            # Distinct adjacent log values can round to the same native double
            # after exp near an exactly repeated center or exchanged mode.
            if not np.all(np.diff(log_grid) > 0) or np.any(np.diff(grid) < 0) or len(grid) != p["point_count"]:
                raise RuntimeError("Profile grid unordered or incomplete: "+filename)
            if abs(p["chi2_95_df1"]-chi2.ppf(.95, 1)) > 1e-12:
                raise RuntimeError("Likelihood threshold changed")
            for point in p["points"]:
                converged = [a["sse"] for a in point["attempts"] if a["success"]]
                if not converged:
                    raise RuntimeError("Profile selected an unconverged nuisance point")
                maxima["selected_sse_minus_best_converged_absolute_error_C2"] = max(
                    maxima["selected_sse_minus_best_converged_absolute_error_C2"], abs(point["sse"]-min(converged)))
                maxima["fixed_parameter_relative_error"] = max(maxima["fixed_parameter_relative_error"],
                                                                 abs(point["parameters"][j]/point["parameter_value"]-1))
                known = (point["sse"]-fit["sse"])/uncertainty["known_sigma_C"]**2
                unknown = fit["n"]*np.log(point["sse"]/fit["sse"])
                maxima["known_lr_absolute_error"] = max(maxima["known_lr_absolute_error"], abs(known-point["lr_known_sigma_iid"]))
                maxima["unknown_lr_absolute_error"] = max(maxima["unknown_lr_absolute_error"], abs(unknown-point["lr_unknown_sigma_iid"]))
            nonestimable = name in uncertainty["nonestimable_native_parameters"]
            issue = False
            if spec["family"] == "original" and name in ("Ub", "CpH", "CpS"):
                members = family[family.calibration == record["calibration"]]
                for statistic in ("known", "unknown"):
                    intervals = p["intervals_"+statistic+"_sigma_iid"]
                    excluded = members[~members[name].map(lambda value: contains(intervals, value))]
                    if len(excluded):
                        issue = True
                        family_issues.append({"profile": filename, "noise_scale": statistic, "excluded_exact_family_samples": len(excluded),
                                              "largest_equivalent_prediction_difference_C": float(excluded.maximum_prediction_difference_C.max()),
                                              "excluded_parameter_sample_minimum": float(excluded[name].min()),
                                              "excluded_parameter_sample_maximum": float(excluded[name].max()),
                                              "cause": "Coarse interpolation of a flat profile and its rising edge can exclude known exact equivalents; selected pointwise minima alone do not validate an interpolated confidence-set boundary.",
                                              "action": "Withhold a physical-parameter confidence claim. Use the analytic equivalence class; refine numerical crossings before using their endpoints."})
            index[filename] = {"path": str(path.relative_to(root)), "sha256": sha(path), "retried": retried,
                               "nonestimable_in_recorded_experiment": nonestimable,
                               "analytic_family_inclusion_failed": issue,
                               "interval_interpretation": "undefined_as_a_unique_physical_parameter_confidence_claim" if nonestimable else "conditional_asymptotic_IID_grid_profile_set"}
            rows.append({"case": fit_path.stem, "calibration": record["calibration"], **spec, "parameter": name,
                         "point_count": p["point_count"], "retried": retried, "nonestimable": nonestimable,
                         "grid_endpoint_refinement_required": issue,
                         "known_sigma_components": json.dumps(p["intervals_known_sigma_iid"], separators=(",", ":")),
                         "unknown_sigma_components": json.dumps(p["intervals_unknown_sigma_iid"], separators=(",", ":"))})
    if len(rows) != first["expected_jobs"]:
        raise RuntimeError("Merged profile catalog incomplete")
    limits = {"known_lr_absolute_error": 1e-9, "unknown_lr_absolute_error": 1e-9,
              "selected_sse_minus_best_converged_absolute_error_C2": 1e-9, "fixed_parameter_relative_error": 1e-12}
    passed = all(maxima[key] <= limit for key, limit in limits.items())
    pd.DataFrame(rows).to_csv(root/"profile_summary.csv", index=False)
    save_json(root/"profile_index.json", {"profiles": index, "producer_sha256": sha(HERE/"summarize_profiles.py"),
                                          "first_pass_manifest_sha256": sha(root/"profiles/manifest.json"),
                                          "retry_manifest_sha256": sha(root/"profile_retries/manifest.json")})
    save_json(root/"profile_audit.json", {"arithmetic_status": "passed" if passed else "failed", "profiles": len(rows),
                                          "recovered_jobs": sum(row["retried"] for row in rows),
                                          "maximum_arithmetic_errors": maxima, "tolerances": limits, "input_sha256": inputs,
                                          "structural_endpoint_issues": family_issues,
                                          "notice": "Arithmetic and selected convergence checks pass separately from endpoint interpretation. No finite-grid analysis is a global-minimum certificate. Structural ambiguity and serial residuals prevent routine nominal confidence claims."})
    print(json.dumps({"arithmetic_status": "passed" if passed else "failed", "profiles": len(rows),
                      "recovered_jobs": sum(row["retried"] for row in rows),
                      "profiles_failing_exact_family_endpoint_inclusion": sum(row["grid_endpoint_refinement_required"] for row in rows),
                      "maximum_arithmetic_errors": maxima}))
    return not passed


if __name__ == "__main__":
    raise SystemExit(main())
