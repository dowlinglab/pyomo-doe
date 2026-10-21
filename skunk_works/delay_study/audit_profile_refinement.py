"""Audit the selected refinement and publish a separate 300-profile catalog.

Numerical endpoints remain conditional asymptotic profile sets. Algebraic family
inclusion is a necessary check, not evidence of unique physical estimability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from data import load
from inference import ModelSpec, parameter_box, residuals
from models import original_transfer
from original_family_bounds import parameter_extrema
from profiles import interval_components
from run_fits import save_json
from run_profile_refinement import selected_names

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def includes_component(accepted, compatible, tolerance=1e-8):
    """A whole closed compatibility interval must fit in one accepted component."""
    left, right = compatible
    return any(item["lower"] <= left+tolerance*max(1., abs(left)) and
               item["upper"] >= right-tolerance*max(1., abs(right)) for item in accepted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--refinement", type=Path)
    args = parser.parse_args()
    root = args.results
    refined = args.refinement or root/"profile_refinement"
    manifest = json.loads((refined/"manifest.json").read_text())
    catalog_path = root/"profile_index.json"
    old_catalog = json.loads(catalog_path.read_text())
    if manifest["status"] != "complete" or manifest["failed"] or set(manifest["completed"]) != set(selected_names()):
        raise RuntimeError("A complete selected refinement is required before publishing a catalog")
    if manifest["identity"]["source_catalog_sha256"] != sha(catalog_path):
        raise RuntimeError("Input catalog changed")
    catalog = json.loads(json.dumps(old_catalog))
    errors = {"likelihood_arithmetic": 0., "selected_attempt_sse_C2": 0.,
              "fixed_parameter_relative": 0., "new_point_forward_sse_C2": 0.,
              "witness_transfer_relative": 0.}
    checks, inputs = [], {str(catalog_path): sha(catalog_path), str(refined/"manifest.json"): sha(refined/"manifest.json")}
    for name in selected_names():
        path = refined/name
        saved = json.loads(path.read_text())
        old_entry = old_catalog["profiles"][name]
        old_path = root/old_entry["path"]
        previous = json.loads(old_path.read_text())["profile"]
        fit_path = root/"fits"/saved["fit_file"]
        record = json.loads(fit_path.read_text())
        if saved["identity"] != manifest["identity"] or saved["inputs_sha256"] != {"fit": sha(fit_path), "profile": sha(old_path)}:
            raise RuntimeError("Refinement input identity mismatch: "+name)
        p = saved["profile"]
        if p["refinement_requires_revisit"] or p["baseline_requires_revisit"]:
            raise RuntimeError("Unresolved refinement flag: "+name)
        spec = ModelSpec(**record["fit"]["spec"])
        j, baseline, count = p["parameter_index"], record["fit"]["sse"], record["fit"]["n"]
        sigma = record["uncertainty"]["known_sigma_C"]
        lo, hi = parameter_box(spec)
        old_log_values = {row["log_value"] for row in previous["points"]}
        data = [load(item) for item in record["datasets"]]
        grid = np.array([point["log_value"] for point in p["points"]])
        if len(grid) != p["point_count"] or not np.all(np.diff(grid) > 0):
            raise RuntimeError("Refined profile grid is incomplete or unordered")
        for point in p["points"]:
            parameters = np.asarray(point["parameters"])
            if np.any(parameters < lo*(1-1e-8)) or np.any(parameters > hi*(1+1e-8)):
                raise RuntimeError("Selected profile point violates its numerical box")
            errors["fixed_parameter_relative"] = max(errors["fixed_parameter_relative"], abs(parameters[j]/point["parameter_value"]-1))
            values = [(point["sse"]-baseline)/sigma**2, count*np.log(point["sse"]/baseline)]
            for statistic, value in zip(("known", "unknown"), values):
                errors["likelihood_arithmetic"] = max(errors["likelihood_arithmetic"], abs(value-point["lr_"+statistic+"_sigma_iid"]))
            accepted = [attempt["sse"] for attempt in point["attempts"] if attempt["success"]]
            if not accepted:
                raise RuntimeError("Selected profile point lacks a converged solve or verified feasible witness")
            errors["selected_attempt_sse_C2"] = max(errors["selected_attempt_sse_C2"], abs(point["sse"]-min(accepted)))
            if point["log_value"] not in old_log_values or point.get("selected_kind") == "verified_equivalence_witness":
                r = residuals(np.log(parameters), data, spec)
                errors["new_point_forward_sse_C2"] = max(errors["new_point_forward_sse_C2"], abs(float(r@r)-point["sse"]))
            for attempt in point["attempts"]:
                if attempt.get("kind") == "verified_equivalence_witness":
                    reference = [record["fit"]["parameters"][key] for key in spec.names]
                    transfer = original_transfer(np.asarray(attempt["parameters"]))
                    errors["witness_transfer_relative"] = max(errors["witness_transfer_relative"], float(np.max(abs(transfer/original_transfer(reference)-1))))
        family = None
        if spec.family == "original":
            reference = [record["fit"]["parameters"][key] for key in spec.names]
            family = parameter_extrema(reference, j, lo, hi)
        for statistic in ("known", "unknown"):
            recomputed = interval_components(grid, [row["lr_"+statistic+"_sigma_iid"] for row in p["points"]], p["chi2_95_df1"], np.log(lo[j]), np.log(hi[j]))
            if recomputed != p["intervals_"+statistic+"_sigma_iid"]:
                raise RuntimeError("Saved intervals differ from independently reconstructed grid crossings")
            if family and not all(includes_component(recomputed, component) for component in family["compatible_parameter_components"]):
                raise RuntimeError("Refined profile still excludes an analytically compatible interval: "+name)
        inputs[str(path)] = sha(path)
        catalog["profiles"][name].update(path=str(path.relative_to(root)), sha256=sha(path), refined=True,
                                         analytic_family_inclusion_failed=False,
                                         source_first_pass_path=old_entry["path"])
        checks.append({"profile": name, "source_points": len(previous["points"]), "selected_points": len(p["points"]),
                       "crossings": p["refinement_crossings"], "analytic_compatible_components": None if family is None else family["compatible_parameter_components"],
                       "known_intervals": p["intervals_known_sigma_iid"], "unknown_intervals": p["intervals_unknown_sigma_iid"],
                       "interpretation": old_entry["interval_interpretation"]})
    tolerances = {"likelihood_arithmetic": 1e-9, "selected_attempt_sse_C2": 1e-9,
                  "fixed_parameter_relative": 1e-12, "new_point_forward_sse_C2": 1e-7,
                  "witness_transfer_relative": 1e-8}
    if any(errors[key] > tolerances[key] for key in errors):
        raise RuntimeError("Refined profile arithmetic/forward audit failed: "+json.dumps(errors))
    audit = {"status": "passed", "refined_profiles": len(checks), "maximum_errors": errors,
             "tolerances": tolerances, "profiles": checks, "input_sha256": inputs,
             "notice": "Verified arithmetic, new-point forward SSE, and complete inclusion of the fitted-transfer equivalence components inside the declared box. No claim of unique physical confidence, global nuisance minima, or absence of unobserved modes. First-pass catalog and files remain preserved."}
    save_json(root/"profile_refinement_audit.json", audit)
    catalog["refinement_manifest_sha256"] = sha(refined/"manifest.json")
    catalog["refinement_audit_sha256"] = sha(root/"profile_refinement_audit.json")
    catalog["selection_notice"] = "Eighteen audited refinements plus 282 preserved first-pass selections; use the first-pass catalog to reproduce earlier draft figures."
    save_json(root/"selected_profile_index.json", catalog)
    print(json.dumps({"status": "passed", "selected_profiles": len(catalog["profiles"]), "refined_profiles": len(checks), "maximum_errors": errors}))


if __name__ == "__main__":
    main()
