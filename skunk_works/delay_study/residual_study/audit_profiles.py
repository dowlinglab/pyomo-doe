"""Independently audit profile minima/crossings and near-tied flagged attempts.

Original scans remain immutable. A stationary saved attempt can reconcile a
numerical near tie only within 1e-8 NLL and after dense derivative checks.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from audit_mean import reference as independent_mean
from covariance_fit import covariance_bounds, projected_gradient
from data import load
from fit_diagnostics import evaluate
from run_mean import write_json


def reconstructed(reference, attempt):
    x = attempt["coordinates"]
    return dict(reference, best={"tau": float(np.exp(x[0])), "theta": float(np.exp(x[1])),
                "coefficients": attempt["coefficients"], "sigma_total_ML_C": attempt["sigma_total_ML_C"],
                "noise_parameters": [float(np.exp(x[2])), x[3]],
                "negative_log_likelihood": attempt["negative_log_likelihood"]})


def dense_gradient(records, fit, parameter, config):
    b = fit["best"]
    length, eta = b["noise_parameters"]
    q, n = 0., 0
    pieces = []
    for j, data in enumerate(records):
        delta = [b["coefficients"][j+1]]*2
        y = independent_mean(data, b["coefficients"][0], b["tau"], b["theta"], delta, fit["hold"])[1:]
        residual = data.temperature[1:]-y
        distance = np.abs(data.time[1:, None]-data.time[None, 1:])/length
        exponential = np.exp(-distance)
        R = (1-eta)*exponential+eta*np.eye(len(residual))
        factor = cho_factor(R, lower=True)
        alpha = cho_solve(factor, residual)
        q += residual@alpha
        n += len(residual)
        shape_derivatives = [(1-eta)*distance*exponential, np.eye(len(residual))-exponential]
        covariance_parts = [(float(np.trace(cho_solve(factor, dR))), float(alpha@dR@alpha)) for dR in shape_derivatives]
        mean_parts = []
        for name in ("tau", "theta"):
            values = []
            for h in (1e-4, 5e-5):
                low, high = dict(b), dict(b)
                low[name] *= np.exp(-h); high[name] *= np.exp(h)
                ml = independent_mean(data, b["coefficients"][0], low["tau"], low["theta"], delta, fit["hold"])[1:]
                mh = independent_mean(data, b["coefficients"][0], high["tau"], high["theta"], delta, fit["hold"])[1:]
                values.append(float(((mh-ml)/(2*h))@alpha))
            mean_parts.append(values)
        pieces.append((covariance_parts, mean_parts))
    gradient = np.zeros(4)
    gradient_coarse = np.zeros(2)
    for covariance_parts, mean_parts in pieces:
        for k, (trace, quadratic) in enumerate(covariance_parts):
            gradient[2+k] += .5*(trace-n/q*quadratic)
        for k, values in enumerate(mean_parts):
            gradient[k] -= n/q*values[1]
            gradient_coarse[k] -= n/q*values[0]
    bounds = [tuple(np.log(config["thermal_bounds"][k])) for k in ("tau", "theta")]+covariance_bounds("ou_nugget")
    x = np.r_[np.log([b["tau"], b["theta"], length]), eta]
    free = [i for i in range(4) if i != {"tau": 0, "theta": 1}.get(parameter)]
    pg = projected_gradient(x[free], gradient[free], [bounds[i] for i in free])
    return {"gradient_logtau_logtheta_logell_eta": gradient.tolist(),
            "free_indices": free, "projected_max_abs": float(np.max(np.abs(pg))),
            "thermal_step_max_abs_disagreement": float(np.max(np.abs(gradient[:2]-gradient_coarse))),
            "method": "Independent augmented-exponential mean derivatives and exact dense Gaussian covariance derivatives; envelope derivative at saved bounded linear optimum."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.matrix/"manifest.json").read_text())
    if manifest["status"] not in {"complete", "complete_with_issues"}:
        raise ValueError("Profile matrix is not terminal")
    results = []
    for pair in ["P"+str(i) for i in range(1, 7)]:
        ref_path = args.references/pair/"fit.json"
        ref = json.loads(ref_path.read_text())
        config = json.loads((args.references/pair/"effective_config.json").read_text())
        records = [load(n) for n in ref["record_names"]]
        for parameter in ("K", "tau", "theta"):
            path = args.matrix/(pair+"_"+parameter)/"profile.json"
            original = json.loads(path.read_text())
            if original["binding"]["reference_fit_sha256"] != hashlib.sha256(ref_path.read_bytes()).hexdigest():
                raise ValueError("Reference fit binding changed")
            selected = []
            selected.append(min(original["points"], key=lambda r: r["best"]["negative_log_likelihood"] if r["best"] else np.inf))
            for crossing in original["refined_crossings"]:
                for value in crossing["log_bisection_bracket"]:
                    selected.append(min(original["points"], key=lambda r: abs(r["value"]-value)))
            selected.extend(r for r in original["points"] if r["status"] != "complete")
            audited, reconciled, unresolved = [], [], []
            for row in {r["value"]: r for r in selected}.values():
                if row["best"] is None:
                    unresolved.append(row["value"])
                    continue
                diagnostic, _ = evaluate(records, reconstructed(ref, row["best"]), config)
                item = {"value": row["value"], "original_status": row["status"],
                        "twice_nll_difference": row["twice_nll_difference"],
                        "independent_likelihood_audit": diagnostic["audit"]}
                if row["status"] != "complete":
                    candidates = [a for a in row["attempts"] if a.get("success") and
                                  a["projected_gradient_max_abs"] <= .05 and
                                  abs(a["negative_log_likelihood"]-row["best"]["negative_log_likelihood"]) <= 1e-8]
                    if candidates:
                        candidate = min(candidates, key=lambda a: a["projected_gradient_max_abs"])
                        restored = reconstructed(ref, candidate)
                        check, _ = evaluate(records, restored, config)
                        gradient = dense_gradient(records, restored, parameter, config)
                        item.update(candidate=copy.deepcopy(candidate), independent_candidate_likelihood_audit=check["audit"],
                                    independent_candidate_gradient=gradient,
                                    candidate_nll_difference=candidate["negative_log_likelihood"]-row["best"]["negative_log_likelihood"])
                        if gradient["projected_max_abs"] <= .05 and gradient["thermal_step_max_abs_disagreement"] <= .005:
                            reconciled.append(row["value"])
                        else:
                            unresolved.append(row["value"])
                    else:
                        unresolved.append(row["value"])
                audited.append(item)
            results.append({"pair": pair, "parameter": parameter, "source_profile_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "lower_minimum_found": original["lower_minimum_found"], "audits": audited,
                            "near_tied_stationary_attempts_reconciled": reconciled, "unresolved_points": unresolved,
                            "included_grid_components": original["included_grid_components"],
                            "refined_crossings": original["refined_crossings"]})
            print(pair, parameter, "reconciled", len(reconciled), "unresolved", len(unresolved), flush=True)
    write_json(args.output/"audit.json", {"status": "requires_revisit" if any(r["unresolved_points"] or r["lower_minimum_found"] for r in results) else "passed",
               "curves": results, "notice": "Original profile results and flags preserved. Reconciliation uses an already saved stationary attempt within 1e-8 NLL and independently checked dense derivatives; no objective, bounds, cutoff or stationarity tolerance is relaxed. Finite scans are not global or coverage certificates."})


if __name__ == "__main__":
    main()
