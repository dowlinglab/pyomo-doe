"""Regenerate every diagnostic replicate and independently recompute searches."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np

from covariance import Whitening, profiled_likelihood
from mean import fitted_records
import refit_bootstrap
from run_mean import write_json


def independent_statistics(records, fit):
    predictions = fitted_records(records, fit)
    residual = np.concatenate([(d.temperature-p)[1:] for d, p in zip(records, predictions)])
    W = Whitening([d.time[1:] for d in records], fit["kernel"], fit["best"]["noise_parameters"])
    nll, sigma, e = profiled_likelihood(residual, W)
    rows = []
    cursor = 0
    for data in records:
        n = len(data.time)-1
        innovation = e[cursor:cursor+n]
        centered = innovation-innovation.mean()
        covariance = np.correlate(centered, centered, mode="full")[n-1:]
        acf = covariance[1:31]/covariance[0]
        command = data.input[1:]
        candidates = []
        for lag in range(-30, 31):
            a, b = (innovation[lag:], command[:-lag]) if lag > 0 else ((innovation[:lag], command[-lag:]) if lag < 0 else (innovation, command))
            if np.ptp(b) == 0:
                continue
            count = len(a)
            numerator = np.sum(a*b)-np.sum(a)*np.sum(b)/count
            denominator = np.sqrt((np.sum(a*a)-np.sum(a)**2/count)*(np.sum(b*b)-np.sum(b)**2/count))
            candidates.append(float(abs(numerator/denominator)))
        Q = float(n*(n+2)*np.sum(acf*acf/(n-np.arange(1, 31))))
        rows.append({"record": data.name, "acf1": float(acf[0]),
                     "max_abs_acf_1_to_30": float(np.max(np.abs(acf))),
                     "max_abs_input_correlation": max(candidates) if candidates else None,
                     "ljung_box_Q_30": Q})
        cursor += n
    return {"per_run": rows, "global_max_abs_acf": max(r["max_abs_acf_1_to_30"] for r in rows),
            "global_max_abs_input_correlation": max(r["max_abs_input_correlation"] for r in rows if r["max_abs_input_correlation"] is not None),
            "global_max_Q30": max(r["ljung_box_Q_30"] for r in rows)}, nll, sigma


def difference(first, second):
    errors = []
    for name in ("global_max_abs_acf", "global_max_abs_input_correlation", "global_max_Q30"):
        errors.append(abs(first[name]-second[name]))
    for a, b in zip(first["per_run"], second["per_run"]):
        if a["record"] != b["record"]:
            raise AssertionError("Run order differs")
        for name in ("acf1", "max_abs_acf_1_to_30", "max_abs_input_correlation", "ljung_box_Q_30"):
            if a[name] is None or b[name] is None:
                if a[name] != b[name]:
                    raise AssertionError("Undefined command correlation differs")
            else:
                errors.append(abs(a[name]-b[name]))
    return max(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for folder in args.input:
        checkpoint_path = folder/"checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text())
        summary_path = folder/"summary.json"
        summary = json.loads(summary_path.read_text())
        identity = checkpoint["identity"]
        if checkpoint["status"] != "complete" or sorted(d["index"] for d in checkpoint["draws"]) != list(range(identity["draws"])):
            raise AssertionError("Diagnostic draws missing or duplicated")
        refit_bootstrap.initialize(args.reference, identity["starts_per_refit"], identity["measurement"], identity["seed_base"], identity.get("block_length", 60))
        S = refit_bootstrap.STATE
        observed, nll, _ = independent_statistics(S["records"], S["reference"])
        observed_saved = json.loads((folder/"observed.json").read_text())
        if difference(observed, observed_saved) > 1e-7:
            raise AssertionError("Observed diagnostic arithmetic failed")
        maximum_statistic_error = 0.
        maximum_nll_error = 0.
        for draw in checkpoint["draws"]:
            if draw["status"] != "complete":
                raise AssertionError("Declared complete calibration includes an unreviewed failed/flagged draw")
            seed, data = refit_bootstrap.synthetic(draw["index"])
            if seed != draw["seed"]:
                raise AssertionError("Seed mismatch")
            stats, nll, sigma = independent_statistics(data, draw["fit"])
            err = difference(stats, draw["statistics"])
            objective_error = abs(nll-draw["fit"]["best"]["negative_log_likelihood"])
            scale_error = abs(sigma-draw["fit"]["best"]["sigma_total_ML_C"])
            if err > 1e-7 or objective_error > 2e-5 or scale_error > 1e-7:
                raise AssertionError((folder.name, draw["index"], err, objective_error, scale_error))
            maximum_statistic_error = max(maximum_statistic_error, err)
            maximum_nll_error = max(maximum_nll_error, objective_error)
        for name in ("global_max_abs_acf", "global_max_abs_input_correlation", "global_max_Q30"):
            values = [d["statistics"][name] for d in checkpoint["draws"]]
            fraction = (1+sum(v >= observed[name] for v in values))/(len(values)+1)
            saved = summary["global_comparisons"][name]
            if abs(fraction-saved["tail_fraction_plus_one"]) > 1e-15 or np.max(np.abs(np.quantile(values, [.5, .95, .99])-saved["simulation_quantiles_50_95_99"])) > 1e-12:
                raise AssertionError("Calibration summary arithmetic mismatch")
        receipts.append({"mechanism": identity["measurement"], "draws": len(checkpoint["draws"]),
                         "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                         "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                         "max_statistic_error": maximum_statistic_error, "max_nll_error": maximum_nll_error})
        print(folder.name, "audited", len(checkpoint["draws"]), flush=True)
    write_json(args.output/"audit.json", {"status": "passed", "mechanisms": receipts,
               "scope": "Every saved draw regenerated from its seed; independent correlation/Q arithmetic, likelihood/scale and summary checks. Filter likelihood is separately dense-audited on selected datasets. Not global optimization, scientific adequacy or coverage proof."})


if __name__ == "__main__":
    main()
