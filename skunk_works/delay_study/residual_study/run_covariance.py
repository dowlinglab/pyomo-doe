"""Run one pooled ambient/covariance cell, preserving iteration and diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from scipy.stats import chi2, kurtosis, skew

from audit_covariance import audit
from audit_saved_mean import acf
from covariance import profiled_likelihood
from covariance_fit import alternating_fit, joint_fit, projected_state
from data import load
from mean import fitted_records
from run_mean import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--mode", choices=["constant", "linear"], required=True)
    parser.add_argument("--kernel", choices=["iid", "ou", "ou_nugget", "matern32_nugget"], required=True)
    parser.add_argument("--starts", type=int, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    begun = time.time()
    job = {"status": "running", "pid": os.getpid(), "started_unix": begun,
           "mode": args.mode, "kernel": args.kernel}
    write_json(args.output/"job.json", job)
    try:
        config = json.loads((HERE/"config.json").read_text())
        if args.starts is not None:
            config["multistarts"] = args.starts
        write_json(args.output/"effective_config.json", config)
        baseline_path = args.baseline/(args.mode+"_fit.json")
        baseline = json.loads(baseline_path.read_text())
        producer = json.loads((args.baseline/"source_snapshot.json").read_text())
        records = [load(name) for name in config["records"]]
        if baseline["record_names"] != config["records"] or baseline["hold"] != config["primary_hold"]:
            raise AssertionError("Baseline membership or hold differs")
        for d in records:
            path = HERE.parents[2]/d.source
            if hashlib.sha256(path.read_bytes()).hexdigest() != producer["sha256"][d.source]:
                raise AssertionError("Baseline raw data differ from frozen inputs")
        write_json(args.output/"covariance_audit.json", audit())
        print("Starting alternating GLS", args.mode, args.kernel, flush=True)
        noise_start = [] if args.kernel == "iid" else [10.]+([.2] if args.kernel.endswith("_nugget") else [])
        alternating = alternating_fit(records, config, args.mode, config["primary_hold"], args.kernel, baseline, noise_start)
        write_json(args.output/"alternating.json", alternating)
        print("Alternating iterations", len(alternating["history"]), "stable", alternating["converged"], flush=True)
        def callback(row):
            write_json(args.output/"progress.json", {"latest_start": row, "elapsed_s": time.time()-begun})
            print("joint", row["start"], row.get("negative_log_likelihood", row.get("exception")), row["success"], flush=True)
        result = joint_fit(records, config, args.mode, config["primary_hold"], args.kernel,
                           baseline, callback, alternating)
        best = result["best"]
        predictions = fitted_records(records, result)
        raw = np.concatenate([d.temperature[1:]-p[1:] for d, p in zip(records, predictions)])
        nll, coefficients, sigma, W = projected_state(records, config, args.mode,
                                                    config["primary_hold"], args.kernel,
                                                    np.asarray(best["coordinates"]))
        checked_nll, checked_sigma, standardized = profiled_likelihood(raw, W)
        if abs(checked_nll-nll) > 1e-7 or abs(checked_sigma-sigma) > 1e-8:
            raise AssertionError("Saved trajectory likelihood differs from projected fit")
        metrics = []
        arrays = {}
        offset = 0
        for j, (d, pred) in enumerate(zip(records, predictions)):
            count = len(d.time)-1
            residual = raw[offset:offset+count]
            innovation = standardized[offset:offset+count]
            innovation_acf = acf(innovation)
            lags = np.arange(1, len(innovation_acf)+1)
            Q = float(count*(count+2)*np.sum(np.asarray(innovation_acf)**2/(count-lags)))
            command = d.input[1:]-d.input[1:].mean()
            correlation = None if np.linalg.norm(command) < 1e-12 else float(np.corrcoef(command, innovation)[0, 1])
            metrics.append({"name": d.name, "n": count, "RMSE_C": float(np.sqrt(np.mean(residual**2))),
                            "bias_C": float(residual.mean()), "raw_acf_1_to_30": acf(residual),
                            "innovation_acf_1_to_30": innovation_acf,
                            "innovation_mean": float(innovation.mean()),
                            "innovation_std": float(innovation.std(ddof=0)),
                            "innovation_skew": float(skew(innovation)),
                            "innovation_excess_kurtosis": float(kurtosis(innovation)),
                            "innovation_input_correlation": correlation,
                            "ljung_box_Q_30": Q, "ljung_box_reference_p_30": float(chi2.sf(Q, len(lags))),
                            "diagnostic_notice": "Reference chi-square p-values do not adjust for fitted parameters; calibrated adequacy requires refit simulation diagnostics"})
            arrays.update({"time_"+str(j): d.time, "observed_"+str(j): d.temperature,
                           "predicted_"+str(j): pred, "input_"+str(j): d.input,
                           "standardized_innovation_"+str(j): innovation})
            offset += count
        n = result["nobs"]
        nparams = 3+(1 if args.mode == "constant" else 2)*len(records)+len(best["noise_parameters"])+1
        result.update({"per_run": metrics, "raw_RMSE_C": float(np.sqrt(np.mean(raw**2))),
                       "logdet_unit_shape": W.logdet,
                       "nominal_parameter_count": nparams,
                       "AICc_nominal": float(2*nll+2*nparams+2*nparams*(nparams+1)/(n-nparams-1)),
                       "information_criterion_notice": "Nominal count includes profiled K/ambient/variance; active bounds and serial dependence complicate regular interpretations",
                       "alternating_minus_joint_nll": alternating["history"][-1]["negative_log_likelihood"]-nll,
                       "baseline_fit_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                       "elapsed_total_s": time.time()-begun})
        write_json(args.output/"fit.json", result)
        np.savez_compressed(args.output/"trajectories.npz", **arrays)
        job.update(status="complete", elapsed_s=time.time()-begun)
        write_json(args.output/"job.json", job)
    except Exception as exc:
        job.update(status="failed", elapsed_s=time.time()-begun, exception=repr(exc), traceback=traceback.format_exc())
        write_json(args.output/"job.json", job)
        raise


if __name__ == "__main__":
    main()
