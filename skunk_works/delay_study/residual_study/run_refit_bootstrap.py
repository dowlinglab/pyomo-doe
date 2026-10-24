"""Checkpointed, source-bound diagnostic/parameter bootstrap with full refits."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

import refit_bootstrap
from audit_covariance import audit as audit_covariance
from run_mean import write_json


def summarize(observed, draws):
    good = [d for d in draws if d["status"] == "complete"]
    result = {"requested_or_completed_draws": len(draws), "usable": len(good),
              "failed": sum(d["status"] == "failed" for d in draws),
              "requires_revisit": sum(d["status"] == "requires_revisit" for d in draws),
              "notice": "Plug-in refit diagnostic calibration or measurement-rounding sensitivity, conditional on fitted structure/startup. Plus-one tail fractions have finite simulation resolution; not exact hypothesis tests or coverage guarantees."}
    if not good:
        return result
    names = ["global_max_abs_acf", "global_max_abs_input_correlation", "global_max_Q30"]
    comparisons = {}
    for name in names:
        if observed[name] is None:
            continue
        sample = np.array([d["statistics"][name] for d in good])
        comparisons[name] = {"observed": observed[name], "simulation_quantiles_50_95_99": np.quantile(sample, [.5, .95, .99]).tolist(),
                             "tail_fraction_plus_one": float((1+np.sum(sample >= observed[name]))/(len(sample)+1))}
    result["global_comparisons"] = comparisons
    per_run = []
    for j, row in enumerate(observed["per_run"]):
        stats = {}
        for name in ("max_abs_acf_1_to_30", "max_abs_input_correlation", "ljung_box_Q_30"):
            if row[name] is None:
                continue
            sample = np.array([d["statistics"]["per_run"][j][name] for d in good])
            stats[name] = {"observed": row[name], "simulation_95": float(np.quantile(sample, .95)),
                           "tail_fraction_plus_one": float((1+np.sum(sample >= row[name]))/(len(sample)+1))}
        per_run.append({"record": row["record"], "comparisons": stats})
    result["per_run"] = per_run
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--measurement", choices=["gaussian", "rounded", "innovation_blocks"], default="gaussian")
    parser.add_argument("--block-length", type=int, default=60)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.draws <= 0 or args.starts < 2:
        raise ValueError("Invalid bootstrap resource/count settings")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/"whitening_audit.json", audit_covariance())
    begun = time.time()
    seed_base = 23261024
    refit_bootstrap.initialize(args.reference, args.starts, args.measurement, seed_base, args.block_length)
    S = refit_bootstrap.STATE
    identity = {"reference_fit_sha256": hashlib.sha256((args.reference/"fit.json").read_bytes()).hexdigest(),
                "reference_config_sha256": hashlib.sha256((args.reference/"effective_config.json").read_bytes()).hexdigest(),
                "measurement": args.measurement, "draws": args.draws, "starts_per_refit": args.starts,
                "block_length": args.block_length,
                "seed_base": seed_base,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")},
                "source_parent_data_sha256": hashlib.sha256((HERE.parent/"data.py").read_bytes()).hexdigest()}
    checkpoint = {"identity": identity, "draws": [], "status": "running", "pid": os.getpid(), "started_unix": begun}
    if args.resume_from:
        old = json.loads(args.resume_from.read_text())
        if old["identity"] != identity:
            raise ValueError("Resume identity differs; do not mix bootstrap sources/settings")
        checkpoint["draws"] = old["draws"]
    observed = refit_bootstrap.statistics(S["records"], S["reference"])
    write_json(args.output/"observed.json", observed)
    write_json(args.output/"generation.json", {"measurement": args.measurement, "grid": S["grid"],
               "block_length_samples": args.block_length if args.measurement == "innovation_blocks" else None,
               "block_notice": "Noncircular moving blocks of centered per-run standardized innovations, inverse-whitened through the fitted temporal covariance. Retains residual dependence as a sensitivity; does not correct deterministic input-related bias, justify stationarity or guarantee coverage.",
               "conditioning": "First recorded temperature fixed; initial lag/prehistory as reference; residual processes reset per run",
               "rounded_notice": "When rounded, replace Gaussian measurement noise by grid rounding; retain fitted slow-process variance. Refit the same Gaussian working likelihood to both mechanisms."})
    write_json(args.output/"checkpoint.json", checkpoint)
    done = {d["index"] for d in checkpoint["draws"]}
    if len(done) != len(checkpoint["draws"]) or any(i < 0 or i >= args.draws for i in done):
        raise ValueError("Invalid/duplicate draw identities")
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=refit_bootstrap.initialize,
                                 initargs=(str(args.reference), args.starts, args.measurement, seed_base, args.block_length)) as pool:
            futures = {pool.submit(refit_bootstrap.execute, i): i for i in range(args.draws) if i not in done}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"index": index, "status": "failed", "exception": repr(exc), "traceback": traceback.format_exc()}
                checkpoint["draws"].append(row)
                checkpoint["draws"].sort(key=lambda r: r["index"])
                checkpoint["elapsed_s"] = time.time()-begun
                write_json(args.output/"checkpoint.json", checkpoint)
                if len(checkpoint["draws"]) % 10 == 0 or len(checkpoint["draws"]) == args.draws:
                    print(args.measurement, len(checkpoint["draws"]), "/", args.draws, row["status"], flush=True)
        checkpoint.update(status="complete", elapsed_s=time.time()-begun)
        write_json(args.output/"checkpoint.json", checkpoint)
        write_json(args.output/"summary.json", summarize(observed, checkpoint["draws"]))
    except Exception:
        checkpoint.update(status="failed", elapsed_s=time.time()-begun, traceback=traceback.format_exc())
        write_json(args.output/"checkpoint.json", checkpoint)
        raise


if __name__ == "__main__":
    main()
