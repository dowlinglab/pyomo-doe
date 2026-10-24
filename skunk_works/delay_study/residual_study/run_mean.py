"""Source-bound audit and pooled OLS baseline for the residual teaching study."""
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
import pandas as pd

from audit_mean import audit
from data import load
from mean import fit_mean, fitted_records


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=None)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = json.loads((HERE/"config.json").read_text())
    if args.starts is not None:
        config["multistarts"] = args.starts
    write_json(output/"effective_config.json", config)
    write_json(output/"job.json", {"status": "running", "pid": os.getpid(), "started_unix": started})
    try:
        checks = audit()
        write_json(output/"forward_audit.json", checks)
        records = [load(name) for name in config["records"]]
        data_rows = []
        signatures = set()
        for d in records:
            path = HERE.parents[2]/d.source
            frame = pd.read_csv(path)
            signature = hashlib.sha256(frame.to_numpy(dtype="<f8").tobytes()).hexdigest()
            if signature in signatures:
                raise AssertionError("Duplicate primary recording")
            signatures.add(signature)
            data_rows.append({"name": d.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "numeric_sha256": signature, "n": len(d.time),
                              "first_T1_C": d.temperature[0],
                              "duration_s": d.time[-1]-d.time[0],
                              "dt_min_s": np.diff(d.time).min(), "dt_max_s": np.diff(d.time).max(),
                              "Q2_nonzero": bool(np.any(frame.Q2.to_numpy() != 0))})
        write_json(output/"data_audit.json", {"unique_ENV2_records": len(signatures), "records": data_rows,
                   "hold_evidence": "Historical acquisition b114b13f: lab.Q1(control_signal(t)) immediately precedes p.update(t); supports left hold, not a per-file timing log"})
        for mode in ("constant", "linear"):
            print("Starting pooled "+mode+" ambient OLS", flush=True)
            def progress(row):
                write_json(output/"progress.json", {"mode": mode, "latest_start": row,
                                                     "elapsed_s": time.time()-started})
                print(mode, row["start"], row.get("SSE", row.get("exception")), flush=True)
            result = fit_mean(records, config, mode, config["primary_hold"], progress)
            preds = fitted_records(records, result)
            metrics = []
            arrays = {}
            for j, (d, pred) in enumerate(zip(records, preds)):
                residual = d.temperature[1:]-pred[1:]
                centered = residual-residual.mean()
                acf1 = float(centered[1:]@centered[:-1]/(centered@centered))
                metrics.append({"name": d.name, "RMSE_C": float(np.sqrt(np.mean(residual**2))),
                                "bias_C": float(residual.mean()), "raw_residual_acf1": acf1})
                arrays.update({"time_"+str(j): d.time, "observed_"+str(j): d.temperature,
                               "predicted_"+str(j): pred, "input_"+str(j): d.input})
            result["per_run"] = metrics
            write_json(output/(mode+"_fit.json"), result)
            np.savez_compressed(output/(mode+"_trajectories.npz"), **arrays)
        write_json(output/"job.json", {"status": "complete", "pid": os.getpid(),
                   "started_unix": started, "elapsed_s": time.time()-started})
    except Exception as exc:
        write_json(output/"job.json", {"status": "failed", "pid": os.getpid(),
                   "started_unix": started, "elapsed_s": time.time()-started,
                   "exception": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
