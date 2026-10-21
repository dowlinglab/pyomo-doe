"""Fresh multistart reproduction for three representative calibrated families.

This is a handoff check, not a rerun of the complete matrix or a global-minimum
certificate. Original physical coordinates are compared by prediction, not by
which equivalent family member an optimizer happens to return.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from data import load
from inference import ModelSpec, fit, predict
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((HERE/"config.json").read_text())
    rows = []
    names = ["joint__chain_n2_io_cold_linear_exact", "joint__pure_delay_n0_io_cold_linear_exact",
             "joint__original_n0_physics_cold_linear_exact"]
    for name in names:
        source = args.results/"fits"/(name+".json")
        record = json.loads(source.read_text())
        spec = ModelSpec(**record["fit"]["spec"])
        datasets = [load(n) for n in record["datasets"]]
        result = fit(datasets, spec, n_starts=config["n_starts"], seed=config["seed"], max_nfev=config["max_nfev"])
        expected = np.array([record["fit"]["parameters"][n] for n in spec.names])
        observed = np.array([result["parameters"][n] for n in spec.names])
        maximum = max(float(np.max(abs(predict(d, expected, spec)-predict(d, observed, spec)))) for d in datasets)
        delta = abs(record["fit"]["sse"]-result["sse"])
        if delta > 1e-6 or maximum > 1e-5:
            raise RuntimeError("Fresh fit does not reproduce saved mean/objective: "+name)
        save_json(args.output/(name+".json"), {"fit": result, "reference_fit_sha256": sha(source),
                                               "absolute_sse_difference_C2": delta,
                                               "maximum_prediction_difference_C": maximum})
        rows.append({"case": name, "reference_fit_sha256": sha(source), "starts": len(result["starts"]),
                     "successful_starts": sum(s["success"] for s in result["starts"]),
                     "absolute_sse_difference_C2": delta, "maximum_prediction_difference_C": maximum})
        print(name+" reproduced", flush=True)
    save_json(args.output/"audit.json", {"status": "passed", "cases": rows,
                                         "producer_sha256": sha(Path(__file__)),
                                         "scope": "Fresh 24-start fits for three representative joint cold families; means/objectives, not unique physical coordinates. No complete-matrix rerun or global-minimum certificate."})


if __name__ == "__main__":
    main()
