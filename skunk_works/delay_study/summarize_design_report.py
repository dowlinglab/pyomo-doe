"""Prepare audited design scalars and deterministic temperature envelopes.

No input optimization or parameter refitting. Temperature envelopes range over
the sixteen declared fitted mean models, without a probability interpretation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_design import dense_command
from data import Experiment
from inference import ModelSpec, predict
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--output", type=Path, default=HERE/"results/design_report")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs, summaries, references, cache = {}, {}, {}, {}

    def read(path):
        path = path.resolve()
        if path not in cache:
            inputs[str(path)] = sha(path)
            cache[path] = json.loads(path.read_text())
        return cache[path]

    for phase, count in [("design_final", 112), ("discrimination_designs", 12), ("discrimination_refits", 72)]:
        base = args.results/phase
        audit, manifest = read(base/"audit.json"), read(base/"manifest.json")
        if (audit["status"] != "passed" or audit.get("cases", audit.get("designs")) != count
                or manifest["status"] != "complete" or manifest["failed"]
                or len(manifest["completed"]) != count):
            raise RuntimeError("Full audited phase required: "+phase)
        for name, digest in audit["input_sha256"].items():
            if sha(Path(name)) != digest:
                raise RuntimeError("Audited phase input changed: "+name)
        path = base/"audited_summary.csv"
        if sha(path) != audit["summary_sha256"]:
            raise RuntimeError("Audited summary changed")
        inputs[str(path.resolve())] = sha(path)
        summaries[phase] = pd.read_csv(path).replace({np.nan: None}).to_dict("records")

    config = read(HERE/"design_config.json")
    traces, maximum_peak_difference = {}, 0.
    choices = [("constant", "ensemble__mean__D__prior0", "constant"),
               ("multisine", "ensemble__mean__D__prior0", "multisine"),
               ("mean_D", "ensemble__mean__D__prior0", "selected_local_design"),
               ("min_D", "ensemble__min__D__prior0", "selected_local_design")]
    for label, case, waveform in choices:
        path = args.results/"design_final"/(case+".json")
        record = read(path)
        evaluation = record["evaluations"][waveform]
        amplitudes, knots = np.asarray(evaluation["amplitudes_percent"]), np.asarray(record["knots_s"])
        time, command = dense_command(amplitudes, knots, config)
        temperatures, peaks = [], []
        for filename, digest in record["identity"]["fit_sha256"].items():
            fit_path = args.results/"fits"/filename
            if sha(fit_path) != digest:
                raise RuntimeError("Fitted mean changed")
            fitted = read(fit_path)
            spec = ModelSpec(**fitted["fit"]["spec"])
            parameters = np.array([fitted["fit"]["parameters"][key] for key in spec.names])
            data = Experiment("design_report", time, np.full(len(time), config["ambient_C"]), command, config["ambient_C"])
            temperature = predict(data, parameters, spec)
            name = ("pure" if spec.family == "pure_delay" else "n"+str(spec.order))+"__"+spec.initialization+"__iid"
            saved = next(row for row in evaluation["dense_audit"]["scenarios"] if row["scenario"] == name)
            peak = float(temperature.max())
            maximum_peak_difference = max(maximum_peak_difference, abs(peak-saved["maximum_C"]))
            peaks.append({"scenario": name, "maximum_C": peak, "saved_maximum_C": saved["maximum_C"]})
            temperatures.append(temperature)
        traces[label] = {"case": case, "waveform": waveform, "time_s": time.tolist(),
                         "command_percent": command.tolist(),
                         "minimum_model_temperature_C": np.min(temperatures, axis=0).tolist(),
                         "maximum_model_temperature_C": np.max(temperatures, axis=0).tolist(),
                         "model_peaks": peaks}
    if maximum_peak_difference > 1e-7:
        raise RuntimeError("Report trajectories disagree with independent dense audit")
    # Expose simple saved JSON fields for the formatting-only table stage.
    for phase, rows in summaries.items():
        for row in rows:
            path = args.results/phase/(row["case"]+".json")
            record = read(path)
            if phase in ("design_final", "discrimination_designs"):
                pointer = ["optimization", "best", "objective"]
            else:
                pointer = ["refit", "best", "distance_squared"]
            value = record
            for part in pointer:
                value = value[part]
            references[phase+"/"+row["case"]] = {"path": str(path.resolve()), "pointer": pointer, "value": value}
    result = {"summaries": summaries, "traces": traces, "references": references,
              "maximum_peak_difference_C": maximum_peak_difference,
              "notice": "Fixed fitted-mean scenarios, declared working covariances and local multistart solutions. No global optimum, model probability, continuous-time constraint or hardware certificate. Closest-found refit distances are upper bounds on the global minimum; fixed-mean classification errors are not refitted-model power."}
    destination = args.output/"summary.json"
    save_json(destination, result)
    manifest = {"status": "prepared", "input_sha256": inputs, "summary_sha256": sha(destination),
                "producer_sha256": sha(Path(__file__)), "config_sha256": sha(HERE/"design_config.json"),
                "maximum_peak_difference_C": maximum_peak_difference,
                "sources_sha256": {name: sha(HERE/name) for name in ["models.py", "inference.py", "audit_design.py"]}}
    snapshot = HERE.parents[1]/"snapshot.json"
    if snapshot.exists():
        manifest["source_commit"] = json.loads(snapshot.read_text())["commit"]
    save_json(args.output/"manifest.json", manifest)
    print(json.dumps({"status": "prepared", "phases": list(summaries), "waveforms": len(traces),
                      "maximum_peak_difference_C": maximum_peak_difference}))


if __name__ == "__main__":
    main()
