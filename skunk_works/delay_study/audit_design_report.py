"""Check design display summaries, formatting and trajectory peak receipts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--summary", type=Path, default=here/"results/design_report")
    parser.add_argument("--figures", type=Path, default=here/"manuscript/figures")
    parser.add_argument("--tables", type=Path, default=here/"manuscript/tables")
    args = parser.parse_args()
    summary_path = args.summary/"summary.json"
    prepared = json.loads((args.summary/"manifest.json").read_text())
    display_path = args.figures/"design_report_manifest.json"
    display = json.loads(display_path.read_text())
    for manifest in (prepared, display):
        for name, digest in manifest["input_sha256"].items():
            if sha(Path(name)) != digest:
                raise RuntimeError("Display source changed")
    if prepared["summary_sha256"] != sha(summary_path):
        raise RuntimeError("Prepared summary changed")
    summary = json.loads(summary_path.read_text())
    count, maximum_relative = 0, 0.
    for phase, rows in summary["summaries"].items():
        for row in rows:
            reference = summary["references"][phase+"/"+row["case"]]
            record = json.loads(Path(reference["path"]).read_text())
            expected = record
            for part in reference["pointer"]:
                expected = expected[part]
            if expected != reference["value"]:
                raise RuntimeError("Saved record reference differs")
            if phase == "design_final":
                pairs = [(row["relative_utility"], -expected)]
            elif phase == "discrimination_designs":
                pairs = [(row["relative_log_distance_utility"], -expected),
                         (row["smallest_selected_pair_distance_squared"],
                          min(p["distance_squared"] for p in record["evaluations"]["selected_local_design"]["pairs"]))]
            else:
                pairs = [(row["closest_found_distance_squared_upper_bound"], expected),
                         (row["fixed_distance_squared"], record["refit"]["fixed_parameter_distance_squared"])]
            for saved, expected in pairs:
                error = abs(saved-expected)/max(1., abs(expected))
                maximum_relative = max(maximum_relative, error)
                if error > 1e-10:
                    raise RuntimeError("Summary scalar differs from audited record")
                count += 1
    table_count = 0
    for name, rows in display["table_cells"].items():
        path = args.tables/(name+".tex")
        if sha(path) != display["table_sha256"][path.name]:
            raise RuntimeError("Printed table changed")
        phase = "design_final" if name == "parameter_design" else "discrimination_designs"
        for row in rows:
            source = next(r for r in summary["summaries"][phase] if r["case"] == row["case"])
            tokens = [format(source[field], ".3f") for field in row["fields"]]
            if tokens != row["tokens"] or row["prefix"]+" & "+" & ".join(tokens)+r" \\" not in path.read_text():
                raise RuntimeError("Printed table cell differs from saved scalar")
            table_count += len(tokens)
    peaks = 0
    for trace in summary["traces"].values():
        t = np.array(trace["time_s"])
        lo, hi = np.array(trace["minimum_model_temperature_C"]), np.array(trace["maximum_model_temperature_C"])
        if (not np.all(np.diff(t) > 0) or len(t) != len(lo) or len(lo) != len(hi)
                or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi)):
            raise RuntimeError("Malformed deterministic temperature envelope")
        if abs(max(p["saved_maximum_C"] for p in trace["model_peaks"])-hi.max()) > 1e-7:
            raise RuntimeError("Envelope maximum differs from saved independent dense audit")
        for peak in trace["model_peaks"]:
            if abs(peak["maximum_C"]-peak["saved_maximum_C"]) > 1e-7:
                raise RuntimeError("Model trajectory peak differs from independent dense audit")
            peaks += 1
    receipt = {"status": "passed", "source_scalars_checked": count, "table_cells_checked": table_count,
               "model_peaks_checked": peaks, "maximum_summary_relative_difference": maximum_relative,
               "summary_sha256": sha(summary_path), "display_manifest_sha256": sha(display_path),
               "auditor_sha256": sha(Path(__file__)),
               "scope": "All displayed design/refit scalar sources and table formats, source hashes and 64 deterministic model-peak receipts. Temperature curves use the separately verified propagator; this check is not an independent all-time integrator, global optimizer, coverage or hardware certificate. Color/grayscale and rendered pages require visual review."}
    (args.figures/"design_report_numeric_audit.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
