"""Check every printed coverage count and generated prose scalar against data."""
import argparse
import json
from pathlib import Path
import re

from plot_coverage import MODELS, NOISES, load_delay_summary, sha
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=Path, default=HERE/"results/coverage")
    parser.add_argument("--summary", type=Path, default=HERE/"results/coverage_summary")
    parser.add_argument("--config", type=Path, default=HERE/"coverage_config.json")
    parser.add_argument("--figures", type=Path, default=HERE/"manuscript/figures")
    parser.add_argument("--tables", type=Path, default=HERE/"manuscript/tables")
    args = parser.parse_args()
    rows, inputs = load_delay_summary(args.summary, args.coverage, args.config)
    manifest_path = args.tables/"coverage_table_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit_path = args.figures/"coverage_numeric_audit.json"
    inputs[str(audit_path)] = sha(audit_path)
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed" or manifest["input_sha256"] != inputs:
        raise RuntimeError("Coverage table source/audit identity changed")
    if set(manifest["outputs"]) != {"coverage_rates.tex", "coverage_values.tex"}:
        raise RuntimeError("Coverage table producer output ownership changed")
    for name, digest in manifest["outputs"].items():
        if sha(args.tables/name) != digest:
            raise RuntimeError("Coverage table or scalar bytes changed")
    indexed = {(r["case"], r["true_noise"], r["method"]): r for r in rows}
    methods = ["iid_known_sigma", "iid_estimated_sigma", "hac60", "profile_estimated_sigma_iid"]
    if manifest["methods"] != methods:
        raise RuntimeError("Printed coverage method selection changed")
    lines = [line for line in (args.tables/"coverage_rates.tex").read_text().splitlines()
             if any(line.startswith(title+" & ") for _, title, _, _ in MODELS)]
    expected = []
    for case, title, _, _ in MODELS:
        for noise, _ in NOISES:
            cells = [title, {"iid": "IID", "ou10": "OU10", "ou60": "OU60"}[noise]]
            for method in methods:
                row = indexed[(case, noise, method)]
                cells.append(str(int(row["covered"]))+"/"+str(int(row["defined"])) if row["defined"] else "--")
            expected.append(cells)
    actual = [[cell.strip().removesuffix(r" \\").strip() for cell in line.split(" & ")] for line in lines]
    if actual != expected:
        raise RuntimeError("Missing, duplicated or incorrect printed coverage count")
    values = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(\d+)\}", (args.tables/"coverage_values.tex").read_text()))
    row = indexed[(MODELS[1][0], "ou60", "profile_known_sigma_iid")]
    expected_values = {"CoverageReplicates": str(int(row["requested"])),
                       "CoverageTotalDatasets": str(json.loads((args.summary/"audit.json").read_text())["requested_draws"]),
                       "CoverageTwoStageOUKnownGridChanges": str(int(row["quadratic_interpolation_coverage_class_changes"]))}
    if values != expected_values:
        raise RuntimeError("Generated coverage prose scalar changed")
    save_json(args.tables/"coverage_table_numeric_audit.json", {"status": "passed", "printed_count_cells": 48,
        "prose_scalars": 3, "table_manifest_sha256": sha(manifest_path), "outputs_sha256": manifest["outputs"],
        "source_figure_audit_sha256": sha(audit_path), "auditor_sha256": sha(HERE/"audit_coverage_table.py"),
        "scope": "Exact printed counts, method/model/noise row selection and three direct-field macros checked against the independently audited interval statistics. Layout review is separate."})
    print("48 coverage count cells and 3 prose scalars passed")


if __name__ == "__main__":
    main()
