"""Prespecified delay coverage/width views; refuse unfinished Monte Carlo runs.

Read saved summaries only. Conditional Wilson intervals, undefined outcomes,
requested denominators and disconnected-set widths retain separate meanings.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tclab-matplotlib")

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import figure_style as fs
from run_fits import save_json

HERE = Path(__file__).resolve().parent
METHODS = ["iid_known_sigma", "iid_estimated_sigma", "hac10", "hac30", "hac60",
           "profile_known_sigma_iid", "profile_estimated_sigma_iid"]
METHOD_LABELS = ["IID\nknown", "IID\nfitted", "HAC\n10", "HAC\n30", "HAC\n60",
                 "Profile\nknown", "Profile\nfitted"]
MODELS = [("joint__chain_n1_io_cold_linear_exact", r"$n=1$", 5, "o"),
          ("joint__chain_n2_io_cold_linear_exact", r"$n=2$", 1, "s"),
          ("joint__chain_n12_io_cold_linear_exact", r"$n=12$", 3, "^"),
          ("joint__pure_delay_n0_io_cold_linear_exact", "Pure delay", 4, "D")]
NOISES = [("iid", "IID truth"), ("ou10", "OU truth, 10 s"), ("ou60", "OU truth, 60 s")]
FIELDS = ["case", "true_noise", "method", "requested", "defined", "covered", "undefined_or_failed",
          "coverage_when_defined", "wilson_when_defined_lower", "wilson_when_defined_upper",
          "binomial_mcse_when_defined", "unconditional_coverage_lower", "unconditional_coverage_upper",
          "completed_draws", "failed_draws", "missing_draws", "undefined_on_completed_draws",
          "mean_width", "median_width", "width_q025", "width_q975", "mean_envelope_width",
          "mean_component_count", "open_exploration_sets", "quadratic_interpolation_defined",
          "quadratic_interpolation_coverage_class_changes"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_delay_summary(summary, coverage, config_path):
    """Source integrity gate runs before any outcome values are plotted."""
    audit_path, table_path = summary/"audit.json", summary/"coverage_summary.csv"
    manifest_path = coverage/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    if (manifest["status"] not in {"complete", "complete_with_failures"} or manifest["filter"]
            or manifest["identity"]["replicate_override"] is not None
            or len(manifest["completed"])+len(manifest["failed"]) != manifest["expected_chunks"]
            or manifest["expected_draws"] != 3600 or config["replicates"] != 300
            or config["cases"] != [m[0] for m in MODELS] or set(config["noise_scenarios"]) != {n[0] for n in NOISES}):
        raise RuntimeError("Full terminal prespecified coverage run required before plotting")
    audit = json.loads(audit_path.read_text())
    if (audit["status"] != "passed" or audit["requested_draws"] != 3600 or audit["summary_rows"] != 384
            or audit["input_sha256"].get(str(manifest_path)) != sha(manifest_path)
            or manifest["identity"].get("config_sha256") != sha(config_path)
            or sha(config_path) not in audit["input_sha256"].values()
            or audit.get("summary_sha256") != sha(table_path)):
        raise RuntimeError("Coverage summary does not match its completed arithmetic audit")
    for path, digest in audit["input_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError("Coverage source changed after the summary audit: "+path)
    frame = pd.read_csv(table_path)
    if len(frame) != 384 or frame.duplicated(["case", "true_noise", "coordinates", "parameter", "method"]).any():
        raise RuntimeError("Coverage summary lost or duplicated rows")
    selected = frame[(frame.coordinates == "io") & (frame.parameter == "theta")].copy()
    expected = {(m[0], n[0], method) for m in MODELS for n in NOISES for method in METHODS}
    if set(zip(selected.case, selected.true_noise, selected.method)) != expected or len(selected) != 84:
        raise RuntimeError("Delay coverage summary lacks the complete 84-row selection")
    inputs = {str(path): sha(path) for path in [audit_path, table_path, manifest_path, config_path]}
    rows = []
    for noise, _ in NOISES:
        for case, _, _, _ in MODELS:
            for method in METHODS:
                row = selected[(selected.case == case) & (selected.true_noise == noise) & (selected.method == method)].iloc[0]
                if row.requested != 300 or not 0 <= row.covered <= row.defined <= 300:
                    raise RuntimeError("Invalid coverage denominator or counts")
                values = {name: None if pd.isna(row[name]) else row[name] for name in FIELDS}
                rows.append({name: value.item() if isinstance(value, np.generic) else value for name, value in values.items()})
    return rows, inputs


def figures(rows, output, inputs):
    indexed = {(r["case"], r["true_noise"], r["method"]): r for r in rows}
    maximum_width = max((r["median_width"] for r in rows if r["median_width"] is not None), default=0.)
    outputs = []
    for kind in ["coverage", "width"]:
        figure, axes = fs.figure(fs.MULTI_PANEL, nrows=3, sharex=True, sharey=True)
        for j, (noise, title) in enumerate(NOISES):
            ax = axes[j]
            for k, (case, model, color, marker) in enumerate(MODELS):
                x = np.arange(7)+(k-1.5)*.13
                for i, method in enumerate(METHODS):
                    row = indexed[(case, noise, method)]
                    value = row["coverage_when_defined" if kind == "coverage" else "median_width"]
                    if value is None:
                        ax.annotate("NA", (x[i], .015), xycoords=("data", "axes fraction"),
                                    rotation=90, ha="center", va="bottom", color=fs.color(color))
                        continue
                    hollow = row["defined"] < row["requested"]
                    error = None
                    if kind == "coverage":
                        error = np.array([[value-row["wilson_when_defined_lower"]],
                                          [row["wilson_when_defined_upper"]-value]])
                    ax.errorbar(x[i], value, yerr=error, color=fs.color(color), marker=marker,
                                markersize=4, markerfacecolor="white" if hollow else fs.color(color),
                                linestyle="none", elinewidth=fs.DENSE_LINEWIDTH, capsize=2,
                                label=model if i == 0 else None)
            ax.set_title("("+chr(97+j)+") "+title)
            ax.set_ylabel("Coverage" if kind == "coverage" else "Median width\n(s)")
            ax.set_xlim(-.6, 6.6)
            if kind == "coverage":
                ax.axhline(.95, color="0.6", linestyle="--", linewidth=fs.REFERENCE_LINEWIDTH)
                ax.set_ylim(-.01, 1.03)
                ax.set_yticks([0., .5, 1.])
            else:
                ax.set_ylim(0., maximum_width*1.08 if maximum_width > 0 else 1.)
        axes[-1].set_xticks(range(7))
        axes[-1].set_xticklabels(METHOD_LABELS)
        handles = [Line2D([], [], color=fs.color(color), marker=marker, linestyle="none", label=model)
                   for _, model, color, marker in MODELS]
        figure.legend(handles=handles, loc="upper center", ncol=4,
                      bbox_to_anchor=(.5, 1.015), columnspacing=1.)
        figure.tight_layout(rect=(0., 0., 1., .94), h_pad=.8)
        name = "coverage_delay_"+kind
        notes = ("All four generating points and three true noise laws; joint cold records, sigma=0.25 C. "
                 "IID known/fitted variance, Bartlett HAC bandwidths 10/30/60 samples, and IID profiles known/fitted variance. "
                 "Coverage is conditional on a defined interval; bars are pointwise 95% Wilson Monte Carlo intervals. "
                 "Dashed line is nominal 0.95. Hollow markers flag fewer than 300 defined intervals; NA is undefined, never zero coverage. "
                 "Widths are medians among defined intervals; disconnected profile widths sum component lengths, not their envelope. "
                 "Exact counts, unconditional failure bounds, exploration flags and interpolation sensitivity remain in coverage_delay_data.json/CSV. "
                 "No simultaneous, universal or real-model-discrepancy coverage claim.")
        fs.grayscale_preview(figure, str(output/name))
        fs.save_fig(figure, str(output/name), sources=list(inputs), notes=notes)
        outputs.append(name)
        plt.close(figure)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=Path, default=HERE/"results/coverage")
    parser.add_argument("--summary", type=Path, default=HERE/"results/coverage_summary")
    parser.add_argument("--config", type=Path, default=HERE/"coverage_config.json")
    parser.add_argument("--output", type=Path, default=HERE/"manuscript/figures")
    args = parser.parse_args()
    rows, inputs = load_delay_summary(args.summary, args.coverage, args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    data_path = args.output/"coverage_delay_data.json"
    csv_path = args.output/"coverage_delay_data.csv"
    save_json(data_path, {"rows": rows})
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.15g")
    outputs = figures(rows, args.output, inputs)
    result = {"status": "generated_awaiting_independent_numeric_and_visual_audit", "input_sha256": inputs,
              "producer_sha256": {n: sha(HERE/n) for n in ["plot_coverage.py", "figure_style.py"]},
              "plot_data_sha256": sha(data_path), "csv_sha256": sha(csv_path), "figures": outputs,
              "selection": "84 IO delay rows, four generating models x three true covariances x seven interval methods"}
    snapshot = HERE.parents[1]/"snapshot.json"
    if snapshot.exists():
        result["source_commit"] = json.loads(snapshot.read_text())["commit"]
        (args.output/"coverage_source_snapshot.json").write_text(snapshot.read_text())
    save_json(args.output/"coverage_figure_manifest.json", result)


if __name__ == "__main__":
    main()
