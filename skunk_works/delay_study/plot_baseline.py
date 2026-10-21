"""Plot completed baseline files only; never fit or simulate a model.

The style module is adapted from the user's claude-for-researchers resource.
The only added standard size supports a two-column, three-row comparison.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tclab-matplotlib")

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_style as fs

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--output", type=Path, default=HERE/"manuscript/figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [args.results/"fit_summary.csv", args.results/"manifest.json"]
    source = pd.read_csv(sources[0])
    summary = source[(source.coordinates == "io") & source.family.isin(["chain", "pure_delay"])]
    orders = [1, 2, 3, 4, 6, 8, 12, 0]
    plot_data = {}

    # Question: how do order and initialization change calibration error?
    figure, axes = fs.figure(fs.TALL, nrows=3, sharex=True, sharey=True)
    upper = np.ceil(10*summary.rmse_C.max()*1.12)/10
    for ax, calibration, letter in zip(axes, ["sine", "step", "joint"], ["a", "b", "c"]):
        ax.set_title("("+letter+") "+calibration.capitalize(), loc="left")
        rows = summary[summary.calibration == calibration]
        for initialization, label, marker, style, color in [
            ("cold", "Cold", "o", "-", fs.color(5)),
            ("input_equilibrated", "Equilibrated", "s", "--", fs.color(1)),
        ]:
            selected = rows[rows.initialization == initialization].set_index("order").loc[orders]
            values = selected.rmse_C.to_numpy()
            plot_data["order_rmse__"+calibration+"__"+initialization] = values.tolist()
            ax.plot(np.arange(len(orders)), values, color=color, marker=marker, linestyle=style, label=label)
        ax.set_ylabel("RMSE (°C)")
        ax.set_ylim(0., upper)
        ax.set_yticks(np.arange(0., upper+.01, .4))
    axes[-1].set_xticks(np.arange(len(orders)))
    axes[-1].set_xticklabels([str(n) if n else r"$\infty$" for n in orders])
    axes[-1].set_xlabel(r"Delay stages $n$")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.02), columnspacing=.8, handlelength=1.4)
    figure.tight_layout(rect=(0., 0., 1., .95))
    fs.grayscale_preview(figure, str(args.output/"baseline_order"))
    fs.save_fig(figure, str(args.output/"baseline_order"), sources=[str(p) for p in sources],
                notes="Saved IO-coordinate fits. Infinity is the separately fitted pure-delay limit. Lines connect categorical orders as guides; no error bars are implied.")

    # Question: which time regions account for differences among mean models?
    selected_models = [("chain", 1, r"$n=1$", fs.color(5), "-"),
                       ("chain", 2, r"$n=2$", fs.color(1), "--"),
                       ("chain", 12, r"$n=12$", fs.color(3), "-."),
                       ("pure_delay", 0, "Pure delay", fs.color(4), ":")]
    figure, axes = fs.figure(fs.MULTI_PANEL, nrows=3, ncols=2, sharex="col", sharey="row",
                             gridspec_kw={"height_ratios": [1, 2, 1.4]})
    legend_handles = []
    for family, order, label, color, style in selected_models:
        path = args.results/"predictions"/(f"joint__{family}_n{order}_io_cold_linear_exact.csv.gz")
        sources.append(path)
        frame = pd.read_csv(path)
        for column, experiment in enumerate(["sine", "step"]):
            data = frame[(frame.role == "calibration") & (frame.experiment == experiment)]
            key = family+str(order)+"__"+experiment
            plot_data[key] = {name: data[name].tolist() for name in ("time_s", "input_percent", "measured_C", "predicted_C")}
            if family == "chain" and order == 1:
                axes[0, column].plot(data.time_s, data.input_percent, color=fs.color(0))
                observed, = axes[1, column].plot(data.time_s, data.measured_C, ".", color="0.5", markersize=fs.DENSE_MARKERSIZE, label="Observed", zorder=3)
            line, = axes[1, column].plot(data.time_s, data.predicted_C, color=color, linestyle=style, label=label, zorder=2, linewidth=fs.DENSE_LINEWIDTH)
            axes[2, column].plot(data.time_s, data.predicted_C-data.measured_C, color=color, linestyle=style, linewidth=fs.DENSE_LINEWIDTH)
            if column == 0:
                legend_handles.append(line)
    for column, name in enumerate(["Sine", "Step"]):
        axes[0, column].set_title(name)
        axes[2, column].axhline(0., color="0.7", linewidth=fs.REFERENCE_LINEWIDTH, zorder=0)
        axes[2, column].set_xlabel("Time (s)")
        axes[2, column].set_xticks([0, 300, 600, 900])
        experiment = ["sine", "step"][column]
        axes[2, column].set_xlim(0., plot_data["chain1__"+experiment]["time_s"][-1])
    axes[0, 0].set_ylabel(r"$u$ (%)")
    axes[1, 0].set_ylabel(r"$T$ (°C)")
    axes[2, 0].set_ylabel("Residual (°C)")
    figure.legend([observed]+legend_handles, ["Observed"]+[x[2] for x in selected_models],
                  loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.02), columnspacing=1.)
    figure.tight_layout(rect=(0., 0., 1., .88), h_pad=2.)
    fs.grayscale_preview(figure, str(args.output/"baseline_trajectories"))
    fs.save_fig(figure, str(args.output/"baseline_trajectories"), sources=[str(p) for p in sources],
                notes="Jointly fitted cold-start models; every recorded calibration sample is shown. Residual sign is prediction minus observation. These are mean predictions, without confidence or prediction bands.")

    data_path = args.output/"baseline_plot_data.json"
    data_path.write_text(json.dumps(plot_data, indent=2)+"\n")
    manifest = {"producer": "plot_baseline.py", "source_sha256": {p.name: sha(p) for p in [HERE/"plot_baseline.py", HERE/"figure_style.py"]},
                "inputs": {str(p): sha(p) for p in sources}, "plot_data_sha256": sha(data_path),
                "style_origin": "claude-for-researchers/resources/scripts/figure_style/figure_style.py; added MULTI_PANEL=(8,6.4)",
                "figures": ["baseline_order", "baseline_trajectories"],
                "status": "generated_awaiting_visual_review",
                "notice": "Formatting saved numerical outputs only. PDF timestamps may change on regeneration. Grayscale previews support visual checking, not automatic accessibility certification."}
    snapshot = HERE.parents[1]/"snapshot.json"
    if snapshot.exists():
        receipt = json.loads(snapshot.read_text())
        manifest["source_commit"] = receipt["commit"]
        (args.output/"baseline_source_snapshot.json").write_text(snapshot.read_text())
        manifest["source_snapshot_sha256"] = sha(args.output/"baseline_source_snapshot.json")
        manifest["source_snapshot_notice"] = "The phase-specific baseline_source_snapshot.json records this plot generation. A generic source_snapshot.json in the directory is the first launch, which can predate restyling."
    else:
        manifest["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    (args.output/"baseline_figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
