"""Plot prepared design results and format saved audited summary cells."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tclab-matplotlib")
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figure_style as fs
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=HERE/"results/design_report")
    parser.add_argument("--output", type=Path, default=HERE/"manuscript/figures")
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    source = args.summary/"summary.json"
    manifest_path = args.summary/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "prepared" or sha(source) != manifest["summary_sha256"]:
        raise RuntimeError("Prepared summary identity changed")
    for name, digest in manifest["input_sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError("Prepared summary input changed")
    data = json.loads(source.read_text())
    outputs, cells = [], {}

    def save(fig, name, notice):
        fs.grayscale_preview(fig, str(args.output/name))
        fs.save_fig(fig, str(args.output/name), sources=[str(source), str(manifest_path)], notes=notice)
        outputs.append(name)

    fig, axes = fs.figure(fs.MULTI_PANEL, nrows=2, sharex=True)
    curves = [("constant", "Constant", 0, ":"), ("multisine", "Multisine", 1, "--"),
              ("mean_D", "Ensemble mean D", 5, "-"), ("min_D", "Ensemble min D", 3, "-.")]
    for key, label, color, style in curves:
        trace = data["traces"][key]
        for ax, field in zip(axes, ["command_percent", "maximum_model_temperature_C"]):
            ax.plot(trace["time_s"], trace[field], color=fs.color(color), linestyle=style,
                    linewidth=fs.DENSE_LINEWIDTH, label=label)
    axes[0].set(title="(a) Candidate commands through shutdown", ylabel="Command (%)", ylim=(-3, 103))
    axes[1].set(title="(b) Upper envelope of sixteen mean models", ylabel=r"Temperature ($^\circ$C)", xlabel="Time (s)")
    axes[1].axhline(85, color="0.5", linewidth=fs.REFERENCE_LINEWIDTH)
    for ax in axes:
        ax.axvline(900, color="0.65", linewidth=fs.REFERENCE_LINEWIDTH)
        ax.set_xlim(0, 1200)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, .88), h_pad=1.7)
    save(fig, "design_commands_temperature", "Two common candidates and two selected D designs without historical information. The lower panel is a pointwise maximum across sixteen assumed fitted thermal means; it is not a confidence band or the temperature trajectory of one identified physical system. The dense 0.25 s grid includes shutdown and the 300 s tail. No hardware or continuous-time guarantee.")

    rows = data["summaries"]["design_final"]
    fig, axes = fs.figure(fs.MULTI_PANEL, nrows=2, sharex=True)
    criteria = ["D", "E", "theta", "prediction"]
    for prior, ax in enumerate(axes):
        for aggregation, color, marker in [("mean", 5, "o"), ("min", 1, "s")]:
            ys = [next(row["relative_utility"] for row in rows
                       if row["case"] == "ensemble__"+aggregation+"__"+criterion+"__prior"+str(prior))
                  for criterion in criteria]
            ax.plot(range(4), ys, color=fs.color(color), marker=marker, markersize=5,
                    linewidth=fs.DENSE_LINEWIDTH, label=aggregation.capitalize())
        ax.set(title=["(a) Future data only", "(b) Including historical information"][prior], ylabel=r"$\Delta$ log utility")
        ax.axhline(0, color="0.65", linewidth=fs.REFERENCE_LINEWIDTH)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(["D", "E", "Delay", "Prediction"])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.015))
    fig.tight_layout(rect=(0, 0, 1, .93), h_pad=1.5)
    save(fig, "design_relative_utilities", "Directly audited selected optimization utilities relative to each scenario's reference. Future-only ensembles use multisine; historical-information ensembles use constant input. Mean/min aggregate log improvements over 48 fixed scenarios. Criteria have different scales and targets; comparing ordinate heights across criteria or reference changes is not an absolute information comparison.")

    refits = data["summaries"]["discrimination_refits"]
    fig, axes = fs.figure(fs.MULTI_PANEL, nrows=2, ncols=2, sharex=True, sharey=True)
    waveforms = ["constant", "multisine", "selected_mean", "selected_min"]
    for i, initialization in enumerate(["cold", "input_equilibrated"]):
        for order, label, color, marker in [(2, r"$n=2$", 1, "s"), (12, r"$n=12$", 3, "^"), (0, "Pure delay", 4, "D")]:
            selected = [next(row for row in refits if row["initialization"] == initialization
                             and row["noise_length_s"] is None and row["alternative_order"] == order
                             and row["waveform"] == waveform) for waveform in waveforms]
            for j, field in enumerate(["fixed_distance_squared", "closest_found_distance_squared_upper_bound"]):
                axes[i, j].plot(range(4), [row[field] for row in selected], color=fs.color(color),
                                marker=marker, markersize=4, linewidth=fs.DENSE_LINEWIDTH, label=label)
    for i, ax in enumerate(axes.flat):
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_title(["(a) Cold, fixed means", "(b) Cold, refitted alternative",
                      "(c) Equilibrated, fixed means", "(d) Equilibrated, refitted alternative"][i])
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Const.", "Multi.", "Mean", "Min"], rotation=20)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Separation $D^2$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.015))
    fig.tight_layout(rect=(0, 0, 1, .93), h_pad=2., w_pad=1.)
    save(fig, "discrimination_refits", "First declared working covariance (IID, 0.25 C); n1 fixed generating mean, n2/n12/pure alternatives. Left columns hold the original alternative parameters; right columns use the smallest converged 24-start distance. Selected mean/min waveforms come from fixed-mean discrimination designs, not parameter D designs. Symmetric-log axes are linear below 1e-8 and retain exact zeros. These distances are not real-data testing power; refitted minima are upper bounds on the global closest alternative.")

    def table(name, headers, selected_rows, caption, label):
        text = ["% Generated by plot_design_results.py from separately audited phase summaries.",
                r"\begin{table}[tbp]", r"\centering\small", r"\begin{tabular}{llrr}", r"\toprule",
                " & ".join(headers)+r" \\", r"\midrule"]
        cells[name] = []
        for title, category, row, fields in selected_rows:
            tokens = [format(row[field], ".3f") for field in fields]
            text.append(title+" & "+category+" & "+" & ".join(tokens)+r" \\")
            cells[name].append({"prefix": title+" & "+category, "case": row["case"], "fields": fields, "tokens": tokens})
        text += [r"\bottomrule", r"\end{tabular}", r"\caption{"+caption+"}", r"\label{"+label+"}", r"\end{table}", ""]
        (args.tables/(name+".tex")).write_text("\n".join(text))
    selected_rows = []
    for row in rows:
        if row["case"].startswith("ensemble"):
            title = row["aggregation"]+", "+("history" if row["include_historical_information"] else "future")
            criterion = {"theta": "Delay", "prediction": "Prediction"}.get(row["criterion"], row["criterion"])
            selected_rows.append((title, criterion, row, ["relative_utility", "dense_maximum_C"]))
    table("parameter_design", ["Ensemble, information", "Criterion", r"$\Delta\log$ utility", r"Max. (${}^\circ$C)"], selected_rows,
          r"Selected local ensemble designs over 48 model/startup/covariance scenarios. Utility is the mean or minimum scenario-relative log improvement. Future-only ensembles use multisine as reference; ensembles with historical information use constant input, whose total information is nonsingular after adding calibration information. Criterion scales differ. Temperature is the largest independently checked value across sixteen fitted mean models through the shutdown tail. These local results do not establish global optimality, probabilistic model robustness or hardware limits.", "tab:parameter-design")
    selected_rows = []
    for row in data["summaries"]["discrimination_designs"]:
        noise = "IID" if row["noise_length_s"] is None else "OU"+str(int(row["noise_length_s"]))
        startup = "Cold" if row["initialization"] == "cold" else "Equilibrated"
        selected_rows.append((startup+", "+noise, row["aggregation"], row,
                              ["relative_log_distance_utility", "smallest_selected_pair_distance_squared"]))
    table("discrimination_design", ["Startup, covariance", "Aggregate", r"$\Delta\log D^2$", r"Smallest $D^2$"], selected_rows,
          r"Fixed-mean discrimination designs compare six pairs among the four representative formulations. The objective aggregates pairwise log improvements relative to multisine, whereas the last column is the smallest raw selected pair distance; these are different comparisons. Covariances are common known working laws with marginal standard deviation $0.25\,{}^\circ\mathrm C$. Alternative-parameter refitting is evaluated separately in Fig.~\ref{fig:discrimination-refits}.", "tab:discrimination-design")
    receipt = {"input_sha256": {str(source): sha(source), str(manifest_path): sha(manifest_path)},
               "figures": outputs, "table_sha256": {name+".tex": sha(args.tables/(name+".tex")) for name in cells},
               "table_cells": cells, "producer_sha256": sha(Path(__file__)),
               "style_sha256": sha(HERE/"figure_style.py"), "status": "generated_awaiting_scalar_and_visual_review"}
    snapshot = HERE.parents[1]/"snapshot.json"
    if snapshot.exists():
        receipt["source_commit"] = json.loads(snapshot.read_text())["commit"]
        (args.output/"design_report_source_snapshot.json").write_text(snapshot.read_text())
    save_json(args.output/"design_report_manifest.json", receipt)


if __name__ == "__main__":
    main()
