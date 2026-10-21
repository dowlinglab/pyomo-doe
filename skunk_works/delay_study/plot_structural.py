"""Figures from saved equivalence, nonregular-limit and boundary-function data."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tclab-matplotlib")

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_style as fs
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--output", type=Path, default=HERE/"manuscript/figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources, data, names = {}, {}, []

    def source(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def save(figure, name, notes):
        fs.grayscale_preview(figure, str(args.output/name))
        fs.save_fig(figure, str(args.output/name), sources=list(sources), notes=notes)
        names.append(name)
        plt.close(figure)

    curves = pd.read_csv(source(args.results/"structural/equivalence_curves.csv"))
    joint = curves[curves.calibration == "joint"]
    fit_path = source(args.results/"fits/joint__original_n0_physics_cold_linear_exact.json")
    fit = json.loads(fit_path.read_text())["fit"]["parameters"]
    r0 = fit["CpS"]/fit["Ub"]
    figure, axes = fs.figure(fs.MULTI_PANEL, nrows=2, ncols=2, sharex=True)
    for ax, parameter, title, unit in zip(axes.flat[:3], ["CpH", "CpS", "Ub"],
                                           [r"(a) Heater capacity $C_{p,H}$", r"(b) Sensor capacity $C_{p,S}$", r"(c) Coupling $U_b$"],
                                           [r"J K$^{-1}$", r"J K$^{-1}$", r"W K$^{-1}$"]):
        ax.plot(joint.r_s, joint[parameter], color=fs.color(5), linewidth=fs.DENSE_LINEWIDTH)
        ax.plot(r0, fit[parameter], "o", color=fs.color(1), markersize=6)
        ax.set(title=title, ylabel=unit)
        ax.locator_params(axis="y", nbins=4)
    axes[1, 1].plot(joint.r_s, joint.maximum_prediction_difference_C*1e12, color=fs.color(0),
                    linewidth=fs.DENSE_LINEWIDTH)
    axes[1, 1].set(title="(d) Forward numerical difference", ylabel=r"Maximum difference ($10^{-12}$ C)", ylim=(0., 2.1))
    for ax in axes[-1]:
        ax.set_xlabel(r"Family coordinate $r=C_{p,S}/U_b$ (s)")
    for ax in axes.flat:
        ax.set_xlim(10., 155.)
        ax.set_xticks([25, 75, 125])
    figure.tight_layout(h_pad=1.6, w_pad=1.6)
    data["physical_family"] = {column: joint[column].tolist() for column in joint.columns if column != "calibration"}
    data["physical_family"]["selected_fit_parameters"] = fit
    save(figure, "physical_family", "Joint zero-state original transfer function, 403 positive equivalent members. Orange dots mark the arbitrary member selected by the original optimizer, not uniquely estimated physical values. Maximum temperature differences use both calibration records and expose floating-point agreement, not measurement precision. Numerical fitting boxes restrict the positive family further; plotted full-family endpoints are approached but not attained.")

    summary = pd.read_csv(source(args.results/"family_uncertainty/summary.csv"))
    summary = summary[summary.case == "joint__chain_n1_io_cold_linear_exact"]
    methods = ["parametric", "block1", "block10", "block30", "block60"]
    labels = ["IID simulation", "Residual b=1", "Residual b=10", "Residual b=30", "Residual b=60"]
    functions = ["CpH_inf_J_K", "CpH_sup_J_K", "CpS_sup_J_K", "Ub_sup_W_K"]
    titles = [r"(a) $\inf C_{p,H}$", r"(b) $\sup C_{p,H}$", r"(c) $\sup C_{p,S}$", r"(d) $\sup U_b$"]
    figure, axes = fs.figure(fs.MULTI_PANEL, nrows=2, ncols=2, sharey=True)
    boundary = {}
    for ax, name, title in zip(axes.flat, functions, titles):
        values = []
        for y, method in enumerate(methods):
            row = summary[(summary.function == name) & (summary.method == method)].iloc[0]
            lower, center, upper = 100*(np.array([row.lower, row['median'], row.upper])/row.point-1)
            ax.errorbar(center, y, xerr=[[center-lower], [upper-center]], fmt="o", color=fs.color(5),
                        linewidth=fs.DENSE_LINEWIDTH, markersize=4, capsize=3)
            values.append({"method": method, "point": float(row.point), "lower": float(row.lower),
                           "median": float(row['median']), "upper": float(row.upper)})
        ax.axvline(0., color="0.65", linestyle="--", linewidth=fs.REFERENCE_LINEWIDTH)
        ax.set(title=title, xlabel="Change from fitted boundary (%)", ylim=(4.6, -.6))
        ax.set_yticks(range(5))
        ax.set_yticklabels(labels)
        ax.locator_params(axis="x", nbins=4)
        boundary[name] = values
    figure.tight_layout(h_pad=1.8, w_pad=1.5)
    data["family_boundary_intervals"] = boundary
    save(figure, "family_boundary_intervals", "Joint cold n1, 400 saved draws per resampling method. Bars are conditional 95 percent percentiles of identifiable positive-family boundary functions, expressed as percentage changes from each fitted boundary. They are not marginal intervals for an arbitrary original physical parameter, and do not include uncertainty in heater calibration or model discrepancy. Strict-positive family boundaries precede rectangular fit boxes.")

    poles = pd.read_csv(source(args.results/"structural/repeated_poles.csv"))
    onset_path = source(args.results/"onset_limit/onset.json")
    onset = json.loads(onset_path.read_text())
    figure, axes = fs.figure(fs.TWO_PANEL, ncols=2)
    pole_data = {}
    for i, order in enumerate([1, 2, 12]):
        rows = poles[(poles.order == order) & (poles.delta_s > 0)].sort_values("delta_s")
        axes[0].loglog(rows.delta_s, rows.squared_prediction_distance_C2, color=fs.color([5, 1, 3][i]),
                       label=r"$n="+str(order)+"$", linewidth=fs.DENSE_LINEWIDTH, linestyle=["-", "--", "-."][i])
        pole_data[str(order)] = {name: rows[name].tolist() for name in ["delta_s", "squared_prediction_distance_C2"]}
    reference_rows = poles[(poles.order == 2) & (poles.delta_s > 0)].sort_values("delta_s")
    delta = np.array([.001, 3.])
    coefficient = reference_rows.iloc[0].squared_prediction_distance_C2/reference_rows.iloc[0].delta_s**4
    axes[0].loglog(delta, coefficient*delta**4, color="0.55", linestyle=":", linewidth=fs.REFERENCE_LINEWIDTH, label=r"$\propto\delta^4$")
    axes[0].set(title="(a) Repeated pole", xlabel=r"Fixed-mean perturbation $\delta$ (s)", ylabel=r"Squared mean distance (C$^2$)")
    rows = onset["rows"]
    order = np.array([row["order"] for row in rows])
    rise = np.array([row["rise_at_mean_delay_C"] for row in rows])
    axes[1].loglog(order, rise, "o-", color=fs.color(5), markersize=4, linewidth=fs.DENSE_LINEWIDTH, label="Exact onset rise")
    axes[1].loglog(order, onset["CLT_coefficient_C"]/np.sqrt(order), color="0.5", linestyle="--", linewidth=fs.REFERENCE_LINEWIDTH,
                   label=r"$A/\sqrt{n}$")
    axes[1].set(title="(b) Delay-chain onset", xlabel="Delay stages n", ylabel=r"Rise at $t=\theta$ (C)")
    for ax in axes:
        ax.legend(loc="lower left", bbox_to_anchor=(0., 1.12), ncol=2, columnspacing=.6, handlelength=1.4)
    figure.tight_layout(rect=(0., 0., 1., .79), w_pad=1.5)
    data["nonregular_limits"] = {"repeated_poles": pole_data, "quartic_reference_coefficient": float(coefficient),
                                  "onset_rows": rows, "CLT_coefficient_C": onset["CLT_coefficient_C"]}
    save(figure, "nonregular_limits", "Left: noiseless squared prediction distance along tau=10+delta and theta=10n-delta for the recorded sine command, K=.78, cold states. The dotted quartic guide is anchored at the smallest positive n2 perturbation. Right: exact cold constant-step rise at t=theta for K=.78,tau=150s,theta=18s,u=50 percent; pure delay is still at ambient at this instant. A=Ku theta/(tau sqrt(2pi)). These separate deterministic calculations are not fitted order-selection results or confidence intervals.")

    path = args.output/"structural_plot_data.json"
    save_json(path, data)
    manifest = {"producer_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ("plot_structural.py", "figure_style.py")},
                "input_sha256": sources, "plot_data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "figures": names,
                "status": "generated_awaiting_independent_numeric_and_visual_audit"}
    snapshot = HERE.parents[1]/"snapshot.json"
    if snapshot.exists():
        manifest["source_commit"] = json.loads(snapshot.read_text())["commit"]
        (args.output/"structural_source_snapshot.json").write_text(snapshot.read_text())
    save_json(args.output/"structural_figure_manifest.json", manifest)


if __name__ == "__main__":
    main()
