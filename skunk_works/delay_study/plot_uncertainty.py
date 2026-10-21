"""Publication figures from saved likelihood, resampling and sensitivity files."""
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
from scipy.stats import norm

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

    def read(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    def fit(order, calibration="joint", initialization="cold", family=None):
        family = family or ("chain" if order else "pure_delay")
        coordinates = "physics" if family == "original" else "io"
        path = args.results/"fits"/(f"{calibration}__{family}_n{order}_{coordinates}_{initialization}_linear_exact.json")
        return read(path), path

    def save(figure, name, notes):
        fs.grayscale_preview(figure, str(args.output/name))
        fs.save_fig(figure, str(args.output/name), sources=list(sources), notes=notes)
        names.append(name)
        plt.close(figure)

    # Show each pole-label mode on its own declared x window. A full union is
    # retained in the source; the table does not collapse it to an envelope.
    index = read(args.results/"profile_index.json")["profiles"]
    profile_name = "joint__chain_n1_io_cold_linear_exact__theta.json"
    profile_path = args.results/index[profile_name]["path"]
    profile = read(profile_path)["profile"]
    x = np.array([p["parameter_value"] for p in profile["points"]])
    known = np.array([p["lr_known_sigma_iid"] for p in profile["points"]])
    unknown = np.array([p["lr_unknown_sigma_iid"] for p in profile["points"]])
    components = profile["intervals_unknown_sigma_iid"]
    if len(components) != 2:
        raise RuntimeError("Expected both one-stage pole modes")
    windows = []
    for component in components:
        width = component["upper"]-component["lower"]
        windows.append([component["lower"]-width, component["upper"]+width])
    figure, axes = fs.figure(fs.TWO_PANEL, ncols=2, sharey=True)
    for ax, window, title in zip(axes, windows, ["(a) Short-delay mode", "(b) Exchanged-pole mode"]):
        ax.plot(x, np.maximum(0., known), color=fs.color(5), label="Known variance", linewidth=fs.DENSE_LINEWIDTH)
        ax.plot(x, np.maximum(0., unknown), color=fs.color(1), linestyle="--", label="Profiled variance", linewidth=fs.DENSE_LINEWIDTH)
        ax.axhline(profile["chi2_95_df1"], color="0.5", linestyle=":", linewidth=fs.REFERENCE_LINEWIDTH)
        ax.set(xlim=window, ylim=(-.15, 10), xlabel=r"Delay $\theta$ (s)", title=title)
        ax.locator_params(axis="x", nbins=4)
    axes[0].set_ylabel("Likelihood-ratio statistic")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.04))
    figure.tight_layout(rect=(0., 0., 1., .86))
    data["profile_modes"] = {"source": str(profile_path), "theta_s": x.tolist(), "known_lr_raw": known.tolist(),
                              "unknown_lr_raw": unknown.tolist(), "windows_s": windows, "threshold": profile["chi2_95_df1"],
                              "plot_notice": "Negative numerical LR roundoff is clamped to zero for display only; raw values and both disjoint windows are retained."}
    save(figure, "profile_modes", "Joint cold n1 theta profile, two explicitly separate x windows. Curves join saved nuisance-optimized grid points. Known sigma is 0.25 C; the other curve profiles a common IID variance. Dotted line is nominal chi-square 95 percent df1, not verified real-data coverage.")

    bootstrap_path = args.results/"bootstrap/summary.csv"
    sources[str(bootstrap_path)] = hashlib.sha256(bootstrap_path.read_bytes()).hexdigest()
    bootstrap = pd.read_csv(bootstrap_path)
    figure, axes = fs.figure(fs.MULTI_PANEL, nrows=2, ncols=2, sharex=True, sharey=True)
    interval_data = {}
    labels = ["IID simulation", "Residual b=1", "Residual b=10", "Residual b=30", "Residual b=60", "IID profile", "OU curvature"]
    for ax, order in zip(axes.flat, [1, 2, 12, 0]):
        record, fit_path = fit(order)
        key = fit_path.stem
        rows = []
        for method in ("parametric", "block1", "block10", "block30", "block60"):
            selected = bootstrap[(bootstrap.case == key) & (bootstrap.method == method) &
                                 (bootstrap.coordinates == "io") & (bootstrap.parameter == "theta")]
            if len(selected) != 1:
                raise RuntimeError("Missing bootstrap interval")
            row = selected.iloc[0]
            rows.append({"method": method, "lower": float(row.lower), "center": float(row['median']), "upper": float(row.upper)})
        path = args.results/index[key+"__theta.json"]["path"]
        likelihood = read(path)["profile"]
        point = record["fit"]["parameters"]["theta"]
        component = next(c for c in likelihood["intervals_unknown_sigma_iid"] if c["lower"] <= point <= c["upper"])
        rows.append({"method": "profile_unknown_iid_canonical_component", "lower": component["lower"], "center": point, "upper": component["upper"]})
        ou_path = args.results/"ou"/fit_path.name
        ou = read(ou_path)
        point_ou = ou["fit"]["parameters"]["theta"]
        covariance = ou["uncertainty"]["observed_cov_log_mean_and_length"]
        width_log = norm.ppf(.975)*np.sqrt(covariance[2][2])
        rows.append({"method": "ou_observed_log_normal_curvature", "lower": float(point_ou*np.exp(-width_log)),
                     "center": point_ou, "upper": float(point_ou*np.exp(width_log))})
        for y, row in enumerate(rows):
            ax.errorbar(row["center"], y, xerr=[[row["center"]-row["lower"]], [row["upper"]-row["center"]]],
                        fmt="o", color=fs.color(y), markersize=4, capsize=3, linewidth=fs.DENSE_LINEWIDTH)
        ax.axvline(point, color="0.7", linewidth=fs.REFERENCE_LINEWIDTH, linestyle="--", zorder=0)
        ax.set_title((r"$n="+str(order)+"$" if order else "Pure delay")+(" (canonical)" if order == 1 else ""), loc="left")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlim(12.7, 18.2)
        ax.set_ylim(6.6, -.6)
        ax.set_xticks([13, 15, 17])
        interval_data[key] = rows
    for ax in axes[-1]:
        ax.set_xlabel(r"Delay $\theta$ (s)")
    figure.tight_layout(h_pad=1.8, w_pad=2.)
    data["delay_intervals"] = interval_data
    save(figure, "delay_intervals", "Joint cold fits. Error bars are conditional intervals under different assumptions, not one common uncertainty estimate. Bootstrap uses 400 draws per method. The n1 profile component and bootstrap use the short-delay canonical mode; profile_modes shows both labels. OU endpoints exponentiate the local normal log-theta curvature, and remain challenged by correlated innovations. Gray lines are baseline LS theta values.")

    series = []
    for order in [1, 2, 12, 0]:
        record, path = fit(order)
        series.append((r"$n="+str(order)+"$" if order else "Pure delay", record, path))
    original, path = fit(0, family="original")
    series.append(("Original (physical)", original, path))
    step, path = fit(2, calibration="step", initialization="input_equilibrated")
    series.append(("Step, n=2, equilibrated", step, path))
    figure, ax = fs.figure(fs.WIDE)
    spectra = []
    for i, (name, record, path) in enumerate(series):
        u = record["uncertainty"]
        values = np.array(u["eigenvalues_log_unit_variance"])[::-1]
        relative = values/values[0]
        rank = u["rank_relative_1e-8"]
        ax.semilogy(np.arange(1, rank+1), relative[:rank], marker=["o", "s", "^", "D", "x", "+"][i],
                    linestyle=["-", "--", "-.", ":", "--", ":"][i], color=fs.color(i),
                    label=name+f" ({rank}/{len(values)})", linewidth=fs.DENSE_LINEWIDTH, markersize=5)
        spectra.append({"label": name, "fit_path": str(path), "relative_eigenvalues_descending": relative.tolist(),
                        "rank": rank, "parameter_dimension": len(values)})
    ax.set(xlabel="Positive eigenvalue, largest first", ylabel=r"$\lambda_j/\lambda_{\max}$", xlim=(.85, 3.15), ylim=(.001, 1.5))
    ax.set_xticks([1, 2, 3])
    figure.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, 1.04), ncol=2,
                  columnspacing=.7, handlelength=1.6)
    figure.tight_layout(rect=(0., 0., 1., .72))
    data["information_spectra"] = spectra
    save(figure, "information_spectra", "Only positive information eigenvalues above the relative J rank cutoff are plotted, normalized within each case. Legend parentheses give identifiable local rank / native parameter dimension; null eigenvalues are omitted rather than assigned a plotting floor. Reduced models use log IO coordinates; the original uses four physical logs. This permits rank inspection, not an invariant comparison of information magnitude across metrics.")

    figure, axes = fs.figure(fs.TWO_PANEL, ncols=2, sharey=False)
    vectors_data = {"io": [], "physics": [], "orders": [1, 2, 12, 0]}
    M = np.array([[-1., 0., 0.], [-1., 1., 0.], [0., 0., 1.]])
    for order in [1, 2, 12, 0]:
        record, _ = fit(order)
        io = np.array(record["uncertainty"]["eigenvectors_log_columns"])[:, 0]
        F = np.array(record["uncertainty"]["fim_log_unit_variance"])
        _, physics = np.linalg.eigh(M.T@F@M)
        for key, vector in (("io", io), ("physics", physics[:, 0])):
            vector *= np.sign(vector[np.argmax(abs(vector))])
            vectors_data[key].append(vector.tolist())
    for ax, key, title, labels_y in zip(axes, ["io", "physics"], ["(a) IO log coordinates", "(b) Physical log coordinates"],
                                        [[r"$\log K$", r"$\log\tau$", r"$\log\theta$"],
                                         [r"$\log U_a$", r"$\log C_p$", r"$\log\theta$"]]):
        array = np.array(vectors_data[key]).T
        shown = ax.imshow(array, vmin=-1, vmax=1, cmap=fs.DIVERGING, aspect="auto",
                          interpolation="nearest", resample=False)
        for (i, j), value in np.ndenumerate(array):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > .6 else "black")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(labels_y)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["1", "2", "12", r"$\infty$"])
        ax.set_xlabel("Delay stages")
        ax.set_title(title)
    figure.tight_layout(rect=(0., 0., .9, 1.))
    color_axis = figure.add_axes([.92, .2, .02, .6])
    figure.colorbar(shown, cax=color_axis, ticks=[-1., 0., 1.])
    data["weak_directions"] = vectors_data
    save(figure, "weak_directions", "Unit weakest information eigenvectors for joint cold fits. The largest absolute component has positive sign by convention. Physical directions are re-diagonalized at the same model point after the exact log-coordinate transformation, not interpreted as extra observations. Entries refer to the chosen local pole mode and metric.")

    path = args.output/"uncertainty_plot_data.json"
    save_json(path, data)
    manifest = {"producer_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ("plot_uncertainty.py", "figure_style.py")},
                "input_sha256": sources, "plot_data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "figures": names,
                "status": "generated_awaiting_independent_numeric_and_visual_audit"}
    snapshot = HERE.parents[1]/"snapshot.json"
    if snapshot.exists():
        manifest["source_commit"] = json.loads(snapshot.read_text())["commit"]
        (args.output/"uncertainty_source_snapshot.json").write_text(snapshot.read_text())
    save_json(args.output/"uncertainty_figure_manifest.json", manifest)


if __name__ == "__main__":
    main()
