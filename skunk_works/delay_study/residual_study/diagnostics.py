"""Generate residual/PACF/input diagnostics and plots from saved covariance fits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_toeplitz

from audit_saved_mean import acf
from run_mean import write_json


def pacf(values, nlags=30):
    correlations = np.r_[1., acf(values, nlags)]
    result = []
    for lag in range(1, len(correlations)):
        coefficients = solve_toeplitz(correlations[:lag], correlations[1:lag+1])
        result.append(float(coefficients[-1]))
    return result


def input_correlation(innovation, command, maxlag=30):
    result = []
    for lag in range(-maxlag, maxlag+1):
        if lag > 0:
            a, b = innovation[lag:], command[:-lag]
        elif lag < 0:
            a, b = innovation[:lag], command[-lag:]
        else:
            a, b = innovation, command
        correlation = None if np.std(b) < 1e-12 else float(np.corrcoef(a, b)[0, 1])
        result.append({"lag_samples": lag, "correlation": correlation})
    return result


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "legend.fontsize": 7, "savefig.dpi": 180})
    bindings = {}
    full = []
    fits = []
    for path in sorted(args.input.glob("*/fit.json")):
        fit = json.loads(path.read_text())
        bindings[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        predictions = path.parent/"trajectories.npz"
        bindings[str(predictions)] = hashlib.sha256(predictions.read_bytes()).hexdigest()
        arrays = np.load(predictions)
        runs = []
        for j, name in enumerate(fit["record_names"]):
            innovation = arrays["standardized_innovation_"+str(j)]
            residual = (arrays["observed_"+str(j)]-arrays["predicted_"+str(j)])[1:]
            observed = arrays["observed_"+str(j)]
            n = len(innovation)
            windows = np.array_split(innovation, 6)
            levels = np.unique(observed)
            runs.append({"name": name, "raw_residual_pacf_1_to_30": pacf(residual),
                         "innovation_pacf_1_to_30": pacf(innovation),
                         "innovation_input_crosscorrelation": input_correlation(innovation, arrays["input_"+str(j)][1:]),
                         "innovation_block_means": [float(w.mean()) for w in windows],
                         "innovation_block_stds": [float(w.std()) for w in windows],
                         "first_150_innovation_mean": float(innovation[:150].mean()),
                         "last_150_innovation_mean": float(innovation[-150:].mean()),
                         "unique_T1_levels": len(levels),
                         "rounded_positive_level_spacing_C": np.unique(np.round(np.diff(levels), 2)).tolist(),
                         "median_dt_s": float(np.median(np.diff(arrays["time_"+str(j)]))),
                         "lag_notice": "Sample lags approximately seconds here; covariance uses exact irregular times. Positive input lag correlates innovation(t) with earlier command(t-lag)."})
        full.append({"cell": path.parent.name, "runs": runs})
        fits.append((fit, arrays, path.parent.name))
    if len(fits) != 8:
        raise AssertionError("Diagnostics require all eight saved cells")
    # Descriptive comparison of each raw/whitened ACF, preserving all runs.
    fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
    for ax, (fit, arrays, cell) in zip(axes.flat, fits):
        for j, row in enumerate(fit["per_run"]):
            ax.plot(np.arange(1, 31), row["innovation_acf_1_to_30"], linewidth=.6, alpha=.65)
        ax.axhline(0, color="black", linewidth=.5)
        ax.set_title(cell.replace("_", " "))
        ax.set_xlim(1, 30)
        ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel("Sample lag (about s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Innovation ACF")
    fig.tight_layout()
    fig.savefig(args.output/"kernel_innovation_comparison.png")
    plt.close(fig)
    # Inspect the lowest-AICc working description within each ambient family;
    # this plotting choice does not freeze scientific model structure.
    for mode in ("constant", "linear"):
        fit, arrays, cell = min((item for item in fits if item[0]["mode"] == mode),
                                key=lambda item: item[0]["AICc_nominal"])
        for protocol, label in [("sin_50_50", "30-second sine"), ("sin_2_50", "2-minute sine"),
                                ("sin_5_50", "5-minute sine"), ("step_50", "Step")]:
            indices = [j for j, name in enumerate(fit["record_names"]) if protocol+"_run" in name]
            fig, axes = plt.subplots(3, 2, figsize=(5, 5.5), gridspec_kw={"width_ratios": [1.5, 1.]})
            for row, j in enumerate(indices):
                time = arrays["time_"+str(j)][1:]
                residual = (arrays["observed_"+str(j)]-arrays["predicted_"+str(j)])[1:]
                axes[row, 0].plot(time, residual, color="black", linewidth=.5)
                axes[row, 0].axhline(0, color="gray", linewidth=.5)
                axes[row, 0].set_ylabel("Run "+str(row+1)+" residual, C")
                axes[row, 1].plot(np.arange(1, 31), fit["per_run"][j]["innovation_acf_1_to_30"], color="#165c85", linewidth=1)
                axes[row, 1].axhline(0, color="gray", linewidth=.5)
                axes[row, 1].set_ylabel("Innovation ACF")
                axes[row, 1].set_ylim(-.6, .6)
            axes[-1, 0].set_xlabel("Time, s")
            axes[-1, 1].set_xlabel("Sample lag (about s)")
            fig.suptitle(label+"; "+cell.replace("_", " "))
            fig.tight_layout(rect=(0, 0, 1, .96))
            fig.savefig(args.output/(mode+"_"+protocol+".png"))
            plt.close(fig)
    write_json(args.output/"diagnostics.json", {"inputs": bindings, "cells": full,
               "interpretation": "Descriptive diagnostics; no simulated adequacy calibration or structure selection has been completed"})
    write_json(args.output/"figure_receipt.json", {"inputs": bindings,
               "figures": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.png")},
               "visual_review": "pending"})


if __name__ == "__main__":
    main()
