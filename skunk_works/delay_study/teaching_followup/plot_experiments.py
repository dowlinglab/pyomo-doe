"""Plot saved Pyomo.DoE inputs and exact nominal temperature responses."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data import Experiment
from inference import ModelSpec, predict
import figure_style as fs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = HERE / "figures"
    destination.mkdir(exist_ok=True)
    fs.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=fs.MULTI_PANEL, sharex=True, sharey="row")
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.84, wspace=0.18, hspace=0.18)
    times = np.arange(0, 900.01, 0.25)
    sources, summaries, commands = {}, {}, {}
    handles = []
    for col, order in enumerate([1, 2]):
        for noise, color, style, marker, label in [
            ("iid", fs.color(5), "-", "o", "Independent errors"),
            ("ou60", fs.color(1), (0, (5, 3)), "s", "OU correlation, 60 s"),
        ]:
            name = f"n{order}_{noise}"
            path = HERE / "results" / (name + ".json")
            sources[str(path.relative_to(HERE))] = sha(path)
            result = json.loads(path.read_text())
            knots = np.asarray(result["command_knots_s"])
            control = np.asarray(result["selected"]["controls"])
            commands[name] = control.tolist()
            data = Experiment(name, times, np.full(len(times), 22.2),
                              np.interp(times, knots, control), 22.2)
            temperature = predict(data, np.asarray(result["nominal_K_tau_theta"]), ModelSpec(order=order))
            peak_index = int(np.argmax(temperature))
            assert abs(temperature[peak_index] - result["selected_dense_peak_C"]) < 1e-8
            summaries[name] = {"peak_temperature_C": float(temperature[peak_index]),
                               "peak_time_min": float(times[peak_index] / 60),
                               "control_knots_percent": control.tolist()}
            line, = axes[0, col].plot(knots / 60, control, color=color, linestyle=style,
                                     marker=marker, markersize=5, markerfacecolor="white", label=label)
            axes[1, col].plot(times / 60, temperature, color=color, linestyle=style)
            if col == 0:
                handles.append(line)
        baseline = Experiment("constant", times, np.full(len(times), 22.2), np.full(len(times), 50), 22.2)
        base_temperature = predict(baseline, np.asarray(result["nominal_K_tau_theta"]), ModelSpec(order=order))
        line = axes[0, col].axhline(50, color="0.5", linestyle=":", linewidth=1.6, label="Constant 50%")
        axes[1, col].plot(times / 60, base_temperature, color="0.5", linestyle=":", linewidth=1.6)
        if col == 0:
            handles.append(line)
        axes[0, col].set_title(f"$n={order}$", fontsize=16)
        axes[0, col].set_ylim(-3, 105)
        axes[0, col].set_yticks([0, 50, 100])
        axes[1, col].set_ylim(20, 90)
        axes[1, col].set_yticks([20, 40, 60, 80])
        axes[1, col].axhline(85, color="#a13c3c", linestyle=":", linewidth=1.2)
        axes[1, col].text(14.7, 86, "85°C limit", ha="right", va="bottom", fontsize=10, color="#a13c3c")
        axes[1, col].set_xlabel("Time (min)")
        for ax in axes[:, col]:
            ax.set_xlim(0, 15)
            ax.set_xticks([0, 5, 10, 15])
            ax.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0, 0].set_ylabel("Command (%)")
    axes[1, 0].set_ylabel("Temperature (°C)")
    fig.suptitle("Optimized TC-Lab experiments", y=0.99, fontsize=18)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.945),
               ncol=3, frameon=False, fontsize=11, handlelength=2.5, columnspacing=1.2)
    fig.text(0.14, 0.035, "Linear ramps between command knots; cold startup; nominal model predictions.", fontsize=10)
    for extension in ["png", "svg"]:
        fig.savefig(destination / ("optimized_experiments." + extension), dpi=300)
    plt.close(fig)
    with (destination / "command_knots.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_min", *commands])
        for i, t in enumerate(knots):
            writer.writerow([t / 60, *[commands[k][i] for k in commands]])
    receipt = {"status": "saved_source_and_peak_checks_passed_visual_review_separate",
               "source_sha256": sources, "script_sha256": sha(Path(__file__)),
               "simulator_sha256": {name: sha(HERE.parent / name) for name in ["models.py", "inference.py", "data.py", "figure_style.py"]},
               "summaries": summaries,
               "n2_max_command_difference_between_covariances_percent": float(np.max(np.abs(np.array(commands["n2_iid"]) - commands["n2_ou60"]))),
               "artifacts_sha256": {p.name: sha(p) for p in sorted(destination.iterdir()) if p.suffix in {".png", ".svg", ".csv"}},
               "scope": "Saved local D-optimal teaching designs; exact nominal trajectories on 0.25-s grid. Curves show separate working-covariance designs, not uncertainty bands. No optimization rerun, global optimum or hardware safety claim."}
    (destination / "plot_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["summaries"], indent=2))


if __name__ == "__main__":
    main()
