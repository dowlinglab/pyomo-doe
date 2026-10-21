"""Check structural plotted arrays against formulas and independent saved draws."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_fits import save_json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--figures", type=Path, default=HERE/"manuscript/figures")
    args = parser.parse_args()
    manifest = json.loads((args.figures/"structural_figure_manifest.json").read_text())
    path = args.figures/"structural_plot_data.json"
    if sha(path) != manifest["plot_data_sha256"]:
        raise RuntimeError("Structural plot data changed")
    for name, expected in manifest["input_sha256"].items():
        if sha(Path(name)) != expected:
            raise RuntimeError("Structural figure input changed: "+name)
    data = json.loads(path.read_text())
    total, maximum = 0, 0.

    def compare(actual, expected, tolerance=1e-9):
        nonlocal total, maximum
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            raise RuntimeError("Structural figure data shape differs")
        error = float(np.max(abs(actual-expected)))
        if error > tolerance:
            raise RuntimeError("Structural figure numeric difference: "+str(error))
        total += actual.size
        maximum = max(maximum, error)

    family = data["physical_family"]
    p = json.loads((args.results/"fits/joint__original_n0_physics_cold_linear_exact.json").read_text())["fit"]["parameters"]
    Ua, Ub, H, S = [p[key] for key in ("Ua", "Ub", "CpH", "CpS")]
    a, b = H/Ua+S/Ub+S/Ua, H*S/(Ua*Ub)
    r = np.asarray(family["r_s"])
    capacity = Ua*(a-r-b/r)
    compare(family["Ua"], np.full(len(r), Ua))
    compare(family["CpH"], Ua*b/r)
    compare(family["CpS"], capacity)
    compare(family["Ub"], capacity/r)
    curves = pd.read_csv(args.results/"structural/equivalence_curves.csv")
    joint = curves[curves.calibration == "joint"]
    compare(family["maximum_prediction_difference_C"], joint.maximum_prediction_difference_C)
    compare(r, joint.r_s)
    if family["within_original_fit_box"] != joint.within_original_fit_box.tolist() or family["selected_fit_parameters"] != p:
        raise RuntimeError("Physical family marker/box membership changed")
    functions = {"CpH_inf_J_K": lambda K, t, d: .032/K*np.minimum(t, d),
                 "CpH_sup_J_K": lambda K, t, d: .032/K*np.maximum(t, d),
                 "CpS_sup_J_K": lambda K, t, d: .032/K*(t+d-2*np.sqrt(t*d)),
                 "Ub_sup_W_K": lambda K, t, d: .032/K*((t+d)**2/(4*t*d)-1)}
    fit = json.loads((args.results/"fits/joint__chain_n1_io_cold_linear_exact.json").read_text())["fit"]["parameters"]
    for name, rows in data["family_boundary_intervals"].items():
        function = functions[name]
        for row in rows:
            case = args.results/"bootstrap"/("joint__chain_n1_io_cold_linear_exact__"+row["method"]+".npz")
            draws = np.load(case)["parameters"]
            values = function(*draws.T)
            compare([row["lower"], row["median"], row["upper"]], np.quantile(values, [.025, .5, .975]))
            compare(row["point"], function(*[fit[key] for key in ("K", "tau", "theta")]))
    poles = pd.read_csv(args.results/"structural/repeated_poles.csv")
    for order, values in data["nonregular_limits"]["repeated_poles"].items():
        selected = poles[(poles.order == int(order)) & (poles.delta_s > 0)].sort_values("delta_s")
        for key, value in values.items():
            compare(value, selected[key])
    onset = json.loads((args.results/"onset_limit/onset.json").read_text())
    for expected, actual in zip(onset["rows"], data["nonregular_limits"]["onset_rows"]):
        for key in expected:
            compare(actual[key], expected[key])
    parameters = onset["parameters"]
    coefficient = parameters["gain"]*parameters["command"]*parameters["delay"]/parameters["thermal_time"]/np.sqrt(2*np.pi)
    compare(data["nonregular_limits"]["CLT_coefficient_C"], coefficient)
    result = {"status": "passed", "numeric_values_checked": total, "maximum_absolute_difference": maximum,
              "figure_data_sha256": manifest["plot_data_sha256"], "producer_sha256": sha(HERE/"audit_structural_figures.py"),
              "notice": "Independent analytic-family and bootstrap-draw reconstruction plus saved deterministic-limit array checks. Method verification and visual figure/page inspection are separate; no human scientific review is implied."}
    save_json(args.figures/"structural_numeric_audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
