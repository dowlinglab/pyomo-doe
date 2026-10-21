"""Check rendered table cells through independent saved result paths."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from run_fits import save_json

HERE = Path(__file__).resolve().parent


def cells(path):
    result = {}
    for line in path.read_text().splitlines():
        if line.startswith("$n=") or line.startswith("Pure delay &"):
            row = [value.strip().rstrip("\\").strip() for value in line.split("&")]
            order = 0 if row[0] == "Pure delay" else int(re.search(r"n=(\d+)", row[0])[1])
            result[order] = row[1:]
    return result


def scalar(cell):
    text = cell.strip("$")
    scientific = re.fullmatch(r"([\d.]+)\\times10\^\{(-?\d+)\}", text)
    return float(scientific[1])*10**int(scientific[2]) if scientific else float(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--tables", type=Path, default=HERE/"manuscript/tables")
    args = parser.parse_args()
    summary = pd.read_csv(args.results/"fit_summary.csv").set_index("case")
    boot = pd.read_csv(args.results/"bootstrap/summary.csv")
    index = json.loads((args.results/"profile_index.json").read_text())["profiles"]
    checks, worst_rounding_fraction = 0, 0.

    def check(cell, expected, significant=None, decimals=None):
        nonlocal checks, worst_rounding_fraction
        observed = scalar(cell)
        radius = .5*10**(-decimals) if decimals is not None else .5*10**(np.floor(np.log10(abs(expected)))-significant+1)
        fraction = abs(observed-expected)/radius
        worst_rounding_fraction = max(worst_rounding_fraction, fraction)
        if fraction > 1+1e-5:
            raise RuntimeError(f"Table mismatch: {cell}, expected {expected}, rounding radius {radius}")
        checks += 1

    def case(order, coordinates="io", initialization="cold", calibration="joint"):
        family = "chain" if order else "pure_delay"
        return f"{calibration}__{family}_n{order}_{coordinates}_{initialization}_linear_exact"

    for order, row in cells(args.tables/"calibration_rmse.tex").items():
        references = [summary.loc[case(order, calibration=c), "rmse_C"] for c in ("sine", "step", "joint")]
        references.append(summary.loc[case(order, initialization="input_equilibrated"), "rmse_C"])
        for cell, reference in zip(row, references):
            check(cell, reference, decimals=4)
    for order, row in cells(args.tables/"joint_parameters.tex").items():
        physical = json.loads((args.results/"fits"/(case(order, "physics")+".json")).read_text())["fit"]["parameters"]
        # Independently fitted physical parameters, rather than the producer's
        # forward transform of the IO fit, check all five printed quantities.
        Ua, Cp, theta = [physical[name] for name in ("Ua", "Cp", "theta")]
        for cell, reference, decimals in zip(row, [.032/Ua, Cp/Ua, theta, Ua, Cp], [4, 3, 3, 5, 3]):
            check(cell, reference, decimals=decimals)
    for order, row in cells(args.tables/"coordinate_metrics.tex").items():
        io = json.loads((args.results/"fits"/(case(order)+".json")).read_text())
        physical = json.loads((args.results/"fits"/(case(order, "physics")+".json")).read_text())
        p = io["fit"]["parameters"]
        B = np.array([[1., 0., 0.], [0., p["tau"]/(p["tau"]+p["theta"]), p["theta"]/(p["tau"]+p["theta"])], [0., -1., 1.]])
        C = B@np.asarray(io["uncertainty"]["cov_log_iid_unknown_sigma"])@B.T
        values = np.linalg.eigvalsh(C)
        references = [io["uncertainty"]["condition_J_log"], physical["uncertainty"]["condition_J_log"], np.sqrt(values[-1]/values[0])]
        for cell, reference in zip(row[:3], references):
            check(cell, reference, significant=4)
        check(row[3], abs(summary.loc[case(order), "sse"]-summary.loc[case(order, "physics"), "sse"]), significant=2)
    for order, row in cells(args.tables/"delay_uncertainty.tex").items():
        selected = index[case(order)+"__theta.json"]
        p = json.loads((args.results/selected["path"]).read_text())["profile"]
        displayed = re.findall(r"\[([\d.]+),([\d.]+)\]", row[0])
        components = p["intervals_unknown_sigma_iid"]
        if len(displayed) != len(components):
            raise RuntimeError("A disconnected profile component was lost")
        for pair, component in zip(displayed, components):
            for cell, reference in zip(pair, [component["lower"], component["upper"]]):
                check(cell, reference, significant=4)
        for cell, method in zip(row[1:3], ("parametric", "block60")):
            reference = boot[(boot.case == case(order)) & (boot.coordinates == "io") &
                             (boot.parameter == "theta") & (boot.method == method)].iloc[0]
            pair = re.fullmatch(r"\$\[([\d.]+),([\d.]+)\]\$", cell)
            check(pair[1], reference.lower, significant=4)
            check(pair[2], reference.upper, significant=4)
        ou = json.loads((args.results/"ou"/(case(order)+".json")).read_text())
        inverse_hessian = np.linalg.inv(np.asarray(ou["uncertainty"]["observed_profile_variance_hessian_log"]))
        check(row[3], ou["fit"]["parameters"]["theta"]*np.sqrt(inverse_hessian[2, 2]), significant=3)
    diagnostic = json.loads((args.results/"ou/residual_diagnostics.json").read_text())
    for order, row in cells(args.tables/"ou_diagnostics.tex").items():
        ou = json.loads((args.results/"ou"/(case(order)+".json")).read_text())
        references = [ou["fit"]["correlation_length_s"], ou["fit"]["sigma_stationary_ML_C"]]
        references.extend(diagnostic[case(order)][name]["innovation_C"]["acf_lags_1_to_30"][0] for name in ("sine", "step"))
        for cell, reference in zip(row, references):
            check(cell, reference, significant=4)
    result = {"status": "passed", "numeric_cells_checked": checks, "maximum_fraction_of_rounding_radius": worst_rounding_fraction,
              "producer_sha256": hashlib.sha256((HERE/"audit_tables.py").read_bytes()).hexdigest(),
              "table_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.tables.glob("*.tex")},
              "independence": "RMSE/SSE use summary CSV paths; parameters use independent physical fits; physical condition numbers use independent sensitivity SVDs; moment conditioning uses transformed covariance rather than the producer's information congruence; OU SD uses the observed Hessian inverse. Profile/bootstrap cells are checked against their separately audited saved sources.",
              "notice": "Printed numerical/rounding checks are separate from layout inspection and scientific review."}
    save_json(args.tables/"table_numeric_audit.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("table_sha256", "independence", "notice")}))


if __name__ == "__main__":
    main()
