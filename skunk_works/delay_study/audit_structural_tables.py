"""Verify printed boundary and correlation cells by independent result paths."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_fits import save_json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--tables", type=Path, default=HERE/"manuscript/tables")
    args = parser.parse_args()
    manifest = json.loads((args.tables/"structural_table_manifest.json").read_text())
    for name, expected in manifest["outputs"].items():
        if sha(args.tables/name) != expected:
            raise RuntimeError("Generated structural table changed")
    for name, expected in manifest["input_sha256"].items():
        if sha(Path(name)) != expected:
            raise RuntimeError("Structural table source changed")
    values = json.loads((args.tables/"structural_table_values.json").read_text())
    boundary_lines = [line for line in (args.tables/"family_limits.tex").read_text().splitlines()
                      if line.startswith("$\\inf") or line.startswith("$\\sup")]
    correlation_lines = [line for line in (args.tables/"coordinate_correlations.tex").read_text().splitlines()
                         if line.startswith("$n=") or line.startswith("Pure delay")]
    if len(boundary_lines) != 4 or len(correlation_lines) != 8:
        raise RuntimeError("Generated table lost a scientific row")
    parameters = json.loads((args.results/"fits/joint__chain_n1_io_cold_linear_exact.json").read_text())["fit"]["parameters"]
    point = np.array([parameters[name] for name in ("K", "tau", "theta")])

    def functions(p):
        K, t, d = np.asarray(p).T
        return np.array([.032/K*np.minimum(t, d), .032/K*np.maximum(t, d),
                         .032/K*(t+d-2*np.sqrt(t*d)), .032/K*((t+d)**2/(4*t*d)-1)]).T

    independent = {}
    for method in ("parametric", "block60"):
        path = args.results/"bootstrap"/("joint__chain_n1_io_cold_linear_exact__"+method+".npz")
        independent[method] = np.quantile(functions(np.load(path)["parameters"]), [.025, .975], axis=0)
    names = ["CpH_inf_J_K", "CpH_sup_J_K", "CpS_sup_J_K", "Ub_sup_W_K"]
    count, maximum, maximum_correlation = 0, 0., 0.
    for j, name in enumerate(names):
        expected = [float(functions(point)[j]), *independent["parametric"][:, j].tolist(), *independent["block60"][:, j].tolist()]
        error = float(np.max(abs(np.asarray(values["boundary_functions"][name])-expected)))
        if error > 1e-10:
            raise RuntimeError("Boundary table differs from independent bootstrap draws")
        cells = [f"{expected[0]:.4g}", "$["+f"{expected[1]:.4g},{expected[2]:.4g}"+"]$",
                 "$["+f"{expected[3]:.4g},{expected[4]:.4g}"+"]$"]
        if [cell.strip().removesuffix(r" \\").strip() for cell in boundary_lines[j].split(" & ")[1:]] != cells:
            raise RuntimeError("Printed boundary cell changed")
        count += len(expected)
        maximum = max(maximum, error)
    for i, order in enumerate([1, 2, 12, 0]):
        model = "chain_n"+str(order) if order else "pure_delay_n0"
        for offset, (metric, coordinates) in enumerate([("IO", "io"), ("Physical", "physics")]):
            path = args.results/"fits"/("joint__"+model+"_"+coordinates+"_cold_linear_exact.json")
            correlation = np.asarray(json.loads(path.read_text())["uncertainty"]["correlation_iid"])
            expected = correlation[[0, 0, 1], [1, 2, 2]]
            error = float(np.max(abs(np.asarray(values["correlations"][str(order)+"_"+metric])-expected)))
            if error > 1e-5:
                raise RuntimeError("Correlation differs from independently fitted coordinate result")
            cells = [f"{value:.3f}" for value in expected]
            actual = [cell.strip().removesuffix(r" \\").strip() for cell in correlation_lines[2*i+offset].split(" & ")[1:]]
            if actual != cells:
                raise RuntimeError("Printed correlation cell changed")
            count += 3
            maximum_correlation = max(maximum_correlation, error)
    receipt = {"status": "passed", "numeric_values_checked": count, "maximum_boundary_difference": maximum,
               "maximum_independent_coordinate_correlation_difference": maximum_correlation,
               "independent_coordinate_tolerance": 1e-5, "producer_sha256": sha(HERE/"audit_structural_tables.py"),
               "notice": "Printed cells checked against independently reconstructed bootstrap quantiles and separately fitted physical-coordinate correlations. Rendered-page inspection and human scientific review are separate."}
    save_json(args.tables/"structural_numeric_audit.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
