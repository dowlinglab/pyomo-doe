"""Independent saved-source checks for every uncertainty figure data array."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from run_fits import save_json
from profile_catalog import ProfileCatalog

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--figures", type=Path, default=HERE/"manuscript/figures")
    parser.add_argument("--profile-index",type=Path,help="Catalog filename relative to --results, or an absolute path; defaults to the preserved first pass")
    args = parser.parse_args()
    manifest = json.loads((args.figures/"uncertainty_figure_manifest.json").read_text())
    catalog = ProfileCatalog(args.results,args.profile_index)
    catalog.verify_manifest(manifest)
    path = args.figures/"uncertainty_plot_data.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["plot_data_sha256"]:
        raise RuntimeError("Uncertainty plotted-data bytes changed")
    for filename, expected in manifest["input_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Figure input changed: "+filename)
    data = json.loads(path.read_text())
    total, maximum = 0, 0.
    maximum_independent_physical_direction_error = 0.

    def compare(actual, expected, tolerance=1e-9):
        nonlocal total, maximum
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            raise RuntimeError("Plotted data shape changed")
        error = float(np.max(abs(actual-expected)))
        if error > tolerance:
            raise RuntimeError("Plotted value differs from independent source: "+str(error))
        maximum = max(maximum, error)
        total += actual.size
        return error

    p = data["profile_modes"]
    if Path(p["source"]).resolve() != catalog.profile_path("joint__chain_n1_io_cold_linear_exact__theta.json"):
        raise RuntimeError("Pole-mode figure uses another profile selection")
    profile = json.loads(Path(p["source"]).read_text())["profile"]
    compare(p["theta_s"], [point["parameter_value"] for point in profile["points"]])
    compare(p["known_lr_raw"], [point["lr_known_sigma_iid"] for point in profile["points"]])
    compare(p["unknown_lr_raw"], [point["lr_unknown_sigma_iid"] for point in profile["points"]])
    compare(p["threshold"], profile["chi2_95_df1"])
    bootstrap = pd.read_csv(args.results/"bootstrap/summary.csv")
    for case, rows in data["delay_intervals"].items():
        fit = json.loads((args.results/"fits"/(case+".json")).read_text())
        if len(rows) != 7:
            raise RuntimeError("An interval method was dropped")
        for row in rows[:5]:
            source = bootstrap[(bootstrap.case == case) & (bootstrap.method == row["method"]) &
                                (bootstrap.coordinates == "io") & (bootstrap.parameter == "theta")].iloc[0]
            compare([row["lower"], row["center"], row["upper"]], [source.lower, source['median'], source.upper])
        likelihood = json.loads(catalog.profile_path(case+"__theta.json").read_text())["profile"]
        point = fit["fit"]["parameters"]["theta"]
        component = next(c for c in likelihood["intervals_unknown_sigma_iid"] if c["lower"] <= point <= c["upper"])
        compare([rows[5]["lower"], rows[5]["center"], rows[5]["upper"]], [component["lower"], point, component["upper"]])
        ou = json.loads((args.results/"ou"/(case+".json")).read_text())
        # Independent inverse of the observed Hessian, rather than reusing the
        # saved covariance array that the figure producer reads.
        C = np.linalg.inv(ou["uncertainty"]["observed_profile_variance_hessian_log"])
        point = ou["fit"]["parameters"]["theta"]
        sigma = np.sqrt(C[2, 2])
        compare([rows[6]["lower"], rows[6]["center"], rows[6]["upper"]],
                [point*np.exp(-norm.ppf(.975)*sigma), point, point*np.exp(norm.ppf(.975)*sigma)])
    for series in data["information_spectra"]:
        fit = json.loads(Path(series["fit_path"]).read_text())
        u = fit["uncertainty"]
        values = np.asarray(u["eigenvalues_log_unit_variance"])[::-1]
        compare(series["relative_eigenvalues_descending"], values/values[0])
        if series["rank"] != u["rank_relative_1e-8"] or series["parameter_dimension"] != len(values):
            raise RuntimeError("Rank label changed")
    for i, order in enumerate(data["weak_directions"]["orders"]):
        family = "chain" if order else "pure_delay"
        for coordinates, key in (("io", "io"), ("physics", "physics")):
            filename = f"joint__{family}_n{order}_{coordinates}_cold_linear_exact.json"
            fit = json.loads((args.results/"fits"/filename).read_text())
            direction = np.array(fit["uncertainty"]["eigenvectors_log_columns"])[:, 0]
            direction *= np.sign(direction[np.argmax(abs(direction))])
            # Physical directions come from independently fitted physical
            # parameters/SVD, not the producer's congruence transformation.
            tolerance = 1e-5 if coordinates == "physics" else 1e-12
            error = compare(data["weak_directions"][key][i], direction, tolerance)
            if coordinates == "physics":
                maximum_independent_physical_direction_error = max(maximum_independent_physical_direction_error, error)
    result = {"status": "passed", "numeric_values_checked": total, "maximum_absolute_difference": maximum,
              "maximum_independent_physical_direction_error": maximum_independent_physical_direction_error,
              "physical_direction_tolerance": 1e-5,
              "producer_sha256": hashlib.sha256((HERE/"audit_uncertainty_figures.py").read_bytes()).hexdigest(),
              "figure_data_sha256": manifest["plot_data_sha256"],
              "profile_catalog":catalog.selection,"profile_selection_input_sha256":catalog.inputs,
              "support_source_sha256":hashlib.sha256((HERE/"profile_catalog.py").read_bytes()).hexdigest(),
              "notice": "Numerical provenance and independent saved-result comparisons, separate from color/grayscale and rendered-page inspection. No human scientific review is implied."}
    save_json(args.figures/"uncertainty_numeric_audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
