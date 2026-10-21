"""Check the Pyomo demonstration against the study's separate exact simulator.

Use the original study environment: no Pyomo imports or solver calls here.
Full observation covariance inversion also checks the OU whitening construction.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data import Experiment
from inference import ModelSpec, sensitivities, predict


def main():
    source_hash = hashlib.sha256((HERE / "pyomo_doe_reduced.py").read_bytes()).hexdigest()
    rows, peaks, inputs = [], [], {}
    knots = np.arange(0, 901, 150)
    for filename in ["n1_iid", "n2_iid", "n1_ou60", "n2_ou60"]:
        path = HERE / "results" / (filename + ".json")
        d = json.loads(path.read_text())
        inputs[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert d["identity"]["python_source_sha256"] == source_hash
        parameters = np.asarray(d["nominal_K_tau_theta"])
        spec = ModelSpec(order=d["order"])
        t = np.r_[0, d["observation_times_s"]]
        obs = t[1:]
        length = d["working_correlation_length_s"]
        covariance = (np.exp(-np.abs(obs[:, None] - obs[None, :]) / length)
                      if length else np.eye(len(obs))) * d["sigma_C"]**2
        for label, row in [("baseline", d["baseline"]),
                           *[(f"attempt_{i}", r) for i, r in enumerate(d["attempts"])],
                           ("refined_mesh", d["refined_mesh"])]:
            data = Experiment(filename, t, np.full(len(t), 22.2),
                              np.interp(t, knots, row["controls"]), 22.2)
            J = sensitivities(parameters, [data], spec, log=True)
            F = J.T @ np.linalg.solve(covariance, J)
            actual = np.array(row["FIM_log_parameters"])
            logdet = np.linalg.slogdet(F)[1] / np.log(10)
            sd = parameters * np.sqrt(np.diag(np.linalg.inv(F)))
            delta = {"case": filename, "evaluation": label,
                     "FIM_relative_error": float(np.linalg.norm(actual - F) / np.linalg.norm(F)),
                     "log10_determinant_absolute_error": float(abs(logdet - row["log10_determinant"])),
                     "local_sd_max_relative_error": float(np.max(np.abs(np.asarray(row["local_parameter_sd"]) - sd) / sd))}
            assert delta["FIM_relative_error"] < 1e-3, delta
            assert delta["log10_determinant_absolute_error"] < 0.01, delta
            assert delta["local_sd_max_relative_error"] < 0.005, delta
            rows.append(delta)
        selected = d["selected"]["controls"]
        dense = np.arange(0, 900.01, 0.25)
        data = Experiment(filename, dense, np.full(len(dense), 22.2),
                          np.interp(dense, knots, selected), 22.2)
        temperature = predict(data, parameters, spec)
        peak = float(np.max(temperature))
        assert np.min(temperature) >= 22.2 - 1e-9
        assert peak <= 85 and abs(peak - d["selected_dense_peak_C"]) < 1e-8
        assert min(selected) >= -1e-6 and max(selected) <= 100 + 1e-6
        assert np.trapz(selected, knots) <= 45000 + 1e-3
        assert np.max(np.abs(np.diff(selected))) <= 60 + 1e-6
        peaks.append({"case": filename, "dense_peak_C": peak})
    out = {"status": "passed", "evaluations": rows, "dense_peaks": peaks,
           "input_sha256": inputs, "producer_sha256": source_hash,
           "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "scope": "All sixteen saved baseline/start/refinement FIMs and local standard deviations compared to complex-step sensitivities from the separately verified overnight simulator with explicit full covariance inversion. All four selected commands checked on a separate 0.25-second exact trajectory and for dose, bounds and knot changes. No global optimization, continuous-time safety or real-data interval coverage certificate."}
    (HERE / "results/audit.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "evaluations": len(rows),
                      "max_FIM_relative_error": max(r["FIM_relative_error"] for r in rows),
                      "max_SD_relative_error": max(r["local_sd_max_relative_error"] for r in rows)}))


if __name__ == "__main__":
    main()
