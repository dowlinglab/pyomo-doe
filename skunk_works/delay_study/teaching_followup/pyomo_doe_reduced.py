"""A bounded Pyomo.DoE demonstration using the fitted three-parameter models.

Run with the existing pyomo-doe-maint environment. This supplements, rather
than changes, the frozen overnight study. No hardware experiment is run.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import expm
import pyomo
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.contrib.parmest.experiment import Experiment

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
KNOTS = np.arange(0.0, 901.0, 150.0)
OBSERVATIONS = np.arange(10.0, 901.0, 10.0)
SIGMA = 0.25


def nominal(order):
    path = STUDY / f"results/fits/joint__chain_n{order}_io_cold_linear_exact.json"
    fit = json.loads(path.read_text())["fit"]["parameters"]
    return np.array([fit[x] for x in ["K", "tau", "theta"]])


def exact_states(parameters, order, controls, times):
    """Independent matrix-exponential propagation with affine command segments."""
    K, tau, theta = parameters
    A = np.zeros((order + 1, order + 1))
    for j in range(order):
        A[j, j] = -order / theta
        if j:
            A[j, j - 1] = order / theta
    A[-1, -2], A[-1, -1] = K / tau, -1 / tau
    B = np.zeros(order + 1)
    B[0] = order / theta
    augmented = np.zeros((order + 3, order + 3))
    augmented[:order + 1, :order + 1] = A
    augmented[:order + 1, order + 1] = B
    augmented[order + 1, order + 2] = 1
    states = np.zeros((len(times), order + 1))
    u = np.interp(times, KNOTS, controls)
    transitions = {}
    for i, dt in enumerate(np.diff(times)):
        key = round(float(dt), 9)
        if key not in transitions:
            transitions[key] = expm(augmented * dt)
        initial = np.r_[states[i], u[i], (u[i + 1] - u[i]) / dt]
        states[i + 1] = (transitions[key] @ initial)[:order + 1]
    return states


def reference(parameters, order, controls, length):
    times = np.r_[0.0, OBSERVATIONS]
    # Every command knot is also on this grid; affine interpolation is exact.
    mean = exact_states(parameters, order, controls, times)[1:, -1]
    J = np.empty((len(mean), 3))
    for j in range(3):
        plus, minus = parameters.copy(), parameters.copy()
        plus[j] *= np.exp(1e-5)
        minus[j] *= np.exp(-1e-5)
        J[:, j] = (exact_states(plus, order, controls, times)[1:, -1]
                   - exact_states(minus, order, controls, times)[1:, -1]) / 2e-5
    if length:
        phi = np.exp(-10 / length)
        J[1:] = (J[1:] - phi * J[:-1]) / np.sqrt(1 - phi**2)
    F = J.T @ J / SIGMA**2
    return mean, F


class ReducedDelayExperiment(Experiment):
    def __init__(self, order=1, controls=None, length=0, nfe=90, enforce_limits=True):
        self.order, self.length, self.nfe = order, length, nfe
        self.enforce_limits = enforce_limits
        self.controls = np.full(7, 50.0) if controls is None else np.array(controls)
        self.parameters = nominal(order)

    def get_labeled_model(self):
        m = pyo.ConcreteModel()
        m.t = ContinuousSet(initialize=np.arange(0, 901, 10))
        m.stage = pyo.RangeSet(1, self.order)
        m.j = pyo.RangeSet(0, 6)
        m.K = pyo.Var(initialize=self.parameters[0], bounds=(0.2, 1.5))
        m.tau = pyo.Var(initialize=self.parameters[1], bounds=(5, 600))
        m.theta = pyo.Var(initialize=self.parameters[2], bounds=(0.01, 200))
        for v in [m.K, m.tau, m.theta]:
            v.fix()
        m.u = pyo.Var(m.j, initialize=lambda m, j: self.controls[j], bounds=(0, 100))
        for v in m.u.values():
            v.fix()
        m.z = pyo.Var(m.t, m.stage, initialize=0)
        m.y = pyo.Var(m.t, initialize=0,
                      bounds=(None, 85 - 22.2 - 0.01) if self.enforce_limits else (None, None))
        m.dz = DerivativeVar(m.z, wrt=m.t)
        m.dy = DerivativeVar(m.y, wrt=m.t)
        for j in m.stage:
            m.z[0, j].fix(0)
        m.y[0].fix(0)

        def command(m, t):
            j = min(int(t // 150), 5)
            weight = (t - 150 * j) / 150
            return (1 - weight) * m.u[j] + weight * m.u[j + 1]

        def lag(m, t, j):
            if t == 0:
                return pyo.Constraint.Skip
            upstream = command(m, t) if j == 1 else m.z[t, j - 1]
            return m.dz[t, j] == self.order / m.theta * (upstream - m.z[t, j])

        m.lag = pyo.Constraint(m.t, m.stage, rule=lag)
        m.thermal = pyo.Constraint(m.t, rule=lambda m, t: pyo.Constraint.Skip if t == 0
                                   else m.tau * m.dy[t] == m.K * m.z[t, self.order] - m.y[t])
        m.dose = pyo.Constraint(expr=150 * (m.u[0] / 2 + sum(m.u[j] for j in range(1, 6)) + m.u[6] / 2) <= 45000)
        m.slew = pyo.Constraint(range(6), rule=lambda m, j: pyo.inequality(-60, m.u[j + 1] - m.u[j], 60))
        if not self.enforce_limits:
            # A derivative of the mean still exists if an FD perturbation crosses
            # a design bound. Evaluate it, then audit the nominal input separately.
            m.dose.deactivate()
            m.slew.deactivate()
        pyo.TransformationFactory("dae.collocation").apply_to(m, nfe=self.nfe, ncp=3, scheme="LAGRANGE-RADAU")
        m.obs = pyo.Set(initialize=OBSERVATIONS.tolist(), ordered=True)
        m.measured = pyo.Var(m.obs, initialize=0)
        phi = np.exp(-10 / self.length) if self.length else 0

        def observation(m, t):
            if t == 10 or not self.length:
                return m.measured[t] == m.y[t]
            return m.measured[t] == (m.y[t] - phi * m.y[t - 10]) / np.sqrt(1 - phi**2)

        m.observation = pyo.Constraint(m.obs, rule=observation)
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update((m.u[j], None) for j in m.j)
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((v, pyo.value(v)) for v in [m.K, m.tau, m.theta])
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update((m.measured[t], None) for t in m.obs)
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update((m.measured[t], SIGMA) for t in m.obs)
        return m


def solver():
    result = pyo.SolverFactory("ipopt")
    result.options["tol"] = 1e-9
    result.options["max_iter"] = 1000
    result.options["bound_relax_factor"] = 0
    return result


def metrics(F, parameters):
    eig = np.linalg.eigvalsh(F)
    return {"FIM_log_parameters": F.tolist(), "eigenvalues": eig.tolist(),
            "rank": int(np.sum(eig > eig[-1] * 1e-10)),
            "log10_determinant": float(np.linalg.slogdet(F)[1] / np.log(10)),
            "local_parameter_sd": (parameters * np.sqrt(np.diag(np.linalg.inv(F)))).tolist()}


def evaluate(order, controls, length, nfe):
    exp = ReducedDelayExperiment(order, controls, length, nfe, enforce_limits=False)
    doe = DesignOfExperiments(experiment=exp, solver=solver(), step=0.001,
                             scale_nominal_param_value=True)
    doe.compute_FIM()
    F = np.array(doe.seq_FIM)
    mean, expected = reference(exp.parameters, order, controls, length)
    model = doe.compute_FIM_model
    # Sequential FIM leaves the final parameter perturbation in its model.
    for var, value in zip([model.K, model.tau, model.theta], exp.parameters):
        var.fix(value)
    result = solver().solve(model)
    pyo.assert_optimal_termination(result)
    observed = np.array([pyo.value(model.y[t]) for t in OBSERVATIONS])
    return {"controls": list(map(float, controls)), "nfe": nfe, **metrics(F, exp.parameters),
            "exact_FIM_relative_error": float(np.linalg.norm(F - expected) / np.linalg.norm(expected)),
            "exact_mean_max_error_C": float(np.max(np.abs(mean - observed))),
            "exact_reference": metrics(expected, exp.parameters)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, choices=[1, 2], default=1)
    parser.add_argument("--length", type=float, default=0, help="Known OU correlation length in seconds; zero means IID")
    parser.add_argument("--nfe", type=int, default=90)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    out = {"order": args.order, "working_correlation_length_s": args.length,
           "nominal_K_tau_theta": nominal(args.order).tolist(), "sigma_C": SIGMA,
           "observation_times_s": OBSERVATIONS.tolist(), "command_knots_s": KNOTS.tolist(),
           "baseline": evaluate(args.order, np.full(7, 50.0), args.length, args.nfe), "attempts": []}
    if args.optimize:
        for initial in [np.full(7, 50.0), np.array([30, 70, 30, 70, 30, 70, 30])]:
            exp = ReducedDelayExperiment(args.order, initial, args.length, args.nfe)
            doe = DesignOfExperiments(experiment=exp, solver=solver(), step=0.001,
                                     scale_nominal_param_value=True)
            doe.run_doe()
            assert str(doe.results["Termination Condition"]) == "optimal"
            controls = doe.get_experiment_input_values()
            row = evaluate(args.order, controls, args.length, args.nfe)
            row["initial_controls"] = initial.tolist()
            out["attempts"].append(row)
        best = max(out["attempts"], key=lambda x: x["exact_reference"]["log10_determinant"])
        out["selected"] = best
        out["refined_mesh"] = evaluate(args.order, best["controls"], args.length, args.nfe * 2)
        dense_times = np.arange(0, 900.01, 0.25)
        out["selected_dense_peak_C"] = float(22.2 + exact_states(nominal(args.order), args.order, best["controls"], dense_times)[:, -1].max())
        out["selected_dose_percent_seconds"] = float(np.trapezoid(best["controls"], KNOTS))
        out["selected_max_knot_change"] = float(np.max(np.abs(np.diff(best["controls"]))))
        assert out["selected_dense_peak_C"] <= 85
        assert out["selected_dose_percent_seconds"] <= 45000 + 1e-3
        assert out["selected_max_knot_change"] <= 60 + 1e-6
        out["D_efficiency_gain"] = float(10**((best["exact_reference"]["log10_determinant"] - out["baseline"]["exact_reference"]["log10_determinant"]) / 3))
    for row in [out["baseline"], *out["attempts"], *([out["refined_mesh"]] if args.optimize else [])]:
        assert row["exact_FIM_relative_error"] < 1e-3, row
        assert row["exact_mean_max_error_C"] < 1e-3, row
    out["elapsed_seconds"] = time.monotonic() - started
    out["identity"] = {"pyomo_version": pyomo.__version__, "python_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "pyomo_doe_source_sha256": hashlib.sha256(Path(inspect.getfile(DesignOfExperiments)).read_bytes()).hexdigest()}
    out["scope"] = "Local two-start D-optimal teaching demonstration, conditional on fitted cold model, fixed ambient/power and known IID or stationary OU covariance. OU outputs are prewhitened; no covariance parameters are estimated. Ninety observations, seven affine-input knots, 0–100 percent command, 45000 percent-seconds dose and 60-point knot changes. No global optimum, parameter-box coverage or hardware/continuous-time safety certificate; dense audit covers only the 900-second proposed window. Overnight report remains unchanged."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "elapsed_seconds": out["elapsed_seconds"],
                      "baseline_rank": out["baseline"]["rank"], "D_efficiency_gain": out.get("D_efficiency_gain")}))


if __name__ == "__main__":
    main()
