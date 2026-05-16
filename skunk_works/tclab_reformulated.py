"""Simplified TCLab model builders.

This module keeps the core TCLab dynamics in two forms:

1. The original second-order heater/sensor model.
2. A delayed single-state reformulation with an n-stage lag chain.

The goal is to provide a smaller, easier-to-read starting point for
simulation and later parameter estimation.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Param,
    RangeSet,
    Suffix,
    SolverFactory,
    TransformationFactory,
    Var,
    value as pyovalue,
)
from pyomo.dae import ContinuousSet, DerivativeVar


@dataclass
class TCLabData:
    """Container for a single TCLab experiment.

    Parameters
    ----------
    time:
        Time stamps in seconds.
    T:
        Measured sensor temperature in degC.
    u:
        Heater input signal in percent power.
    Tamb:
        Ambient temperature in degC.
    P:
        Maximum heater power in watts.
    name:
        Optional experiment label.
    """

    time: Sequence[float]
    T: Sequence[float]
    u: Sequence[float]
    Tamb: float
    P: float = 200.0
    name: str | None = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.T = np.asarray(self.T, dtype=float)
        self.u = np.asarray(self.u, dtype=float)
        if len(self.time) != len(self.T) or len(self.time) != len(self.u):
            raise ValueError("time, T, and u must have the same length")
        if len(self.time) == 0:
            raise ValueError("time must contain at least one point")
        if np.any(np.diff(self.time) <= 0):
            raise ValueError("time must be strictly increasing")

    def to_dataframe(self) -> pd.DataFrame:
        """Return the experiment as a pandas DataFrame."""

        return pd.DataFrame({"time": self.time, "T": self.T, "u": self.u})


class TeeStream:
    """Mirror writes to multiple text streams."""

    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def helper(values: Sequence[float], time: Sequence[float]) -> dict[float, float]:
    """Build a time-indexed dictionary for Pyomo initialization."""

    if len(values) != len(time):
        raise ValueError("values and time must have the same length")
    data: dict[float, float] = {}
    for t, v in zip(time, values):
        data[float(t)] = 0.0 if v is None else float(v)
    return data


def default_original_theta() -> dict[str, float]:
    """Reasonable initial guesses for the second-order model."""

    return {
        "Ua": 0.0535,
        "Ub": 0.0148,
        "CpH": 6.911,
        "CpS": 0.318,
    }


def default_delayed_theta(parameterization: str = "physics") -> dict[str, float]:
    """Reasonable initial guesses for the delayed single-state model."""

    if parameterization == "physics":
        return {
            "Ua": 0.030,
            "Cp": 7.50,
            "theta": 15.0,
        }
    if parameterization == "io":
        return {
            "K": 1.0,
            "tau": 250.0,
            "theta": 15.0,
        }
    raise ValueError("parameterization must be 'physics' or 'io'")


def discretize_model(
    model: ConcreteModel,
    nfe: int | None = None,
    scheme: str = "BACKWARD",
) -> ConcreteModel:
    """Apply a finite-difference discretization to a DAE model."""

    if nfe is None:
        nfe = max(len(list(model.t)) - 1, 1)
    TransformationFactory("dae.finite_difference").apply_to(
        model, scheme=scheme, nfe=nfe
    )
    return model


def _fix_initial_temperature(model: ConcreteModel, data: TCLabData) -> None:
    """Fix the measured temperature state at the first time point."""

    t0 = float(data.time[0])
    T0 = float(data.T[0]) if data.T[0] is not None else float(data.Tamb)
    model.Tm[t0].fix(T0)

    if hasattr(model, "Th"):
        model.Th[t0].fix(T0)


def _fix_delay_states(model: ConcreteModel, data: TCLabData, delay_order: int) -> None:
    """Fix all delay states at the first time point."""

    t0 = float(data.time[0])
    u0 = float(data.u[0]) if data.u[0] is not None else 0.0
    for i in range(1, delay_order + 1):
        model.z[i, t0].fix(u0)


def _parameter_var(model: ConcreteModel, name: str) -> Var:
    """Convenience accessor for a named Pyomo Var."""

    return getattr(model, name)


def _fix_parameters(model: ConcreteModel, theta: dict[str, float], names: Sequence[str]) -> list[Var]:
    """Create and optionally fix the parameter vars used in a model."""

    params: list[Var] = []
    for name in names:
        var = _parameter_var(model, name)
        var.set_value(theta[name])
        var.fix()
        params.append(var)
    return params


def _label_single_output_experiment(
    model: ConcreteModel,
    data: TCLabData,
    parameter_vars: Sequence[Var],
    measurement_error: float,
) -> None:
    """Add Parmest-style suffixes for a single-output experiment."""

    model.experiment_outputs = Suffix(direction=Suffix.LOCAL)
    model.experiment_outputs.update(
        (model.Tm[float(t)], float(y)) for t, y in zip(data.time, data.T)
    )

    model.unknown_parameters = Suffix(direction=Suffix.LOCAL)
    for p in parameter_vars:
        model.unknown_parameters[p] = pyovalue(p)

    model.experiment_inputs = Suffix(direction=Suffix.LOCAL)
    model.experiment_inputs.update((model.u[float(t)], None) for t in data.time)

    model.measurement_error = Suffix(direction=Suffix.LOCAL)
    model.measurement_error.update(
        (model.Tm[float(t)], measurement_error) for t in data.time
    )


def build_original_tclab_model(
    data: TCLabData,
    theta: dict[str, float] | None = None,
    alpha: float = 0.00016,
    fix_parameters: bool = True,
    fix_input: bool = True,
    discretize: bool = False,
    nfe: int | None = None,
    measurement_error: float = 0.25,
    label: bool = False,
) -> ConcreteModel:
    """Build the original second-order TCLab model.

    The model uses one heater state ``Th`` and one measured sensor state ``Tm``:

    ``CpH * dTh/dt = Ua*(Tamb - Th) + Ub*(Tm - Th) + alpha*P*u``
    ``CpS * dTm/dt = Ub*(Th - Tm)``
    """

    theta = default_original_theta() if theta is None else theta.copy()
    required = {"Ua", "Ub", "CpH", "CpS"}
    missing = required.difference(theta)
    if missing:
        raise KeyError(f"Missing original-model initial values: {sorted(missing)}")

    m = ConcreteModel()
    m.Tamb = Param(initialize=float(data.Tamb))
    m.P = Param(initialize=float(data.P))
    m.alpha = Param(initialize=float(alpha))
    m.Tmax = 85.0

    m.t = ContinuousSet(initialize=[float(t) for t in data.time])
    m.Th = Var(m.t, bounds=(0, m.Tmax), initialize=float(data.Tamb))
    m.Tm = Var(m.t, bounds=(0, m.Tmax), initialize=float(data.Tamb))
    m.dThdt = DerivativeVar(m.Th, wrt=m.t)
    m.dTmdt = DerivativeVar(m.Tm, wrt=m.t)

    m.u = Var(m.t, bounds=(0, 100), initialize=helper(data.u, data.time))
    if fix_input:
        m.u.fix()

    m.Ua = Var(bounds=(1e-6, 0.1), initialize=float(theta["Ua"]))
    m.Ub = Var(bounds=(1e-6, 0.1), initialize=float(theta["Ub"]))
    m.CpH = Var(bounds=(1e-3, 100.0), initialize=float(theta["CpH"]))
    m.CpS = Var(bounds=(1e-3, 100.0), initialize=float(theta["CpS"]))
    if fix_parameters:
        _fix_parameters(m, theta, ["Ua", "Ub", "CpH", "CpS"])

    @m.Constraint(m.t)
    def heater_energy_balance(m, t):
        return (
            m.CpH * m.dThdt[t]
            == m.Ua * (m.Tamb - m.Th[t])
            + m.Ub * (m.Tm[t] - m.Th[t])
            + m.alpha * m.P * m.u[t]
        )

    @m.Constraint(m.t)
    def sensor_energy_balance(m, t):
        return m.CpS * m.dTmdt[t] == m.Ub * (m.Th[t] - m.Tm[t])

    _fix_initial_temperature(m, data)

    if discretize:
        discretize_model(m, nfe=nfe)

    if label:
        _label_single_output_experiment(
            m, data, [m.Ua, m.Ub, m.CpH, m.CpS], measurement_error
        )

    return m


def build_reformulated_tclab_model(
    data: TCLabData,
    theta: dict[str, float] | None = None,
    alpha: float = 0.00016,
    parameterization: str = "physics",
    delay_order: int = 3,
    fix_parameters: bool = True,
    fix_input: bool = True,
    discretize: bool = False,
    nfe: int | None = None,
    measurement_error: float = 0.25,
    label: bool = False,
) -> ConcreteModel:
    """Build the delayed single-state TCLab model.

    Parameters
    ----------
    parameterization:
        ``"physics"`` uses ``Ua``, ``Cp``, and ``theta``.
        ``"io"`` uses ``K``, ``tau``, and ``theta``.
    delay_order:
        Number of first-order lag states used to approximate the delay.
    """

    if delay_order < 1:
        raise ValueError("delay_order must be at least 1")

    theta = default_delayed_theta(parameterization) if theta is None else theta.copy()

    if parameterization == "physics":
        required = {"Ua", "Cp", "theta"}
    elif parameterization == "io":
        required = {"K", "tau", "theta"}
    else:
        raise ValueError("parameterization must be 'physics' or 'io'")

    missing = required.difference(theta)
    if missing:
        raise KeyError(f"Missing delayed-model initial values: {sorted(missing)}")

    m = ConcreteModel()
    m.Tamb = Param(initialize=float(data.Tamb))
    m.P = Param(initialize=float(data.P))
    m.alpha = Param(initialize=float(alpha))
    m.Tmax = 85.0

    m.t = ContinuousSet(initialize=[float(t) for t in data.time])
    m.Tm = Var(m.t, bounds=(0, m.Tmax), initialize=float(data.Tamb))
    m.dTmdt = DerivativeVar(m.Tm, wrt=m.t)

    m.u = Var(m.t, bounds=(0, 100), initialize=helper(data.u, data.time))
    if fix_input:
        m.u.fix()

    m.delay_index = RangeSet(1, delay_order)
    m.z = Var(m.delay_index, m.t, bounds=(0, 100), initialize=float(data.u[0]))
    m.dzdt = DerivativeVar(m.z, wrt=m.t)

    if parameterization == "physics":
        m.Ua = Var(bounds=(1e-6, 0.1), initialize=float(theta["Ua"]))
        m.Cp = Var(bounds=(1e-3, 100.0), initialize=float(theta["Cp"]))
        m.theta = Var(bounds=(1e-6, 1e7), initialize=float(theta["theta"]))
        if fix_parameters:
            _fix_parameters(m, theta, ["Ua", "Cp", "theta"])
    else:
        m.K = Var(bounds=(1e-6, 100.0), initialize=float(theta["K"]))
        m.tau = Var(bounds=(1e-6, 1e5), initialize=float(theta["tau"]))
        m.theta = Var(bounds=(1e-6, 1e7), initialize=float(theta["theta"]))
        if fix_parameters:
            _fix_parameters(m, theta, ["K", "tau", "theta"])

    @m.Constraint(m.t)
    def delay_chain(m, t):
        return m.dzdt[1, t] == (delay_order / m.theta) * (m.u[t] - m.z[1, t])

    @m.Constraint(m.t, m.delay_index)
    def delay_chain_midpoints(m, t, i):
        if delay_order == 1 or i == 1:
            return Constraint.Skip
        return m.dzdt[i, t] == (delay_order / m.theta) * (
            m.z[i - 1, t] - m.z[i, t]
        )

    if parameterization == "physics":
        @m.Constraint(m.t)
        def sensor_energy_balance(m, t):
            return (
                m.Cp * m.dTmdt[t]
                == m.Ua * (m.Tamb - m.Tm[t])
                + m.alpha * m.P * m.z[delay_order, t]
            )

    else:
        @m.Constraint(m.t)
        def sensor_energy_balance(m, t):
            return m.dTmdt[t] == -(m.Tm[t] - m.Tamb) / m.tau + (
                m.K / m.tau
            ) * m.z[delay_order, t]

    _fix_initial_temperature(m, data)
    _fix_delay_states(m, data, delay_order)

    if discretize:
        discretize_model(m, nfe=nfe)

    if label:
        if parameterization == "physics":
            parameter_vars = [m.Ua, m.Cp, m.theta]
        else:
            parameter_vars = [m.K, m.tau, m.theta]
        _label_single_output_experiment(m, data, parameter_vars, measurement_error)

    return m


def build_tclab_model(
    data: TCLabData,
    variant: str = "original",
    **kwargs: Any,
) -> ConcreteModel:
    """Dispatch helper for the two supported model variants."""

    variant = variant.lower().strip()
    if variant in {"original", "second_order", "2state"}:
        return build_original_tclab_model(data, **kwargs)
    if variant in {"reformulated", "delayed", "single_state", "1state"}:
        return build_reformulated_tclab_model(data, **kwargs)
    raise ValueError("variant must be 'original' or 'reformulated'")


def physics_to_io_parameters(Ua: float, Cp: float, alpha: float, P: float) -> dict[str, float]:
    """Convert delayed-model physics parameters to input-output parameters."""

    tau = Cp / Ua
    K = alpha * P / Ua
    return {"K": K, "tau": tau}


def io_to_physics_parameters(
    K: float,
    tau: float,
    alpha: float,
    P: float,
) -> dict[str, float]:
    """Convert delayed-model input-output parameters to physics parameters."""

    Ua = alpha * P / K
    Cp = Ua * tau
    return {"Ua": Ua, "Cp": Cp}


def solution_to_dataframe(model: ConcreteModel) -> pd.DataFrame:
    """Extract a solved model into a tidy DataFrame."""

    rows: list[dict[str, float]] = []
    time_points = [float(t) for t in model.t]
    for t in time_points:
        row: dict[str, float] = {"time": t}
        if hasattr(model, "Th"):
            row["Th"] = float(pyovalue(model.Th[t]))
        if hasattr(model, "Tm"):
            row["Tm"] = float(pyovalue(model.Tm[t]))
        if hasattr(model, "u"):
            row["u"] = float(pyovalue(model.u[t]))
        if hasattr(model, "z"):
            for i in model.delay_index:
                row[f"z{i}"] = float(pyovalue(model.z[i, t]))
        rows.append(row)
    return pd.DataFrame(rows)


def parameter_values(model: ConcreteModel) -> dict[str, float]:
    """Return the scalar parameter values currently stored on a model."""

    names = ["Ua", "Ub", "CpH", "CpS", "Cp", "theta", "K", "tau"]
    values: dict[str, float] = {}
    for name in names:
        if hasattr(model, name):
            values[name] = float(pyovalue(getattr(model, name)))
    return values


def load_sine_wave_dataset(
    csv_path: str | Path | None = None,
) -> TCLabData:
    """Load the sine-wave TCLab dataset used in ``parmest.ipynb``."""

    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / "tclab_sine_test_5min_period.csv"
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    return TCLabData(
        name="Sine Wave Test for Heater 1",
        time=df["Time"].values,
        T=df["T1"].values,
        u=df["Q1"].values,
        Tamb=float(df["T1"].values[0]),
        P=200.0,
    )


def find_ipopt_executable() -> Path | None:
    """Find a usable Ipopt executable on this machine."""

    env_ipopt = os.environ.get("IPOPT_EXEC")
    if env_ipopt:
        candidate = Path(env_ipopt).expanduser()
        if candidate.exists():
            return candidate

    env_prefix = os.environ.get("IPOPT_PREFIX")
    if env_prefix:
        candidate = Path(env_prefix).expanduser() / "bin" / "ipopt"
        if candidate.exists():
            return candidate

    known_candidates = [
        Path("~/opt/ipopt-hsl-pyomo/ipopt/bin/ipopt").expanduser(),
        Path("~/.idaes/bin/ipopt").expanduser(),
    ]
    for candidate in known_candidates:
        if candidate.exists():
            return candidate

    path_ipopt = shutil.which("ipopt")
    if path_ipopt:
        return Path(path_ipopt).expanduser()

    return None


def ensure_ipopt_on_path() -> Path:
    """Prepend the directory containing Ipopt to ``PATH`` if needed."""

    ipopt_exec = find_ipopt_executable()
    if ipopt_exec is None:
        raise FileNotFoundError(
            "Unable to locate Ipopt via IPOPT_EXEC, IPOPT_PREFIX, ~/.idaes/bin, or ~/opt/ipopt-hsl-pyomo."
        )
    solver_dir = str(ipopt_exec.parent)
    current_path = os.environ.get("PATH", "")
    if solver_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = solver_dir + os.pathsep + current_path if current_path else solver_dir
    os.environ["IPOPT_EXEC"] = str(ipopt_exec)
    return ipopt_exec


@dataclass
class EstimationResult:
    """Container for one model-variant fit."""

    variant: str
    delay_order: int
    parameterization: str
    objective: str
    theta_hat: pd.Series
    covariance: pd.DataFrame
    transformed_theta: pd.Series
    transformed_covariance: pd.DataFrame
    fitted_data: pd.DataFrame
    fit_figure: plt.Figure
    covariance_figure: plt.Figure
    rmse: float
    mae: float
    multistart_figure: plt.Figure | None = None
    multistart_results: pd.DataFrame | None = None
    multistart_best_objective: float | None = None
    multistart_sampling_method: str | None = None
    multistart_n_restarts: int | None = None
    multistart_seed: int | None = None


@dataclass
class TCLabParmestExperiment:
    """Parmest-compatible experiment wrapper for TCLab model variants."""

    data: TCLabData
    variant: str = "reformulated"
    parameterization: str = "physics"
    delay_order: int = 2
    alpha: float = 0.00016
    measurement_error: float = 0.25
    theta: dict[str, float] | None = None
    discretize: bool = True
    fix_input: bool = True

    def get_labeled_model(self) -> ConcreteModel:
        if self.variant == "original":
            return build_original_tclab_model(
                self.data,
                theta=self.theta,
                alpha=self.alpha,
                fix_parameters=False,
                fix_input=self.fix_input,
                discretize=self.discretize,
                label=True,
                measurement_error=self.measurement_error,
            )
        return build_reformulated_tclab_model(
            self.data,
            theta=self.theta,
            alpha=self.alpha,
            parameterization=self.parameterization,
            delay_order=self.delay_order,
            fix_parameters=False,
            fix_input=self.fix_input,
            discretize=self.discretize,
            label=True,
            measurement_error=self.measurement_error,
        )


def _theta_and_cov_names(parameterization: str) -> list[str]:
    if parameterization == "physics":
        return ["Ua", "Cp", "theta"]
    if parameterization == "io":
        return ["K", "tau", "theta"]
    raise ValueError("parameterization must be 'physics' or 'io'")


def theta_bounds(
    variant: str,
    parameterization: str,
) -> dict[str, tuple[float, float]]:
    """Return hard bounds for estimated parameters."""

    if variant == "original":
        return {
            "Ua": (1e-6, 0.1),
            "Ub": (1e-6, 0.1),
            "CpH": (1e-3, 100.0),
            "CpS": (1e-3, 100.0),
        }
    if variant == "reformulated" and parameterization == "physics":
        return {
            "Ua": (1e-6, 0.1),
            "Cp": (1e-3, 100.0),
            "theta": (1e-6, 1e7),
        }
    if variant == "reformulated" and parameterization == "io":
        return {
            "K": (1e-6, 100.0),
            "tau": (1e-6, 1e5),
            "theta": (1e-6, 1e7),
        }
    raise ValueError("Unsupported variant / parameterization combination.")


def clip_theta_to_bounds(
    theta: pd.Series | dict[str, float],
    variant: str,
    parameterization: str,
) -> pd.Series:
    """Project parameter values onto the feasible box."""

    bounds = theta_bounds(variant, parameterization)
    series = pd.Series(theta, dtype=float).copy()
    for name, (lb, ub) in bounds.items():
        if name in series.index:
            series[name] = float(np.clip(series[name], lb, ub))
    return series


def transform_theta_and_covariance(
    theta_hat: pd.Series,
    covariance: pd.DataFrame,
    parameterization: str,
    alpha: float,
    P: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Transform the fitted parameters and covariance to the alternate form."""

    cov = pd.DataFrame(
        0.5 * (covariance.values + covariance.values.T),
        index=covariance.index,
        columns=covariance.columns,
    )

    if parameterization == "physics":
        Ua = float(theta_hat["Ua"])
        Cp = float(theta_hat["Cp"])
        theta = float(theta_hat["theta"])
        transformed_theta = pd.Series(
            {"K": alpha * P / Ua, "tau": Cp / Ua, "theta": theta}
        )
        jac = np.array(
            [
                [-(alpha * P) / (Ua**2), 0.0, 0.0],
                [-(Cp) / (Ua**2), 1.0 / Ua, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transformed_names = ["K", "tau", "theta"]
    elif parameterization == "io":
        K = float(theta_hat["K"])
        tau = float(theta_hat["tau"])
        theta = float(theta_hat["theta"])
        transformed_theta = pd.Series(
            {"Ua": alpha * P / K, "Cp": alpha * P * tau / K, "theta": theta}
        )
        jac = np.array(
            [
                [-(alpha * P) / (K**2), 0.0, 0.0],
                [-(alpha * P * tau) / (K**2), (alpha * P) / K, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transformed_names = ["Ua", "Cp", "theta"]
    else:
        raise ValueError("parameterization must be 'physics' or 'io'")

    transformed_cov = jac @ cov.loc[theta_hat.index, theta_hat.index].values @ jac.T
    transformed_cov = pd.DataFrame(
        transformed_cov, index=transformed_names, columns=transformed_names
    )
    return transformed_theta, transformed_cov


def confidence_interval_from_covariance(
    theta_hat: pd.Series,
    covariance: pd.DataFrame,
    n_data: int,
) -> pd.DataFrame:
    """Compute approximate 95% confidence intervals from a covariance matrix."""

    dof = max(n_data - len(theta_hat), 1)
    tcrit = float(stats.t.ppf(0.975, dof))
    se = np.sqrt(np.diag(covariance.loc[theta_hat.index, theta_hat.index].values))
    rows = []
    for name, val, sigma in zip(theta_hat.index, theta_hat.values, se):
        rows.append(
            {
                "parameter": name,
                "estimate": float(val),
                "std_dev": float(sigma),
                "ci_lower": float(val - tcrit * sigma),
                "ci_upper": float(val + tcrit * sigma),
            }
        )
    return pd.DataFrame(rows).set_index("parameter")


def solve_fitted_model(
    data: TCLabData,
    theta: pd.Series,
    variant: str,
    parameterization: str,
    delay_order: int,
    alpha: float,
) -> ConcreteModel:
    """Build and solve a fixed-parameter model for fit visualization."""

    if variant == "original":
        model = build_original_tclab_model(
            data,
            theta=theta.to_dict(),
            alpha=alpha,
            fix_parameters=True,
            fix_input=True,
            discretize=True,
            label=False,
        )
    else:
        model = build_reformulated_tclab_model(
            data,
            theta=theta.to_dict(),
            alpha=alpha,
            parameterization=parameterization,
            delay_order=delay_order,
            fix_parameters=True,
            fix_input=True,
            discretize=True,
            label=False,
        )

    ipopt_exec = ensure_ipopt_on_path()
    solver = SolverFactory("ipopt", executable=str(ipopt_exec))
    results = solver.solve(model, tee=False)
    from pyomo.opt.results.solver import assert_optimal_termination

    assert_optimal_termination(results)
    return model


def plot_fit_quality(
    data: TCLabData,
    fitted: pd.DataFrame,
    title: str,
) -> plt.Figure:
    """Plot measured vs fitted temperature and residuals."""

    residuals = np.asarray(data.T) - np.asarray(fitted["Tm"])
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, constrained_layout=True
    )

    ax_top.plot(data.time, data.T, "o", ms=4, label="measured")
    ax_top.plot(fitted["time"], fitted["Tm"], lw=2.5, label="fitted")
    ax_top.set_ylabel("Temperature (°C)")
    ax_top.set_title(title)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    ax_bottom.axhline(0.0, color="black", lw=1)
    ax_bottom.plot(data.time, residuals, lw=1.8, color="tab:red")
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_ylabel("Residual (°C)")
    ax_bottom.grid(True, alpha=0.3)
    return fig


def plot_covariance_heatmap(cov: pd.DataFrame, title: str) -> plt.Figure:
    """Plot a covariance matrix as a labeled heatmap."""

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    values = cov.values
    im = ax.imshow(values, cmap="viridis")
    ax.set_xticks(np.arange(len(cov.columns)))
    ax.set_xticklabels(list(cov.columns), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cov.index)))
    ax.set_yticklabels(list(cov.index))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Covariance")

    vmax = np.nanmax(np.abs(values)) if np.size(values) else 0.0
    text_color = "white" if vmax > 0 else "black"
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2e}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )
    return fig


def plot_multistart_objectives(results_df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot multistart objective values against restart index."""

    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    if results_df.empty:
        ax.text(0.5, 0.5, "No multistart results", ha="center", va="center")
        ax.set_axis_off()
        return fig

    x = np.arange(len(results_df))
    y = pd.to_numeric(results_df["final objective"], errors="coerce").to_numpy()
    term = results_df["solver termination"].astype(str)
    colors = np.where(
        term.str.contains("optimal", case=False, na=False), "tab:green", "tab:red"
    )
    ax.scatter(x, y, c=colors, s=40, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Restart index")
    ax.set_ylabel("Final objective")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


def estimate_variant(
    data: TCLabData,
    variant: str,
    delay_order: int,
    parameterization: str,
    objective: str = "SSE",
    alpha: float = 0.00016,
    measurement_error: float = 0.25,
    tee: bool = False,
    theta0: dict[str, float] | None = None,
    use_multistart: bool = False,
    multistart_sampling_method: str = "uniform_random",
    n_restarts: int = 20,
    seed: int | None = None,
    solver_options: dict[str, Any] | None = None,
    save_multistart_results: bool = False,
    multistart_results_path: Path | None = None,
) -> EstimationResult:
    """Estimate one model variant and assemble the plots and covariance outputs."""

    from pyomo.contrib.parmest import parmest

    if parameterization not in {"physics", "io"}:
        raise ValueError("parameterization must be 'physics' or 'io'")

    if theta0 is None:
        theta0 = default_delayed_theta(parameterization)

    solver_options = {} if solver_options is None else dict(solver_options)

    experiment = TCLabParmestExperiment(
        data=data,
        variant=variant,
        parameterization=parameterization,
        delay_order=delay_order,
        alpha=alpha,
        measurement_error=measurement_error,
        theta=theta0,
        discretize=True,
        fix_input=True,
    )

    pest = parmest.Estimator(
        [experiment], obj_function=objective, tee=tee, solver_options=solver_options
    )

    multistart_results = None
    multistart_best_obj = None
    multistart_fig = None
    if use_multistart:
        if multistart_sampling_method not in {
            "uniform_random",
            "latin_hypercube",
            "sobol_sampling",
        }:
            raise ValueError(
                "multistart_sampling_method must be uniform_random, "
                "latin_hypercube, or sobol_sampling"
            )
        multistart_kwargs: dict[str, Any] = {
            "n_restarts": n_restarts,
            "multistart_sampling_method": multistart_sampling_method,
            "seed": seed,
            "save_results": save_multistart_results,
        }
        if multistart_results_path is not None:
            multistart_kwargs["file_name"] = str(multistart_results_path)
        multistart_results, best_theta_dict, multistart_best_obj = pest.theta_est_multistart(
            **multistart_kwargs
        )
        if best_theta_dict is None or not np.isfinite(multistart_best_obj):
            raise RuntimeError("Multistart did not identify a finite best solution.")
        best_theta_dict = clip_theta_to_bounds(
            best_theta_dict, variant=variant, parameterization=parameterization
        ).to_dict()
        print(
            f"\nMultistart best objective ({multistart_sampling_method}, n_restarts={n_restarts}): "
            f"{multistart_best_obj:.6g}"
        )
        experiment.theta = best_theta_dict
        pest = parmest.Estimator(
            [experiment], obj_function=objective, tee=tee, solver_options=solver_options
        )
        theta0 = best_theta_dict

    obj_val, theta_hat = pest.theta_est()
    theta_hat = clip_theta_to_bounds(
        theta_hat, variant=variant, parameterization=parameterization
    )
    pest.estimated_theta = theta_hat.to_dict()
    cov = pest.cov_est(method="finite_difference", solver="ipopt", step=1e-3)

    theta_hat = pd.Series(theta_hat)
    cov = pd.DataFrame(cov).loc[theta_hat.index, theta_hat.index]

    fitted_model = solve_fitted_model(
        data=data,
        theta=theta_hat,
        variant=variant,
        parameterization=parameterization,
        delay_order=delay_order,
        alpha=alpha,
    )
    fitted = solution_to_dataframe(fitted_model)
    residuals = np.asarray(data.T) - np.asarray(fitted["Tm"])
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    transformed_theta, transformed_cov = transform_theta_and_covariance(
        theta_hat=theta_hat,
        covariance=cov,
        parameterization=parameterization,
        alpha=alpha,
        P=data.P,
    )

    fit_title = (
        f"TCLab fit: {parameterization} / delay_order={delay_order} / obj={objective}"
    )
    cov_title = (
        f"Covariance: {parameterization} / delay_order={delay_order} / obj={objective}"
    )
    fit_fig = plot_fit_quality(data, fitted, fit_title)
    cov_fig = plot_covariance_heatmap(cov, cov_title)
    if multistart_results is not None:
        multistart_title = (
            f"Multistart: {parameterization} / delay_order={delay_order} / "
            f"{multistart_sampling_method} / obj={objective}"
        )
        multistart_fig = plot_multistart_objectives(multistart_results, multistart_title)

    print("\n" + "=" * 80)
    print(f"Variant: {variant} | parameterization: {parameterization}")
    print(f"Objective value: {obj_val:.6g}")
    if multistart_best_obj is not None:
        print(f"Multistart best objective: {multistart_best_obj:.6g}")
    if theta0 is not None:
        print("Initial guess:")
        print(pd.Series(theta0).to_string())
    print("Estimated parameters:")
    print(theta_hat.to_string())
    print("\nParmest covariance matrix:")
    print(cov.to_string())
    print("\nTransformed parameters:")
    print(transformed_theta.to_string())
    print("\nTransformed covariance matrix:")
    print(transformed_cov.to_string())
    print(f"\nFit quality: RMSE={rmse:.4f} °C, MAE={mae:.4f} °C")
    print("Approximate 95% CIs for estimated parameters:")
    print(confidence_interval_from_covariance(theta_hat, cov, len(data.time)).to_string())

    return EstimationResult(
        variant=variant,
        delay_order=delay_order,
        parameterization=parameterization,
        objective=objective,
        theta_hat=theta_hat,
        covariance=cov,
        transformed_theta=transformed_theta,
        transformed_covariance=transformed_cov,
        fitted_data=fitted,
        fit_figure=fit_fig,
        covariance_figure=cov_fig,
        multistart_figure=multistart_fig,
        rmse=rmse,
        mae=mae,
        multistart_results=multistart_results,
        multistart_best_objective=multistart_best_obj,
        multistart_sampling_method=multistart_sampling_method if use_multistart else None,
        multistart_n_restarts=n_restarts if use_multistart else None,
        multistart_seed=seed if use_multistart else None,
    )


def build_summary_tables(
    results: Sequence[EstimationResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build compact run summaries and long-form covariance summaries."""

    run_rows: list[dict[str, Any]] = []
    cov_rows: list[dict[str, Any]] = []

    for result in results:
        run_row: dict[str, Any] = {
            "variant": result.variant,
            "delay_order": result.delay_order,
            "parameterization": result.parameterization,
            "objective": result.objective,
            "rmse": result.rmse,
            "mae": result.mae,
        }

        if result.multistart_sampling_method is not None:
            run_row["multistart_sampling_method"] = result.multistart_sampling_method
        if result.multistart_n_restarts is not None:
            run_row["multistart_n_restarts"] = result.multistart_n_restarts
        if result.multistart_seed is not None:
            run_row["multistart_seed"] = result.multistart_seed
        if result.multistart_best_objective is not None:
            run_row["multistart_best_objective"] = result.multistart_best_objective

        for name in ["Ua", "Cp", "K", "tau", "theta"]:
            run_row[f"fit_{name}"] = float(result.theta_hat[name]) if name in result.theta_hat.index else np.nan
            run_row[f"alt_{name}"] = (
                float(result.transformed_theta[name])
                if name in result.transformed_theta.index
                else np.nan
            )

        run_rows.append(run_row)

        orig_ci = confidence_interval_from_covariance(
            result.theta_hat, result.covariance, len(result.fitted_data)
        )
        alt_ci = confidence_interval_from_covariance(
            result.transformed_theta, result.transformed_covariance, len(result.fitted_data)
        )

        for space, theta, cov, ci in [
            ("fit", result.theta_hat, result.covariance, orig_ci),
            ("transformed", result.transformed_theta, result.transformed_covariance, alt_ci),
        ]:
            for pname in theta.index:
                cov_rows.append(
                    {
                        "variant": result.variant,
                        "delay_order": result.delay_order,
                        "parameterization": result.parameterization,
                        "objective": result.objective,
                        "space": space,
                        "parameter": pname,
                        "estimate": float(theta[pname]),
                        "variance": float(cov.loc[pname, pname]),
                        "std_dev": float(ci.loc[pname, "std_dev"]),
                        "ci_lower": float(ci.loc[pname, "ci_lower"]),
                        "ci_upper": float(ci.loc[pname, "ci_upper"]),
                    }
                )

    return pd.DataFrame(run_rows), pd.DataFrame(cov_rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the TCLab estimation comparison."""

    parser = argparse.ArgumentParser(
        description=(
            "Estimate the delayed TCLab model in both parameterizations using the "
            "sine-wave dataset from parmest.ipynb."
        )
    )
    parser.add_argument(
        "--objective",
        choices=["SSE", "SSE_weighted"],
        default="SSE",
        help="Parmest objective function to use.",
    )
    parser.add_argument(
        "--delay-order",
        type=int,
        default=2,
        help="Order of the lag chain used to approximate the delay.",
    )
    parser.add_argument(
        "--delay-orders",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Explicit list of lag-chain orders to compare. "
            "If omitted, uses --delay-order unless --compare-delay-orders is set."
        ),
    )
    parser.add_argument(
        "--compare-delay-orders",
        action="store_true",
        help="Compare delay_order=2 and delay_order=3 in one run.",
    )
    parser.add_argument(
        "--measurement-error",
        type=float,
        default=0.25,
        help="Measurement error used for SSE_weighted.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.00016,
        help="Heater gain constant alpha.",
    )
    parser.add_argument(
        "--multistart",
        action="store_true",
        help="Use Parmest multistart instead of a single-start estimate.",
    )
    parser.add_argument(
        "--multistart-method",
        choices=["uniform_random", "latin_hypercube", "sobol_sampling"],
        default="uniform_random",
        help="Sampling method used when --multistart is enabled.",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=15,
        help="Number of multistart restarts to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=532,
        help="Random seed used for multistart sampling.",
    )
    parser.add_argument(
        "--linear-solver",
        default="ma57",
        help="Ipopt linear solver option passed through Parmest.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Ipopt max_iter option passed through Parmest.",
    )
    parser.add_argument(
        "--save-multistart-results",
        action="store_true",
        help="Save the full multistart restart table to CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().with_name("tclab_reformulated_results"),
        help="Directory where figures should be saved.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional text file to receive a copy of console output.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    parser.add_argument(
        "--tee",
        action="store_true",
        help="Show solver output during estimation.",
    )
    args = parser.parse_args(argv)

    log_file = args.log_file
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    def run() -> int:
        ensure_ipopt_on_path()
        data = load_sine_wave_dataset()
        solver_options: dict[str, Any] = {"max_iter": args.max_iter}
        if args.linear_solver:
            solver_options["linear_solver"] = args.linear_solver
        args.output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        if args.delay_orders is not None and len(args.delay_orders) > 0:
            delay_orders = list(dict.fromkeys(args.delay_orders))
        elif args.compare_delay_orders:
            delay_orders = [2, 3]
        else:
            delay_orders = [args.delay_order]

        for delay_order in delay_orders:
            physics_result = estimate_variant(
                data=data,
                variant="reformulated",
                delay_order=delay_order,
                parameterization="physics",
                objective=args.objective,
                alpha=args.alpha,
                measurement_error=args.measurement_error,
                tee=args.tee,
                use_multistart=args.multistart,
                multistart_sampling_method=args.multistart_method,
                n_restarts=args.n_restarts,
                seed=args.seed,
                solver_options=solver_options,
                save_multistart_results=args.save_multistart_results,
                multistart_results_path=(
                    args.output_dir
                    / f"physics_delay{delay_order}_{args.objective}_{args.multistart_method}_multistart.csv"
                    if args.multistart
                    else None
                ),
            )
            results.append(physics_result)

            if not args.multistart:
                print(
                    "\nWarm-starting the io parameterization from the transformed physics estimate."
                )
            io_result = estimate_variant(
                data=data,
                variant="reformulated",
                delay_order=delay_order,
                parameterization="io",
                objective=args.objective,
                alpha=args.alpha,
                measurement_error=args.measurement_error,
                tee=args.tee,
                theta0=physics_result.transformed_theta.to_dict(),
                use_multistart=args.multistart,
                multistart_sampling_method=args.multistart_method,
                n_restarts=args.n_restarts,
                seed=args.seed,
                solver_options=solver_options,
                save_multistart_results=args.save_multistart_results,
                multistart_results_path=(
                    args.output_dir
                    / f"io_delay{delay_order}_{args.objective}_{args.multistart_method}_multistart.csv"
                    if args.multistart
                    else None
                ),
            )
            results.append(io_result)

        for result in results:
            result.fit_figure.canvas.draw_idle()
            result.covariance_figure.canvas.draw_idle()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            fit_name = (
                f"{result.parameterization}_delay{result.delay_order}_{args.objective}_fit.png"
            )
            cov_name = (
                f"{result.parameterization}_delay{result.delay_order}_{args.objective}_cov.png"
            )
            fit_path = args.output_dir / fit_name
            cov_path = args.output_dir / cov_name
            result.fit_figure.savefig(fit_path, dpi=200, bbox_inches="tight")
            result.covariance_figure.savefig(cov_path, dpi=200, bbox_inches="tight")
            print(f"Saved fit figure to {fit_path}")
            print(f"Saved covariance figure to {cov_path}")
            if result.multistart_figure is not None:
                multistart_name = (
                    f"{result.parameterization}_delay{result.delay_order}_{args.objective}_{args.multistart_method}_multistart.png"
                )
                multistart_path = args.output_dir / multistart_name
                result.multistart_figure.savefig(
                    multistart_path, dpi=200, bbox_inches="tight"
                )
                print(f"Saved multistart figure to {multistart_path}")

        if args.show:
            plt.show()
        else:
            plt.close("all")

        summary, covariance_summary = build_summary_tables(results)
        summary_csv = args.output_dir / "tclab_reformulated_fit_summary.csv"
        covariance_csv = args.output_dir / "tclab_reformulated_covariance_summary.csv"
        summary.to_csv(summary_csv, index=False)
        covariance_summary.to_csv(covariance_csv, index=False)
        print(f"Saved fit summary to {summary_csv}")
        print(f"Saved covariance summary to {covariance_csv}")

        print("\nSummary:")
        print(summary.to_string(index=False))
        return 0

    if log_file is None:
        return run()

    with log_file.open("w", encoding="utf-8") as fh, contextlib.redirect_stdout(
        TeeStream(sys.stdout, fh)
    ), contextlib.redirect_stderr(TeeStream(sys.stderr, fh)):
        return run()


if __name__ == "__main__":
    raise SystemExit(main())
