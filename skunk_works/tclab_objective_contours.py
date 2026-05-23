"""Objective contour study for the delayed TCLab model.

This script computes least-squares objective contours over gain and time
constant for fixed delay values. It is intentionally modular so that the grid
search can later be swapped for an optimizer without changing the plotting
layer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import os
import sys
from typing import Callable, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ["MPLCONFIGDIR"] = str(HERE / ".mplconfig")
os.environ["XDG_CACHE_HOME"] = str(HERE / ".cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator, interp1d


@dataclass
class TCLabData:
    """Container for a single TCLab experiment."""

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


@dataclass(frozen=True)
class DatasetConfig:
    """Metadata for a contour-analysis dataset."""

    key: str
    label: str
    loader: Callable[[], TCLabData]


def load_sine_wave_dataset(
    csv_path: str | Path | None = None,
) -> TCLabData:
    """Load the sine-wave TCLab dataset used in ``parmest.ipynb``."""

    if csv_path is None:
        csv_path = ROOT / "data" / "tclab_sine_test_5min_period.csv"
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


def load_step_test_dataset(
    csv_path: str | Path | None = None,
) -> TCLabData:
    """Load the TCLab step-test dataset."""

    if csv_path is None:
        csv_path = ROOT / "data" / "tclab_step_test.csv"
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    return TCLabData(
        name="Step Test for Heater 1",
        time=df["Time"].values,
        T=df["T1"].values,
        u=df["Q1"].values,
        Tamb=float(df["T1"].values[0]),
        P=200.0,
    )


def available_datasets() -> dict[str, DatasetConfig]:
    """Return the datasets supported by the contour study."""

    return {
        "sine": DatasetConfig(
            key="sine",
            label="Sine Wave Test for Heater 1",
            loader=load_sine_wave_dataset,
        ),
        "step": DatasetConfig(
            key="step",
            label="Step Test for Heater 1",
            loader=load_step_test_dataset,
        ),
    }


@dataclass(frozen=True)
class ContourConfig:
    """Configuration for a contour sweep."""

    dataset_key: str
    dataset_label: str
    delay_order: int
    theta_values: tuple[float, ...]
    k_min: float
    k_max: float
    tau_min: float
    tau_max: float
    n_k: int
    n_tau: int


@dataclass(frozen=True)
class MasterGridConfig:
    """Configuration for the 3D sensitivity grid."""

    dataset_key: str
    dataset_label: str
    delay_order: int
    k_values: np.ndarray
    tau_values: np.ndarray
    theta_values: np.ndarray


@dataclass
class MasterGridResult:
    """Result of the 3D sensitivity sweep."""

    config: MasterGridConfig
    sse_tensor: np.ndarray

    def interpolator(self) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            (self.config.k_values, self.config.tau_values, self.config.theta_values),
            self.sse_tensor,
            bounds_error=False,
            fill_value=None,
        )


@dataclass
class ContourSliceResult:
    """Result for one fixed-theta contour slice."""

    dataset_key: str
    delay_order: int
    theta: float
    k_values: np.ndarray
    tau_values: np.ndarray
    sse_grid: np.ndarray
    best_k: float
    best_tau: float
    best_sse: float
    best_prediction: pd.DataFrame


def simulate_delayed_fopdt(
    data: TCLabData,
    delay_order: int,
    K: float,
    tau: float,
    theta: float,
) -> pd.DataFrame:
    """Simulate the delayed first-order TCLab surrogate with RK4."""

    if delay_order < 1:
        raise ValueError("delay_order must be at least 1")
    if theta <= 0:
        raise ValueError("theta must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if K <= 0:
        raise ValueError("K must be positive")

    t = np.asarray(data.time, dtype=float)
    u = np.asarray(data.u, dtype=float)
    if len(t) != len(u):
        raise ValueError("time and u must have the same length")

    u_func = interp1d(
        t,
        u,
        kind="linear",
        bounds_error=False,
        fill_value=(float(u[0]), float(u[-1])),
        assume_sorted=True,
    )

    state_dim = delay_order + 1
    states = np.empty((len(t), state_dim), dtype=float)
    states[0, :delay_order] = float(u[0])
    states[0, -1] = float(data.T[0])

    def rhs(tt: float, y: np.ndarray) -> np.ndarray:
        z = y[:-1]
        temp = float(y[-1])
        u_now = float(u_func(tt))

        dz = np.empty(delay_order, dtype=float)
        dz[0] = (delay_order / theta) * (u_now - z[0])
        for i in range(1, delay_order):
            dz[i] = (delay_order / theta) * (z[i - 1] - z[i])

        dtemp = -(temp - float(data.Tamb)) / tau + (K / tau) * z[-1]
        return np.concatenate([dz, np.array([dtemp], dtype=float)])

    for i in range(len(t) - 1):
        dt = float(t[i + 1] - t[i])
        y = states[i]
        t0 = float(t[i])

        k1 = rhs(t0, y)
        k2 = rhs(t0 + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs(t0 + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs(t0 + dt, y + dt * k3)
        states[i + 1] = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    rows = {"time": t, "Tm": states[:, -1]}
    for i in range(delay_order):
        rows[f"z{i+1}"] = states[:, i]
    return pd.DataFrame(rows)


def objective_sse(data: TCLabData, prediction: pd.DataFrame) -> float:
    """Compute an unweighted sum of squared errors."""

    residuals = np.asarray(data.T, dtype=float) - np.asarray(prediction["Tm"], dtype=float)
    return float(np.sum(residuals**2))


def evaluate_master_grid(
    data: TCLabData,
    config: MasterGridConfig,
) -> MasterGridResult:
    """Evaluate the full K-tau-theta grid once."""

    sse_tensor = np.empty(
        (
            len(config.k_values),
            len(config.tau_values),
            len(config.theta_values),
        ),
        dtype=float,
    )

    for i, K in enumerate(config.k_values):
        for j, tau in enumerate(config.tau_values):
            for k, theta in enumerate(config.theta_values):
                prediction = simulate_delayed_fopdt(
                    data=data,
                    delay_order=config.delay_order,
                    K=float(K),
                    tau=float(tau),
                    theta=float(theta),
                )
                sse_tensor[i, j, k] = objective_sse(data, prediction)
        print(f"  completed K slice {i + 1}/{len(config.k_values)} for delay_order={config.delay_order}")

    return MasterGridResult(config=config, sse_tensor=sse_tensor)


def master_grid_to_dataframe(grid: MasterGridResult) -> pd.DataFrame:
    """Flatten the master grid to a tabular CSV-friendly format."""

    k_grid, tau_grid, theta_grid = np.meshgrid(
        grid.config.k_values,
        grid.config.tau_values,
        grid.config.theta_values,
        indexing="ij",
    )
    return pd.DataFrame(
        {
            "dataset_key": np.repeat(
                grid.config.dataset_key, len(grid.sse_tensor.ravel())
            ),
            "dataset_label": np.repeat(
                grid.config.dataset_label, len(grid.sse_tensor.ravel())
            ),
            "delay_order": np.repeat(grid.config.delay_order, len(grid.sse_tensor.ravel())),
            "K": k_grid.ravel(),
            "tau": tau_grid.ravel(),
            "theta": theta_grid.ravel(),
            "sse": grid.sse_tensor.ravel(),
        }
    )


@dataclass
class SliceContourResult:
    """Result for one fixed-parameter contour slice."""

    plot_mode: str
    dataset_key: str
    dataset_label: str
    delay_order: int
    fixed_parameter_name: str
    fixed_parameter_value: float
    x_values: np.ndarray
    y_values: np.ndarray
    sse_grid: np.ndarray
    best_k: float
    best_tau: float
    best_theta: float
    best_sse: float
    best_prediction: pd.DataFrame


def _axis_labels_for_mode(plot_mode: str) -> tuple[str, str, str]:
    """Return x, y, and fixed-parameter labels for a contour mode."""

    if plot_mode == "ktau":
        return "K", r"$\tau$ (s)", r"$\theta$ (s)"
    if plot_mode == "thetatau":
        return r"$\theta$ (s)", r"$\tau$ (s)", "K"
    raise ValueError("plot_mode must be 'ktau' or 'thetatau'")


def evaluate_slice(
    grid: MasterGridResult,
    data: TCLabData,
    plot_mode: str,
    fixed_parameter_value: float,
) -> SliceContourResult:
    """Evaluate one contour slice from the master grid via interpolation."""

    interpolator = grid.interpolator()
    if plot_mode == "ktau":
        x_values = grid.config.k_values
        y_values = grid.config.tau_values
        x_mesh, y_mesh = np.meshgrid(x_values, y_values, indexing="xy")
        pts = np.column_stack(
            [
                x_mesh.ravel(),
                y_mesh.ravel(),
                np.full(x_mesh.size, float(fixed_parameter_value), dtype=float),
            ]
        )
        sse_grid = interpolator(pts).reshape(x_mesh.shape)
        best_index = int(np.nanargmin(sse_grid))
        best_y_idx, best_x_idx = np.unravel_index(best_index, sse_grid.shape)
        best_k = float(x_mesh[best_y_idx, best_x_idx])
        best_tau = float(y_mesh[best_y_idx, best_x_idx])
        best_theta = float(fixed_parameter_value)
        fixed_parameter_name = "theta"
    elif plot_mode == "thetatau":
        x_values = grid.config.theta_values
        y_values = grid.config.tau_values
        x_mesh, y_mesh = np.meshgrid(x_values, y_values, indexing="xy")
        pts = np.column_stack(
            [
                np.full(x_mesh.size, float(fixed_parameter_value), dtype=float),
                y_mesh.ravel(),
                x_mesh.ravel(),
            ]
        )
        sse_grid = interpolator(pts).reshape(x_mesh.shape)
        best_index = int(np.nanargmin(sse_grid))
        best_y_idx, best_x_idx = np.unravel_index(best_index, sse_grid.shape)
        best_k = float(fixed_parameter_value)
        best_tau = float(y_mesh[best_y_idx, best_x_idx])
        best_theta = float(x_mesh[best_y_idx, best_x_idx])
        fixed_parameter_name = "K"
    else:
        raise ValueError("plot_mode must be 'ktau' or 'thetatau'")

    best_prediction = simulate_delayed_fopdt(
        data=data,
        delay_order=grid.config.delay_order,
        K=best_k,
        tau=best_tau,
        theta=best_theta,
    )
    best_sse = float(sse_grid[best_y_idx, best_x_idx])

    return SliceContourResult(
        plot_mode=plot_mode,
        dataset_key=grid.config.dataset_key,
        dataset_label=grid.config.dataset_label,
        delay_order=grid.config.delay_order,
        fixed_parameter_name=fixed_parameter_name,
        fixed_parameter_value=float(fixed_parameter_value),
        x_values=x_values,
        y_values=y_values,
        sse_grid=sse_grid,
        best_k=best_k,
        best_tau=best_tau,
        best_theta=best_theta,
        best_sse=best_sse,
        best_prediction=best_prediction,
    )


def plot_contour_mode(
    grid: MasterGridResult,
    data: TCLabData,
    plot_mode: str,
    fixed_values: Sequence[float],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build and optionally save one contour figure for a plotting mode."""

    results = [evaluate_slice(grid, data, plot_mode, float(value)) for value in fixed_values]
    all_values = np.concatenate([result.sse_grid.ravel() for result in results])
    vmin = float(max(np.nanmin(all_values), np.finfo(float).tiny))
    vmax = float(np.nanmax(all_values))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin * 10.0
    norm = LogNorm(vmin=vmin, vmax=vmax)
    levels = np.geomspace(vmin, vmax, 18)

    x_label, y_label, fixed_label = _axis_labels_for_mode(plot_mode)
    fig, axes = plt.subplots(
        nrows=len(results),
        ncols=2,
        figsize=(14.0, max(3.0 * len(results), 4.5)),
        constrained_layout=True,
        sharex="col",
    )
    if len(results) == 1:
        axes = np.array([axes])

    contour_artist = None
    summary_rows = []
    for row, result in enumerate(results):
        ax_contour = axes[row, 0]
        ax_fit = axes[row, 1]
        x_mesh, y_mesh = np.meshgrid(result.x_values, result.y_values, indexing="xy")
        contour_artist = ax_contour.contourf(
            x_mesh,
            y_mesh,
            result.sse_grid,
            levels=levels,
            cmap="viridis",
            norm=norm,
        )
        ax_contour.contour(
            x_mesh,
            y_mesh,
            result.sse_grid,
            levels=levels[::3],
            colors="black",
            linewidths=0.5,
            alpha=0.35,
        )
        if plot_mode == "ktau":
            best_x, best_y = result.best_k, result.best_tau
        else:
            best_x, best_y = result.best_theta, result.best_tau
        ax_contour.scatter(
            best_x,
            best_y,
            marker="*",
            s=180,
            c="red",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax_contour.set_title(f"{fixed_label} = {result.fixed_parameter_value:.3g}")
        ax_contour.set_xlabel(x_label)
        ax_contour.set_ylabel(y_label)
        ax_contour.text(
            0.03,
            0.97,
            (
                f"best SSE = {result.best_sse:.3g}\n"
                f"K = {result.best_k:.3g}\n"
                f"tau = {result.best_tau:.3g} s\n"
                f"theta = {result.best_theta:.3g} s"
            ),
            transform=ax_contour.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
        )
        ax_contour.grid(True, alpha=0.15)

        ax_fit.plot(data.time, data.T, "o", ms=3.5, color="black", label="measured")
        ax_fit.plot(
            result.best_prediction["time"],
            result.best_prediction["Tm"],
            lw=2.2,
            color="tab:red",
            label="best grid fit",
        )
        ax_fit.set_title(
            f"Best fit | K = {result.best_k:.3g}, tau = {result.best_tau:.3g} s, "
            f"theta = {result.best_theta:.3g} s"
        )
        ax_fit.set_xlabel("Time (s)")
        ax_fit.set_ylabel("Temperature (°C)")
        ax_fit.grid(True, alpha=0.3)
        ax_fit.legend(loc="best")

        summary_rows.append(
            {
                "dataset_key": result.dataset_key,
                "dataset_label": result.dataset_label,
                "delay_order": result.delay_order,
                "plot_mode": result.plot_mode,
                "fixed_parameter_name": result.fixed_parameter_name,
                "fixed_parameter_value": result.fixed_parameter_value,
                "best_K": result.best_k,
                "best_tau": result.best_tau,
                "best_theta": result.best_theta,
                "best_sse": result.best_sse,
            }
        )

    fig.colorbar(contour_artist, ax=axes[:, 0], shrink=0.92, label="SSE")
    mode_title = "K-tau slices" if plot_mode == "ktau" else "theta-tau slices"
    fig.suptitle(
        f"TCLab objective contours | {grid.config.dataset_label} | "
        f"delay_order={grid.config.delay_order} | {mode_title}",
        fontsize=14,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved contour figure to {output_path}")

    return pd.DataFrame(summary_rows)


def grid_search_theta_slice(
    data: TCLabData,
    dataset_key: str,
    delay_order: int,
    theta: float,
    k_values: np.ndarray,
    tau_values: np.ndarray,
) -> ContourSliceResult:
    """Grid-search the objective for a single fixed theta value."""

    sse_grid = np.empty((len(tau_values), len(k_values)), dtype=float)
    best_k = float("nan")
    best_tau = float("nan")
    best_sse = float("inf")
    best_prediction = None

    for i, tau in enumerate(tau_values):
        for j, K in enumerate(k_values):
            prediction = simulate_delayed_fopdt(
                data=data,
                delay_order=delay_order,
                K=float(K),
                tau=float(tau),
                theta=float(theta),
            )
            sse = objective_sse(data, prediction)
            sse_grid[i, j] = sse
            if sse < best_sse:
                best_sse = sse
                best_k = float(K)
                best_tau = float(tau)
                best_prediction = prediction

    assert best_prediction is not None
    return ContourSliceResult(
        dataset_key=dataset_key,
        delay_order=delay_order,
        theta=float(theta),
        k_values=k_values,
        tau_values=tau_values,
        sse_grid=sse_grid,
        best_k=best_k,
        best_tau=best_tau,
        best_sse=best_sse,
        best_prediction=best_prediction,
    )


def optimize_theta_slice(*args, **kwargs):
    """Future hook for an optimization-based contour search."""

    raise NotImplementedError("Optimization-based contour search is not implemented yet.")


def evaluate_theta_slice(
    data: TCLabData,
    dataset_key: str,
    delay_order: int,
    theta: float,
    k_values: np.ndarray,
    tau_values: np.ndarray,
    search_method: str = "grid",
) -> ContourSliceResult:
    """Evaluate one theta slice using the selected search method."""

    if search_method == "grid":
        return grid_search_theta_slice(
            data, dataset_key, delay_order, theta, k_values, tau_values
        )
    if search_method == "optimize":
        return optimize_theta_slice(data, delay_order, theta, k_values, tau_values)
    raise ValueError("search_method must be 'grid' or 'optimize'")


def build_delay_order_figure(
    data: TCLabData,
    results: list[ContourSliceResult],
    config: ContourConfig,
    output_path: Path | None = None,
) -> plt.Figure:
    """Create the two-column contour/prediction figure for one delay order."""

    nrows = len(results)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(14.0, max(3.0 * nrows, 4.5)),
        constrained_layout=True,
        sharex="col",
    )
    if nrows == 1:
        axes = np.array([axes])

    all_values = np.concatenate([r.sse_grid.ravel() for r in results])
    vmin = float(max(np.nanmin(all_values), np.finfo(float).tiny))
    vmax = float(np.nanmax(all_values))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin * 10.0
    norm = LogNorm(vmin=vmin, vmax=vmax)
    levels = np.geomspace(vmin, vmax, 18)

    K_mesh, tau_mesh = np.meshgrid(results[0].k_values, results[0].tau_values)

    contour_artist = None
    for row, result in enumerate(results):
        ax_contour = axes[row, 0]
        ax_fit = axes[row, 1]

        contour_artist = ax_contour.contourf(
            K_mesh,
            tau_mesh,
            result.sse_grid,
            levels=levels,
            cmap="viridis",
            norm=norm,
        )
        ax_contour.contour(
            K_mesh,
            tau_mesh,
            result.sse_grid,
            levels=levels[::3],
            colors="black",
            linewidths=0.5,
            alpha=0.35,
        )
        ax_contour.scatter(
            result.best_k,
            result.best_tau,
            marker="*",
            s=180,
            c="red",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax_contour.set_title(f"$\\theta$ = {result.theta:.0f} s")
        ax_contour.set_xlabel("K")
        ax_contour.set_ylabel(r"$\tau$ (s)")
        ax_contour.text(
            0.03,
            0.97,
            f"best SSE = {result.best_sse:.3g}\nK = {result.best_k:.3g}\n$\\tau$ = {result.best_tau:.3g} s",
            transform=ax_contour.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
        )
        ax_contour.grid(True, alpha=0.15)

        ax_fit.plot(data.time, data.T, "o", ms=3.5, color="black", label="measured")
        ax_fit.plot(
            result.best_prediction["time"],
            result.best_prediction["Tm"],
            lw=2.2,
            color="tab:red",
            label="best grid fit",
        )
        ax_fit.set_title(
            f"Best fit at $\\theta$ = {result.theta:.0f} s\n"
            f"K = {result.best_k:.3g}, $\\tau$ = {result.best_tau:.3g} s"
        )
        ax_fit.set_xlabel("Time (s)")
        ax_fit.set_ylabel("Temperature (°C)")
        ax_fit.grid(True, alpha=0.3)
        ax_fit.legend(loc="best")

    fig.colorbar(contour_artist, ax=axes[:, 0], shrink=0.92, label="SSE")
    fig.suptitle(
        (
            f"TCLab objective contours | {config.dataset_label} "
            f"| delay_order={config.delay_order} | grid search"
        ),
        fontsize=14,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved contour figure to {output_path}")

    return fig


def summarize_results(results: list[ContourSliceResult], delay_order: int) -> pd.DataFrame:
    """Build a compact table of contour minima."""

    rows = []
    for result in results:
        rows.append(
            {
                "dataset_key": result.dataset_key,
                "delay_order": delay_order,
                "theta": result.theta,
                "best_K": result.best_k,
                "best_tau": result.best_tau,
                "best_sse": result.best_sse,
            }
        )
    return pd.DataFrame(rows)


def run_contour_study(
    data: TCLabData,
    config: ContourConfig,
    search_method: str = "grid",
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the contour study for one delay order."""

    k_values = np.linspace(config.k_min, config.k_max, config.n_k)
    tau_values = np.linspace(config.tau_min, config.tau_max, config.n_tau)

    results: list[ContourSliceResult] = []
    for theta in config.theta_values:
        print(f"Evaluating delay_order={config.delay_order}, theta={theta:.0f} s")
        result = evaluate_theta_slice(
            data=data,
            dataset_key=config.dataset_key,
            delay_order=config.delay_order,
            theta=float(theta),
            k_values=k_values,
            tau_values=tau_values,
            search_method=search_method,
        )
        results.append(result)
        print(
            f"  best SSE={result.best_sse:.6g} at K={result.best_k:.6g}, tau={result.best_tau:.6g}"
        )

    if output_dir is not None:
        fig_path = output_dir / (
            f"tclab_objective_contours_{config.dataset_key}_delay{config.delay_order}.png"
        )
    else:
        fig_path = None
    build_delay_order_figure(data, results, config, output_path=fig_path)

    summary = summarize_results(results, config.delay_order)
    if output_dir is not None:
        summary_path = output_dir / (
            f"tclab_objective_contours_{config.dataset_key}_delay{config.delay_order}.csv"
        )
        summary.to_csv(summary_path, index=False)
        print(f"Saved contour summary to {summary_path}")
    return summary


def parse_theta_values(values: Sequence[str] | None) -> tuple[float, ...]:
    if values is None or len(values) == 0:
        return (10.0, 15.0, 20.0, 25.0, 30.0)
    return tuple(float(v) for v in values)


def parse_k_values(values: Sequence[str] | None) -> tuple[float, ...]:
    if values is None or len(values) == 0:
        return (0.75, 0.80, 0.85)
    return tuple(float(v) for v in values)


def parse_datasets(values: Sequence[str] | None) -> tuple[DatasetConfig, ...]:
    """Parse dataset keys into dataset configurations."""

    datasets = available_datasets()
    if values is None or len(values) == 0:
        values = ("sine", "step")

    resolved = []
    for value in values:
        key = value.lower()
        if key not in datasets:
            valid = ", ".join(sorted(datasets))
            raise ValueError(f"Unknown dataset '{value}'. Valid options are: {valid}")
        resolved.append(datasets[key])
    return tuple(resolved)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description="Generate TCLab objective contours for fixed delay values."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["sine", "step"],
        help="Datasets to analyze (sine, step).",
    )
    parser.add_argument(
        "--delay-orders",
        type=int,
        nargs="*",
        default=[2, 3],
        help="Delay-chain orders to compare.",
    )
    parser.add_argument(
        "--thetas",
        type=float,
        nargs="*",
        default=[10.0, 15.0, 20.0, 25.0, 30.0],
        help="Fixed theta values for the K-tau contour slices.",
    )
    parser.add_argument(
        "--ks",
        type=float,
        nargs="*",
        default=[0.75, 0.80, 0.85],
        help="Fixed K values for the theta-tau contour slices.",
    )
    parser.add_argument(
        "--k-min",
        type=float,
        default=0.5,
        help="Lower bound for K.",
    )
    parser.add_argument(
        "--k-max",
        type=float,
        default=0.9,
        help="Upper bound for the master-grid K axis.",
    )
    parser.add_argument(
        "--tau-min",
        type=float,
        default=120.0,
        help="Lower bound for tau.",
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=200.0,
        help="Upper bound for tau.",
    )
    parser.add_argument(
        "--theta-min",
        type=float,
        default=10.0,
        help="Lower bound for the master-grid theta axis.",
    )
    parser.add_argument(
        "--theta-max",
        type=float,
        default=30.0,
        help="Upper bound for the master-grid theta axis.",
    )
    parser.add_argument(
        "--n-k",
        type=int,
        default=25,
        help="Number of grid points for K in the master grid.",
    )
    parser.add_argument(
        "--n-tau",
        type=int,
        default=25,
        help="Number of grid points for tau in the master grid.",
    )
    parser.add_argument(
        "--n-theta",
        type=int,
        default=25,
        help="Number of grid points for theta in the master grid.",
    )
    parser.add_argument(
        "--plot-modes",
        nargs="*",
        choices=["ktau", "thetatau"],
        default=["ktau", "thetatau"],
        help="Contour styles to render from the master grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().with_name("tclab_objective_contours_results"),
        help="Directory where figures and summary tables will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    grid_tables = []
    datasets = parse_datasets(args.datasets)
    theta_slice_values = parse_theta_values(args.thetas)
    k_slice_values = parse_k_values(args.ks)

    for dataset_config in datasets:
        data = dataset_config.loader()
        print(f"Loaded dataset: {dataset_config.key} ({dataset_config.label})")
        for delay_order in args.delay_orders:
            master_config = MasterGridConfig(
                dataset_key=dataset_config.key,
                dataset_label=dataset_config.label,
                delay_order=delay_order,
                k_values=np.linspace(args.k_min, args.k_max, args.n_k),
                tau_values=np.linspace(args.tau_min, args.tau_max, args.n_tau),
                theta_values=np.linspace(args.theta_min, args.theta_max, args.n_theta),
            )
            print(
                f"Evaluating master grid for {dataset_config.key}, delay_order={delay_order}"
            )
            master_grid = evaluate_master_grid(data=data, config=master_config)

            grid_csv = output_dir / (
                f"tclab_objective_contours_{dataset_config.key}_delay{delay_order}_grid.csv"
            )
            master_grid_to_dataframe(master_grid).to_csv(grid_csv, index=False)
            print(f"Saved master grid to {grid_csv}")

            grid_tables.append(master_grid_to_dataframe(master_grid))

            if "ktau" in args.plot_modes:
                ktau_path = output_dir / (
                    f"tclab_objective_contours_{dataset_config.key}_delay{delay_order}_ktau.png"
                )
                ktau_summary = plot_contour_mode(
                    grid=master_grid,
                    data=data,
                    plot_mode="ktau",
                    fixed_values=theta_slice_values,
                    output_path=ktau_path,
                )
                ktau_csv = output_dir / (
                    f"tclab_objective_contours_{dataset_config.key}_delay{delay_order}_ktau.csv"
                )
                ktau_summary.to_csv(ktau_csv, index=False)
                print(f"Saved contour summary to {ktau_csv}")
                summaries.append(ktau_summary)

            if "thetatau" in args.plot_modes:
                thetatau_path = output_dir / (
                    f"tclab_objective_contours_{dataset_config.key}_delay{delay_order}_thetatau.png"
                )
                thetatau_summary = plot_contour_mode(
                    grid=master_grid,
                    data=data,
                    plot_mode="thetatau",
                    fixed_values=k_slice_values,
                    output_path=thetatau_path,
                )
                thetatau_csv = output_dir / (
                    f"tclab_objective_contours_{dataset_config.key}_delay{delay_order}_thetatau.csv"
                )
                thetatau_summary.to_csv(thetatau_csv, index=False)
                print(f"Saved contour summary to {thetatau_csv}")
                summaries.append(thetatau_summary)

    if grid_tables:
        grid_df = pd.concat(grid_tables, ignore_index=True)
        grid_path = output_dir / "tclab_objective_contours_grid.csv"
        grid_df.to_csv(grid_path, index=False)
        print(f"Saved combined grid table to {grid_path}")

    if summaries:
        summary_df = pd.concat(summaries, ignore_index=True)
        summary_path = output_dir / "tclab_objective_contours_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved combined contour summary to {summary_path}")
        print(summary_df.to_string(index=False))

    if args.show:
        plt.show()
    else:
        plt.close("all")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
