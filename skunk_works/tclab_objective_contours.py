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
from scipy.interpolate import interp1d


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
        help="Fixed delay values in seconds.",
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
        default=2.0,
        help="Upper bound for K.",
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
        "--n-k",
        type=int,
        default=25,
        help="Number of grid points for K.",
    )
    parser.add_argument(
        "--n-tau",
        type=int,
        default=25,
        help="Number of grid points for tau.",
    )
    parser.add_argument(
        "--search-method",
        choices=["grid"],
        default="grid",
        help="Search backend used for each contour slice.",
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
    datasets = parse_datasets(args.datasets)
    theta_values = parse_theta_values(args.thetas)

    for dataset_config in datasets:
        data = dataset_config.loader()
        print(f"Loaded dataset: {dataset_config.key} ({dataset_config.label})")
        for delay_order in args.delay_orders:
            config = ContourConfig(
                dataset_key=dataset_config.key,
                dataset_label=dataset_config.label,
                delay_order=delay_order,
                theta_values=theta_values,
                k_min=args.k_min,
                k_max=args.k_max,
                tau_min=args.tau_min,
                tau_max=args.tau_max,
                n_k=args.n_k,
                n_tau=args.n_tau,
            )
            summary = run_contour_study(
                data=data,
                config=config,
                search_method=args.search_method,
                output_dir=output_dir,
            )
            summaries.append(summary)

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
