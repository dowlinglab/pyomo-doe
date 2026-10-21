"""Estimation and conditional uncertainty for the shared LTI forward models."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc, chi2

try:
    from .models import simulate_chain, simulate_original, simulate_pure_delay, to_io, to_physics
except ImportError:
    from models import simulate_chain, simulate_original, simulate_pure_delay, to_io, to_physics


@dataclass(frozen=True)
class ModelSpec:
    family: str = "chain"
    order: int = 2
    coordinates: str = "io"
    initialization: str = "cold"
    hold: str = "linear"
    method: str = "exact"

    @property
    def names(self):
        if self.family == "original":
            return ["Ua", "Ub", "CpH", "CpS"]
        return ["K", "tau", "theta"] if self.coordinates == "io" else ["Ua", "Cp", "theta"]

    @property
    def identifier(self):
        return f"{self.family}_n{self.order}_{self.coordinates}_{self.initialization}_{self.hold}_{self.method}"


def predict(data, parameters, spec):
    if spec.family == "chain":
        return simulate_chain(data, parameters, spec.order, spec.coordinates, spec.initialization, spec.hold, spec.method)
    if spec.family == "original":
        return simulate_original(data, parameters, spec.hold, spec.method)
    if spec.family == "pure_delay":
        p = to_io(parameters, power=data.power) if spec.coordinates == "physics" else parameters
        return simulate_pure_delay(data, p, spec.initialization, spec.hold)
    raise ValueError("Unknown model family")


def parameter_box(spec):
    # Broad exploration boxes, not confidence limits. Coordinate boxes differ;
    # compare interior common-feasible solutions and report boundary cases.
    if spec.family == "original":
        return np.array([0.005, 0.0001, 0.01, 0.001]), np.array([0.3, 1., 100., 20.])
    if spec.coordinates == "physics":
        return np.array([0.032/1.5, .1, .01]), np.array([0.032/.2, 100., 200.])
    return np.array([.2, 5., .01]), np.array([1.5, 600., 200.])


def default_parameters(spec):
    if spec.family == "original":
        return np.array([.0535, .0148, 6.911, .318])
    p = np.array([.78, 165., 16.])
    return to_physics(p) if spec.coordinates == "physics" else p


def residuals(log_parameters, datasets, spec):
    p = np.exp(log_parameters)
    # The first observation initializes T, so it is conditional data rather than
    # a separate zero-error observation contributing to N or residual variance.
    return np.concatenate([(predict(data, p, spec)-data.temperature)[1:] for data in datasets])


def sensitivities(parameters, datasets, spec, log=True, relative_step=1e-4):
    p = np.asarray(parameters, dtype=float)
    derivatives = []
    for j in range(len(p)):
        if spec.family != "pure_delay":
            z = p.astype(complex)
            z[j] += 1e-20j
            column = np.concatenate([predict(d, z, spec)[1:].imag/1e-20 for d in datasets])
        else:
            delta = np.zeros(len(p))
            delta[j] = p[j]*relative_step
            column = np.concatenate([(predict(d, p+delta, spec)[1:]-predict(d, p-delta, spec)[1:])/(2*delta[j]) for d in datasets])
        derivatives.append(column*(p[j] if log else 1.0))
    return np.column_stack(derivatives)


def fit(datasets, spec, n_starts=12, seed=532, starting=None, max_nfev=300):
    if not datasets:
        raise ValueError("No experiments")
    if spec.coordinates == "io" and any(d.power != datasets[0].power for d in datasets):
        raise ValueError("Shared gain requires matching heater power")
    lower, upper = parameter_box(spec)
    lo, hi = np.log(lower), np.log(upper)
    p0 = default_parameters(spec) if starting is None else np.asarray(starting)
    starts = [np.clip(np.log(p0), lo+1e-9, hi-1e-9)]
    if n_starts > 1:
        starts.extend(qmc.scale(qmc.LatinHypercube(d=len(lo), seed=seed).random(n_starts-1), lo, hi))
    results = []
    best = None
    begun = time.monotonic()
    for index, start in enumerate(starts):
        result = least_squares(residuals, start, args=(datasets, spec), bounds=(lo, hi),
                               jac="3-point", ftol=1e-10, xtol=1e-10, gtol=1e-7,
                               max_nfev=max_nfev)
        sse = float(result.fun@result.fun)
        row = {"start": index, "initial": np.exp(start).tolist(), "parameters": np.exp(result.x).tolist(),
               "sse": sse, "success": bool(result.success), "status": int(result.status),
               "optimality": float(result.optimality), "nfev": int(result.nfev),
               "message": result.message, "active_mask": result.active_mask.tolist()}
        results.append(row)
        if result.success and np.isfinite(sse) and (best is None or sse < best["sse"]):
            best = row
    if best is None:
        raise RuntimeError("No converged multistart solution")
    p = np.asarray(best["parameters"])
    if spec.family == "chain" and spec.order == 1 and spec.initialization == "cold":
        io = p if spec.coordinates == "io" else to_io(p)
        if io[1] < io[2]:
            # Choose a common-feasible representative of two-to-one pole labeling.
            candidate = io[[0, 2, 1]] if spec.coordinates == "io" else to_physics(io[[0, 2, 1]])
            if np.all(candidate >= lower) and np.all(candidate <= upper):
                p = candidate
    residual = residuals(np.log(p), datasets, spec)
    canonical = to_io(p) if spec.coordinates == "physics" and spec.family != "original" else p
    within_io = None
    if spec.family != "original":
        lio, uio = parameter_box(ModelSpec())
        within_io = bool(np.all(canonical >= lio) and np.all(canonical <= uio))
    N, k = len(residual), len(p)
    sse = float(residual@residual)
    # Conditional Gaussian criteria; serial dependence can invalidate their
    # nominal statistical interpretation. The fitted variance counts as a parameter.
    deviance = N*(np.log(2*np.pi)+1+np.log(sse/N))
    kaic = k+1
    return {"spec": asdict(spec), "parameters": dict(zip(spec.names, p.tolist())),
            "canonical_io": canonical.tolist() if spec.family != "original" else None,
            "sse": sse, "n": N, "k": k, "rmse": float(np.sqrt(sse/N)),
            "aic_iid_conditional": float(deviance+2*kaic),
            "aicc_iid_conditional": float(deviance+2*kaic+2*kaic*(kaic+1)/(N-kaic-1)),
            "bic_iid_conditional": float(deviance+kaic*np.log(N)),
            "within_common_io_box": within_io, "active_bounds": best["active_mask"],
            "starts": results, "seed": seed, "elapsed_seconds": time.monotonic()-begun}


def _sandwich(J, residual, sizes, bandwidth, bread):
    scores = J*residual[:, None]
    meat = scores.T@scores
    offset = 0
    for size in sizes:
        x = scores[offset:offset+size]
        for lag in range(1, min(bandwidth, size-1)+1):
            cross = x[lag:].T@x[:-lag]
            meat += (1-lag/(bandwidth+1))*(cross+cross.T)
        offset += size
    correction = len(residual)/max(len(residual)-J.shape[1], 1)
    return correction*bread@meat@bread


def uncertainty(fit_result, datasets, bandwidths=(10, 30, 60), sigma_known=.25):
    spec = ModelSpec(**fit_result["spec"])
    p = np.array([fit_result["parameters"][n] for n in spec.names])
    residual = residuals(np.log(p), datasets, spec)
    J = sensitivities(p, datasets, spec)
    _, singular, Vh = np.linalg.svd(J, full_matrices=False)
    gram = J.T@J
    keep = singular > singular[0]*1e-8
    rank = int(np.sum(keep))
    eigenvalues, eigenvectors = singular[::-1]**2, Vh.T[:, ::-1]
    # A pseudoinverse is diagnostic only when rank deficient, not a finite
    # uncertainty certificate along the unidentifiable directions.
    # Work directly from J rather than inverting J.T@J, which squares its
    # condition number and can yield negative diagonal roundoff at exact rank loss.
    basis = Vh[keep]
    inverse = (basis.T/(singular[keep]**2))@basis
    nonestimable = np.sum(Vh[~keep]**2, axis=0) > 1e-10
    dof = max(len(residual)-len(p), 1)
    iid = inverse*(residual@residual)/dof
    known = inverse*sigma_known**2
    sizes = [len(d.time)-1 for d in datasets]
    hac = {str(b): _sandwich(J, residual, sizes, b, inverse).tolist() for b in bandwidths}
    local_std = np.sqrt(np.maximum(np.diag(iid), 0))*p
    denominator = np.sqrt(np.maximum(np.outer(np.diag(iid), np.diag(iid)), 0))
    correlation = np.divide(iid, denominator, out=np.zeros_like(iid), where=denominator > 0)
    diagnostics = []
    offset = 0
    for data, size in zip(datasets, sizes):
        r = residual[offset:offset+size]
        centered = r-r.mean()
        den = centered@centered
        lags = min(30, size//5)
        acf = np.array([centered[l:]@centered[:-l]/den for l in range(1, lags+1)])
        Q = size*(size+2)*np.sum(acf**2/(size-np.arange(1, lags+1)))
        diagnostics.append({"dataset": data.name, "mean_residual_C": float(r.mean()),
                            "rmse_C": float(np.sqrt(r@r/size)), "acf": acf.tolist(),
                            "ljung_box_Q_30": float(Q), "ljung_box_p_approx": float(chi2.sf(Q, max(lags-len(p), 1))),
                            "notice": "Approximate whiteness diagnostic with fitted parameters; not a model-validity certificate"})
        offset += size
    out = {"parameters": spec.names, "rank_relative_1e-8": rank,
           "singular_values_log_coordinates": singular.tolist(),
           "fim_log_unit_variance": gram.tolist(), "eigenvalues_log_unit_variance": eigenvalues.tolist(),
           "eigenvectors_log_columns": eigenvectors.tolist(),
           "condition_J_log": float(singular[0]/singular[-1]),
           "condition_J_native": float(np.linalg.cond(J/p[None, :])),
           "nonestimable_native_parameters": [name for name, flag in zip(spec.names, nonestimable) if flag],
           "cov_log_iid_unknown_sigma": iid.tolist() if rank == len(p) else None,
           "cov_log_known_sigma": known.tolist() if rank == len(p) else None,
           "known_sigma_C": sigma_known, "cov_log_hac": hac if rank == len(p) else None,
           "pseudoinverse_cov_log_iid_diagnostic_only": iid.tolist() if rank < len(p) else None,
           "pseudoinverse_cov_log_hac_diagnostic_only": hac if rank < len(p) else None,
           "std_native_iid": [None if flag else float(sd) for flag, sd in zip(nonestimable, local_std)],
           "correlation_iid": correlation.tolist() if rank == len(p) else None,
           "residual_diagnostics": diagnostics,
           "uncertainty_status": "locally_rank_deficient_pseudoinverse_is_not_a_confidence_region" if rank < len(p) else "conditional_local_approximation_requires_noise_model_checks"}
    if spec.family != "original":
        # q_physics = constant + M*q_io; M is its own inverse.
        M = np.array([[-1., 0., 0.], [-1., 1., 0.], [0., 0., 1.]])
        out["alternate_cov_log_iid"] = (M@iid@M.T).tolist() if rank == len(p) else None
        out["alternate_cov_log_hac"] = {b: (M@np.asarray(c)@M.T).tolist() for b, c in hac.items()} if rank == len(p) else None
    return out
