"""Explicit ambient and restricted pre-window-history nuisance quantities.

The two initializations are endpoints of a linear state family. Free history
does not mean an arbitrary reconstructed prehistory. Ambient/history bounds are
study assumptions, not independently measured probabilities.
"""
from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from inference import ModelSpec, parameter_box, predict, sensitivities, _sandwich


def prediction(data, parameters, spec, ambient_offset=0., history_fraction=0.):
    if spec.coordinates != "io" or spec.family == "original":
        raise ValueError("Nuisance history is defined for reduced IO formulations")
    cold = predict(data, parameters, replace(spec, initialization="cold"))
    equilibrated = predict(data, parameters, replace(spec, initialization="input_equilibrated"))
    elapsed = data.time-data.time[0]
    return cold+history_fraction*(equilibrated-cold)+ambient_offset*(-np.expm1(-elapsed/parameters[1]))


def mean_derivatives(data, parameters, spec, ambient_offset, history_fraction, endpoints=None):
    """Columns are log K,log tau,log theta,ambient offset C,history fraction."""
    cold_spec, eq_spec = replace(spec, initialization="cold"), replace(spec, initialization="input_equilibrated")
    cold, eq = (predict(data, parameters, cold_spec), predict(data, parameters, eq_spec)) if endpoints is None else endpoints
    J = (1-history_fraction)*sensitivities(parameters, [data], cold_spec)+history_fraction*sensitivities(parameters, [data], eq_spec)
    scaled_time = (data.time-data.time[0])[1:]/parameters[1]
    decay = np.exp(-scaled_time)
    J[:, 1] -= ambient_offset*scaled_time*decay
    return np.column_stack([J, 1-decay, (eq-cold)[1:]])


def layout(spec, count, free_ambient, free_history):
    lower, upper = parameter_box(spec)
    names = list(spec.names)
    lo, hi = list(np.log(lower)), list(np.log(upper))
    if free_ambient:
        names += ["ambient_offset_"+str(i) for i in range(count)]
        lo += [-2.]*count
        hi += [2.]*count
    if free_history:
        names += ["history_fraction_"+str(i) for i in range(count)]
        lo += [0.]*count
        hi += [1.]*count
    return names, np.asarray(lo), np.asarray(hi)


def unpack(q, count, free_ambient, free_history, ambient_offset=0., history_fraction=0.):
    parameters = np.exp(q[:3])
    cursor = 3
    offsets = q[cursor:cursor+count] if free_ambient else np.full(count, ambient_offset)
    cursor += count if free_ambient else 0
    histories = q[cursor:cursor+count] if free_history else np.full(count, history_fraction)
    return parameters, offsets, histories


def residual_and_jacobian(q, data, spec, free_ambient, free_history, ambient_offset=0., history_fraction=0.):
    parameters, offsets, histories = unpack(q, len(data), free_ambient, free_history, ambient_offset, history_fraction)
    columns = 3+len(data)*(int(free_ambient)+int(free_history))
    residuals, derivatives = [], []
    for i, experiment in enumerate(data):
        endpoints = (predict(experiment, parameters, replace(spec, initialization="cold")),
                     predict(experiment, parameters, replace(spec, initialization="input_equilibrated")))
        elapsed = experiment.time-experiment.time[0]
        mean = endpoints[0]+histories[i]*(endpoints[1]-endpoints[0])+offsets[i]*(-np.expm1(-elapsed/parameters[1]))
        residuals.append((mean-experiment.temperature)[1:])
        D = mean_derivatives(experiment, parameters, spec, offsets[i], histories[i], endpoints)
        J = np.zeros((len(D), columns))
        J[:, :3] = D[:, :3]
        if free_ambient:
            J[:, 3+i] = D[:, 3]
        if free_history:
            J[:, 3+len(data)*int(free_ambient)+i] = D[:, 4]
        derivatives.append(J)
    return np.concatenate(residuals), np.vstack(derivatives)


def exchanged_history(data, parameters, offsets, histories):
    """Initial fractions that preserve an n1 response after exchanging poles.

    Return None if the restricted history family cannot represent the exchange.
    The derivation accounts for known initial sensor-minus-ambient deviation.
    """
    K, tau, theta = parameters
    result = []
    for experiment, offset, fraction in zip(data, offsets, histories):
        if experiment.input[0] == 0:
            return None
        initial_deviation = experiment.temperature[0]-experiment.ambient-offset
        value = (fraction*theta+initial_deviation*(tau-theta)/(K*experiment.input[0]))/tau
        if not 0 <= value <= 1:
            return None
        result.append(value)
    return np.asarray(result)


def fit_nuisance(data, spec, starting, free_ambient=False, free_history=False,
                 ambient_offset=0., history_fraction=0., n_starts=24, seed=532, max_nfev=1200):
    if spec.coordinates != "io" or spec.family == "original":
        raise ValueError("Nuisance fit uses reduced IO formulations")
    names, lo, hi = layout(spec, len(data), free_ambient, free_history)
    initial = list(np.log(starting))
    if free_ambient:
        initial += [ambient_offset]*len(data)
    if free_history:
        initial += [history_fraction]*len(data)
    starts = [np.clip(np.asarray(initial), lo+1e-9, hi-1e-9)]
    if n_starts > 1:
        starts.extend(qmc.scale(qmc.LatinHypercube(len(lo), seed=seed).random(n_starts-1), lo, hi))
    attempts, best = [], None
    arguments = (data, spec, free_ambient, free_history, ambient_offset, history_fraction)
    cached_point, cached_evaluation = None, None

    def evaluate(q):
        nonlocal cached_point, cached_evaluation
        if cached_point is None or not np.array_equal(q, cached_point):
            cached_evaluation = residual_and_jacobian(q, *arguments)
            cached_point = q.copy()
        return cached_evaluation

    for index, point in enumerate(starts):
        result = least_squares(lambda q: evaluate(q)[0], point,
                               jac=lambda q: evaluate(q)[1], bounds=(lo, hi),
                               ftol=1e-10, xtol=1e-10, gtol=1e-7, max_nfev=max_nfev)
        sse = float(result.fun@result.fun)
        row = {"start": index, "initial_mixed_coordinates": point.tolist(), "mixed_coordinates": result.x.tolist(),
               "sse": sse, "success": bool(result.success), "status": int(result.status), "nfev": int(result.nfev),
               "optimality": float(result.optimality), "active_mask": result.active_mask.tolist(), "message": str(result.message)}
        attempts.append(row)
        if result.success and (best is None or sse < best["sse"]):
            best = row
    if best is None:
        raise RuntimeError("No nuisance fit converged")
    q = np.asarray(best["mixed_coordinates"])
    parameters, offsets, histories = unpack(q, len(data), free_ambient, free_history, ambient_offset, history_fraction)
    residual, J = residual_and_jacobian(q, *arguments)
    _, singular, Vh = np.linalg.svd(J, full_matrices=False)
    keep = singular > 1e-8*singular[0]
    rank = int(keep.sum())
    bread = (Vh[keep].T/singular[keep]**2)@Vh[keep]
    nonestimable = np.sum(Vh[~keep]**2, axis=0) > 1e-10
    covariance = None
    if rank == len(q):
        covariance = (bread*(residual@residual)/max(1, len(residual)-len(q))).tolist()
    exchange = None
    if spec.family == "chain" and spec.order == 1 and free_history:
        fractions = exchanged_history(data, parameters, offsets, histories)
        if fractions is not None:
            swapped = parameters[[0, 2, 1]]
            difference = max(float(np.max(abs(prediction(d, parameters, spec, a, g)-prediction(d, swapped, spec, a, other))))
                             for d, a, g, other in zip(data, offsets, histories, fractions))
            exchange = {"parameters": swapped.tolist(), "history_fractions": fractions.tolist(),
                        "within_io_fit_box": bool(np.all(swapped >= np.exp(lo[:3])) and np.all(swapped <= np.exp(hi[:3]))),
                        "maximum_prediction_difference_C": difference,
                        "notice": "Admissibility of the exchanged initial fractions is checked. No arbitrary pole-label canonicalization is applied."}
    return {"parameters_io": dict(zip(spec.names, parameters.tolist())), "ambient_offsets_C": offsets.tolist(),
            "history_fractions": histories.tolist(), "mixed_coordinate_names": names, "mixed_coordinates": q.tolist(),
            "bounds_mixed": [lo.tolist(), hi.tolist()], "sse": best["sse"], "n_conditional": len(residual),
            "rmse_C": float(np.sqrt(best["sse"]/len(residual))), "rank_relative_1e-8": rank,
            "singular_values_mixed_metric": singular.tolist(), "right_singular_vectors_rows": Vh.tolist(),
            "cov_mixed_iid_unknown_sigma": covariance, "selected_active_mask": best["active_mask"],
            "cov_mixed_known_sigma_0p25": (bread*.25**2).tolist() if rank == len(q) else None,
            "cov_mixed_hac60": _sandwich(J, residual, [len(d.time)-1 for d in data], 60, bread).tolist() if rank == len(q) else None,
            "nonestimable_mixed_coordinates": [name for name, flag in zip(names, nonestimable) if flag],
            "pseudoinverse_mixed_unit_variance_diagnostic_only": bread.tolist() if rank < len(q) else None,
            "free_ambient": free_ambient, "free_history": free_history, "attempts": attempts,
            "exchanged_n1_representation": exchange,
            "notice": "Mean-model sensitivity to ambient and a restricted initial-state family. Offsets are bounded at +/-2 C and fractions at [0,1] as study assumptions. Mixed coordinates contain log IO parameters, offsets in C and fractions; their condition number is metric dependent. Local IID curvature does not account for mean discrepancy, serial residuals, active bounds or global pole modes."}
