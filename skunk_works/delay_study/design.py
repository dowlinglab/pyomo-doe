"""Finite-horizon input design using exact linear input-to-sensitivity maps.

All utilities are local to the supplied model parameters. A full-rank local
information matrix cannot resolve a global pole-label ambiguity or structural
parameter equivalence. The constraints are study assumptions, not certified
hardware operating limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from data import Experiment
from inference import ModelSpec, predict, sensitivities
from noise import whiten_ou


@dataclass
class DesignScenario:
    name: str
    parameters: np.ndarray
    spec: ModelSpec
    response_map: np.ndarray
    sensitivity_map: np.ndarray
    prior_information: np.ndarray
    ambient: float
    sigma: float
    noise_length_s: Optional[float] = None


def interpolation_basis(time, knots):
    """Map knot amplitudes to a piecewise-linear sampled input."""
    time, knots = np.asarray(time), np.asarray(knots)
    if np.any(np.diff(time) <= 0) or np.any(np.diff(knots) <= 0):
        raise ValueError("Increasing time and knot grids required")
    if time[0] != knots[0] or time[-1] != knots[-1]:
        raise ValueError("Knot and observation horizons must match")
    return np.column_stack([np.interp(time, knots, eye) for eye in np.eye(len(knots))])


def energy_weights(knots):
    """Trapezoidal weights normalized to give a time-average input."""
    dt = np.diff(knots)
    weights = np.zeros(len(knots))
    weights[:-1] += dt/2
    weights[1:] += dt/2
    return weights/(knots[-1]-knots[0])


def build_scenario(name, parameters, spec, time, basis, ambient=25., sigma=.25,
                   prior_information=None, noise_length=None):
    if spec.coordinates != "io" or spec.family == "original":
        raise ValueError("Design uses the three IO parameters")
    responses, derivatives = [], []
    for column in basis.T:
        experiment = Experiment(name, time, np.full(len(time), ambient), column, ambient)
        responses.append(predict(experiment, parameters, spec)-ambient)
        J = sensitivities(parameters, [experiment], spec, log=True)
        if noise_length is not None:
            J, _ = whiten_ou(J, time[1:], noise_length)
        derivatives.append(J)
    prior = np.zeros((3, 3)) if prior_information is None else np.asarray(prior_information)
    return DesignScenario(name, np.asarray(parameters), spec, np.column_stack(responses),
                          np.stack(derivatives, axis=-1), prior, ambient, sigma, noise_length)


def information(scenario, amplitudes, include_prior=False):
    J = np.einsum("tpk,k->tp", scenario.sensitivity_map, amplitudes)/scenario.sigma
    F = J.T@J
    return F+scenario.prior_information if include_prior else F


def information_gradient(scenario, amplitudes):
    D = scenario.sensitivity_map/scenario.sigma
    J = np.einsum("tpk,k->tp", D, amplitudes)
    cross = np.einsum("tpk,tq->kpq", D, J)
    return cross+cross.transpose(0, 2, 1)


def score_and_gradient(scenario, amplitudes, criterion="D", include_prior=False,
                       prediction_information: Optional[np.ndarray] = None):
    F = information(scenario, amplitudes, include_prior)
    dF = information_gradient(scenario, amplitudes)
    values, vectors = np.linalg.eigh(F)
    if not np.all(np.isfinite(values)) or values[-1] <= 0:
        return -np.inf, np.zeros(len(amplitudes))
    keep = values > 1e-16*values[-1]
    if not keep.all() and criterion in {"D", "E"}:
        return -np.inf, np.zeros(len(amplitudes))
    if criterion == "D":
        inverse = np.linalg.solve(F, np.eye(3))
        return float(np.log(values).sum()), np.einsum("pq,kqp->k", inverse, dF)
    if criterion == "E":
        direction = vectors[:, 0]
        return float(np.log(values[0])), np.einsum("p,kpq,q->k", direction, dF, direction)/values[0]
    if criterion == "theta":
        weight = np.diag([0., 0., 1.])
    elif criterion == "prediction" and prediction_information is not None:
        weight = np.asarray(prediction_information)
    else:
        raise ValueError("Unknown criterion or missing prediction weight")
    # Estimable functions can have finite variance while other parameters are
    # unidentifiable. A diagnostic pseudoinverse is meaningful for this target
    # only when its weight annihilates the entire local null space.
    if not keep.all():
        if np.linalg.norm(weight@vectors[:, ~keep]) > 1e-8*max(np.linalg.norm(weight), 1e-30):
            return -np.inf, np.zeros(len(amplitudes))
        inverse = (vectors[:, keep]/values[keep])@vectors[:, keep].T
    else:
        inverse = np.linalg.solve(F, np.eye(3))
    loss = np.trace(weight@inverse)
    if loss <= 0:
        raise ValueError("Prediction/parameter target must have positive variance")
    gradient = np.einsum("pq,kqp->k", inverse@weight@inverse, dF)/loss
    return float(-np.log(loss)), gradient


def candidates(knots, mean_input=50., seed=532):
    """Feasible baseline starts on the same knot, amplitude, energy and slew set."""
    t = np.asarray(knots)-knots[0]
    w = energy_weights(knots)
    rng = np.random.default_rng(seed)
    raw = {
        "constant": np.zeros(len(t)),
        "sine_300s": np.sin(2*np.pi*t/300),
        "sine_150s": np.sin(2*np.pi*t/150),
        "multisine": np.sin(2*np.pi*t/900)+.7*np.sin(2*np.pi*t/300+.7)+.4*np.sin(2*np.pi*t/180+1.3),
        "random_binary_knots": rng.choice([-1., 1.], len(t)),
        "alternating_knots": (-1.)**np.arange(len(t)),
    }
    result = {}
    for name, shape in raw.items():
        shape = shape-w@shape
        peak = np.max(abs(shape))
        if peak > 1e-12:
            shape = shape/peak*min(mean_input, 100-mean_input)
            largest_slew = np.max(abs(np.diff(shape))/np.diff(knots))
            shape /= max(1., largest_slew)
        result[name] = mean_input+shape
    return result


def optimize(scenarios, knots, starts, criterion="D", include_prior=False,
             aggregation="mean", mean_input=50., maximum_temperature=85.,
             maximum_slew=1., maxiter=400, prediction_information=None,
             constraint_scenarios=None):
    """SLSQP multistart with analytic utility and linear-constraint gradients.

    The min aggregation is nonsmooth at scenario ties. We retain actual solver
    outcomes and do not label a multistart local solution globally optimal.
    """
    weights = energy_weights(knots)
    difference = np.diff(np.eye(len(knots)), axis=0)
    slew_bounds = maximum_slew*np.diff(knots)
    safety = scenarios if constraint_scenarios is None else constraint_scenarios
    maps = np.vstack([s.response_map for s in safety])
    bounds = np.concatenate([np.full(len(s.response_map), maximum_temperature-s.ambient) for s in safety])
    constraint_map = np.vstack([maps, difference, -difference])
    constraint_bounds = np.concatenate([bounds, slew_bounds, slew_bounds])
    def prediction_weight(scenario):
        if isinstance(prediction_information, dict):
            return prediction_information[scenario.name]
        return prediction_information

    reference = np.full(len(knots), mean_input)
    reference_scores = np.array([score_and_gradient(s, reference, criterion, include_prior, prediction_weight(s))[0] for s in scenarios])
    # Normalize across scenarios without changing their within-scenario criterion.
    if not np.all(np.isfinite(reference_scores)):
        # Constant equilibrated input can have exactly zero theta information.
        # A varying feasible reference avoids a fictitious information ridge.
        reference = candidates(knots, mean_input)["multisine"]
        reference_scores = np.array([score_and_gradient(s, reference, criterion, include_prior, prediction_weight(s))[0] for s in scenarios])
    if not np.all(np.isfinite(reference_scores)):
        raise ValueError("Reference is rank deficient; design an identifiable target")

    def utility(amplitudes):
        rows = [score_and_gradient(s, amplitudes, criterion, include_prior, prediction_weight(s)) for s in scenarios]
        values = np.array([x[0] for x in rows])-reference_scores
        gradients = np.array([x[1] for x in rows])
        if aggregation == "mean":
            return -float(values.mean()), -gradients.mean(axis=0)
        if aggregation == "min":
            index = np.argmin(values)
            return -float(values[index]), -gradients[index]
        raise ValueError("Unknown scenario aggregation")

    constraints = [
        {"type": "eq", "fun": lambda x: float(weights@x-mean_input), "jac": lambda x: weights},
        {"type": "ineq", "fun": lambda x: constraint_bounds-constraint_map@x, "jac": lambda x: -constraint_map},
    ]
    attempts = []
    for name, initial in starts.items():
        # Singular constant starts have no useful local E/theta direction.
        if not np.isfinite(utility(initial)[0]):
            attempts.append({"start": name, "status": "singular_start_skipped", "success": False})
            continue
        fitted = minimize(utility, initial, jac=True, method="SLSQP", bounds=[(0., 100.)]*len(knots),
                          constraints=constraints, options={"ftol": 1e-9, "maxiter": maxiter})
        amplitudes = fitted.x
        violation = max(float(np.max(constraint_map@amplitudes-constraint_bounds)),
                        abs(float(weights@amplitudes-mean_input)),
                        float(np.max(-amplitudes)), float(np.max(amplitudes-100)))
        attempts.append({"start": name, "success": bool(fitted.success), "solver_status": int(fitted.status),
                         "message": str(fitted.message), "iterations": int(fitted.nit),
                         "objective": float(fitted.fun) if np.isfinite(fitted.fun) else None,
                         "maximum_constraint_violation": violation,
                         "amplitudes": amplitudes.tolist()})
    accepted = [a for a in attempts if a["success"] and a["objective"] is not None
                and a["maximum_constraint_violation"] < 1e-6]
    best = min(accepted, key=lambda a: a["objective"]) if accepted else None
    return {"criterion": criterion, "aggregation": aggregation, "include_prior": include_prior,
            "reference_amplitudes": reference.tolist(), "reference_scores": reference_scores.tolist(),
            "best": best, "attempts": attempts,
            "notice": "Finite-horizon local information design under assumed models and noise. Multistart SLSQP supplies no global optimality certificate. Amplitude, energy, temperature and slew constraints are assumed study constraints; no hardware run or safety certification."}
