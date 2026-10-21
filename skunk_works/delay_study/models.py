"""Continuous-time LTI models with exact propagation between input knots.

The n-stage model has one thermal state and n auxiliary input states. Matrix
exponentials preserve repeated poles, including tau=theta/n. Parameter fitting
must not use a partial-fraction expression singular at that repeated pole.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from numba import njit

try:
    from .data import Experiment
except ImportError:
    from data import Experiment


@njit(cache=True)
def _propagate(transitions, interval_indices, input_values, initial):
    d = len(initial)
    states = np.empty((len(input_values), d), dtype=transitions.dtype)
    states[0] = initial
    for k in range(len(input_values)-1):
        transition = transitions[interval_indices[k]]
        for i in range(d):
            value = transition[i, d] * input_values[k] + transition[i, d+1] * input_values[k+1]
            for j in range(d):
                value += transition[i, j] * states[k, j]
            states[k+1, i] = value
    return states


def propagate_lti(A, B, initial, time, input_values, hold="linear", method="exact"):
    """Return all states; interval durations are rounded to 9 decimals only.

    Linear hold uses a block exponential integrating the exact affine input.
    Left/right hold use the specified endpoint value over each interval.
    Backward Euler/right hold reproduces the inherited Pyomo discretization.
    """
    if hold not in {"linear", "left", "right"}:
        raise ValueError("Unknown input hold")
    if method not in {"exact", "backward_euler"}:
        raise ValueError("Unknown propagation method")
    if method == "backward_euler" and hold != "right":
        raise ValueError("Backward Euler comparison requires right-end input")
    dt, interval_indices = np.unique(np.round(np.diff(time), 9), return_inverse=True)
    if np.any(dt <= 0):
        raise ValueError("Intervals must be positive after rounding")
    d = len(B)
    dtype = np.result_type(A, B, initial)
    transitions = np.empty((len(dt), d, d+2), dtype=dtype)
    augmented = np.zeros((d+2, d+2), dtype=dtype)
    augmented[:d, :d] = A
    augmented[:d, d] = B
    augmented[d, d+1] = 1
    for j, delta in enumerate(dt):
        if method == "backward_euler":
            F = np.linalg.solve(np.eye(d)-delta*A, np.eye(d))
            transitions[j, :, :d] = F
            transitions[j, :, d] = 0
            transitions[j, :, d+1] = F @ (delta*B)
        else:
            E = expm(augmented*delta)
            transitions[j, :, :d] = E[:d, :d]
            G = E[:d, d]
            H = E[:d, d+1]/delta
            transitions[j, :, d] = G-H if hold == "linear" else (G if hold == "left" else 0)
            transitions[j, :, d+1] = H if hold == "linear" else (G if hold == "right" else 0)
    return _propagate(transitions, interval_indices, input_values, np.asarray(initial, dtype=dtype))


def to_physics(io, alpha=0.00016, power=200.0):
    K, tau, theta = np.asarray(io)
    Ua = alpha*power/K
    return np.asarray([Ua, Ua*tau, theta])


def to_io(physics, alpha=0.00016, power=200.0):
    Ua, Cp, theta = np.asarray(physics)
    return np.asarray([alpha*power/Ua, Cp/Ua, theta])


def chain_matrices(parameters, order, coordinates="io", alpha=0.00016, power=200.0):
    if order < 1 or int(order) != order:
        raise ValueError("Delay order must be a positive integer")
    parameters = np.asarray(parameters)
    if np.any(np.real(parameters) <= 0):
        raise ValueError("Model parameters must be positive")
    if coordinates == "physics":
        K, tau, theta = to_io(parameters, alpha, power)
    elif coordinates == "io":
        K, tau, theta = parameters
    else:
        raise ValueError("Unknown parameter coordinates")
    A = np.zeros((order+1, order+1), dtype=np.result_type(parameters, float))
    B = np.zeros(order+1, dtype=A.dtype)
    rate = order/theta
    for j in range(order):
        A[j, j] = -rate
        if j > 0:
            A[j, j-1] = rate
    B[0] = rate
    A[-1, -2] = K/tau
    A[-1, -1] = -1/tau
    return A, B


def simulate_chain(data: Experiment, parameters, order, coordinates="io",
                   initialization="cold", hold="linear", method="exact", return_states=False):
    A, B = chain_matrices(parameters, order, coordinates, power=data.power)
    initial = np.zeros(order+1, dtype=A.dtype)
    if initialization == "input_equilibrated":
        initial[:order] = data.input[0]
    elif initialization != "cold":
        raise ValueError("Unknown delay initialization")
    initial[-1] = data.temperature[0]-data.ambient
    states = propagate_lti(A, B, initial, data.time, data.input, hold, method)
    output = states[:, -1]+data.ambient
    return (output, states) if return_states else output


def simulate_original(data: Experiment, parameters, hold="linear", method="exact"):
    """Physical two-state model with heater and sensor initially at T[0]."""
    Ua, Ub, CpH, CpS = np.asarray(parameters)
    if np.any(np.real(parameters) <= 0):
        raise ValueError("Model parameters must be positive")
    A = np.array([[-(Ua+Ub)/CpH, Ub/CpH], [Ub/CpS, -Ub/CpS]])
    B = np.array([0.00016*data.power/CpH, 0.0])
    initial = np.full(2, data.temperature[0]-data.ambient)
    return propagate_lti(A, B, initial, data.time, data.input, hold, method)[:, 1]+data.ambient


def original_transfer(parameters, alpha=0.00016, power=200.0):
    """Return K,a,b for G(s)=K/(b*s**2+a*s+1)."""
    Ua, Ub, CpH, CpS = np.asarray(parameters)
    return np.asarray([alpha*power/Ua, CpH/Ua+CpS/Ub+CpS/Ua, CpH*CpS/(Ua*Ub)])


def equivalent_original(K, a, b, sensor_time):
    """A continuous family with the same zero-state transfer function.

    sensor_time=CpS/Ub; positivity requires a-r-b/r>0. This demonstrates
    structural redundancy under fixed ambient initial temperatures.
    """
    Ua = 0.032/K
    r = sensor_time
    CpS = Ua*(a-r-b/r)
    if CpS <= 0:
        raise ValueError("Chosen family member is not physically positive")
    return np.asarray([Ua, CpS/r, b*Ua/r, CpS])


def simulate_pure_delay(data: Experiment, parameters, initialization="cold", hold="linear"):
    """Exact scalar integration for FOPDT with shifted piecewise-linear input.

    An input step at the beginning is split at its delayed arrival. It is not
    smeared across the preceding interval. Constant prehistory is explicitly
    either zero or the first input, matching the two lag-chain assumptions.
    """
    K, tau, theta = np.asarray(parameters, dtype=float)
    if min(K, tau, theta) <= 0 or hold != "linear":
        raise ValueError("Positive parameters and linear hold required")
    if initialization not in {"cold", "input_equilibrated"}:
        raise ValueError("Unknown delay initialization")
    time = data.time-data.time[0]
    shifted = time+theta
    knots = np.unique(np.concatenate([time, shifted[shifted < time[-1]]]))
    left_value = 0.0 if initialization == "cold" else data.input[0]
    states = np.empty(len(knots))
    states[0] = data.temperature[0]-data.ambient
    for i, (left, right) in enumerate(zip(knots[:-1], knots[1:])):
        dt = right-left
        if (left+right)/2 < theta:
            uleft = uright = left_value
        else:
            uleft = np.interp(left-theta, time, data.input)
            uright = np.interp(right-theta, time, data.input)
        decay = np.exp(-dt/tau)
        gain = -np.expm1(-dt/tau)
        states[i+1] = decay*states[i]+K*gain*uleft+K*(dt-tau*gain)*(uright-uleft)/dt
    return np.interp(time, knots, states)+data.ambient
