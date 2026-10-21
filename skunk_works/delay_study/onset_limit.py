"""Closed-form step response at the nominal delay; no chain discretization."""
from __future__ import annotations

import numpy as np
from scipy.special import gammainc


def step_at_mean_delay(order, gain=.78, thermal_time=150., delay=18., command=50.):
    """Return the cold-chain temperature rise at t=theta for a constant step.

    Integrating over an Erlang delay D gives K*u*E[1-exp(-(theta-D)/tau);D<theta].
    The exponentially tilted Erlang integral below requires n/theta > 1/tau.
    This narrow helper is for the stated convergence study, not a general solver.
    """
    if int(order) != order or order < 1 or thermal_time <= 0 or delay <= 0:
        raise ValueError("Positive times and an integer stage count required")
    a = delay/thermal_time
    if order <= a:
        raise ValueError("This tilted-CDF expression requires n > theta/tau")
    tilt = np.exp(-a-order*np.log1p(-a/order))
    return float(gain*command*(gammainc(order, order)-tilt*gammainc(order, order-a)))


def clt_coefficient(gain=.78, thermal_time=150., delay=18., command=50.):
    """Coefficient of n^-1/2 in the cold step rise at t=theta."""
    return gain*command*delay/(thermal_time*np.sqrt(2*np.pi))
