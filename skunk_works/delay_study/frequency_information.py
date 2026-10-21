"""Analytic frequency sensitivities for a periodic steady-state comparison.

These are transfer-function identities, not the finite-record startup/noise
assumptions used by the fitted-data or prospective-design runners.
"""
import numpy as np


def frequency_response(parameters, order, omega):
    """Return G(i*omega) and its three log-IO derivatives, last axis K,tau,theta."""
    K, tau, theta = np.asarray(parameters, dtype=float)
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    if min(K, tau, theta) <= 0 or not np.isfinite([K, tau, theta]).all():
        raise ValueError("Positive finite transfer parameters required")
    if not isinstance(order, (int, np.integer)) or order < 0 or np.any(omega < 0) or not np.isfinite(omega).all():
        raise ValueError("Nonnegative integer order and finite nonnegative frequencies required")
    s = 1j*omega
    if order:
        G = K/((1+s*tau)*(1+s*theta/order)**order)
        htheta = -s*theta/(1+s*theta/order)
    else:
        G = K*np.exp(-s*theta)/(1+s*tau)
        htheta = -s*theta
    h = np.stack([np.ones_like(s), -s*tau/(1+s*tau), htheta], axis=-1)
    return G, G[..., None]*h


def lag_sensitivity_area(tau, theta, order, omega):
    """Determinant of real/imaginary log-transfer lag sensitivities.

The sign distinguishes local lag sensitivity independence; zero is a necessary
singularity. Positive parameters/frequency and finite order give zero exactly
when tau=theta/order. Pure delay gives positive area at every positive frequency.
"""
    denominator = 1+(omega*tau)**2
    if order:
        return omega**3*tau*theta*(tau-theta/order)/(denominator*(1+(omega*theta/order)**2))
    return omega**3*tau*tau*theta/denominator
