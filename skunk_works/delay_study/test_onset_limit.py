"""Independent quadrature, matrix propagation and asymptotic-limit checks."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import gamma

from data import Experiment
from inference import ModelSpec, predict
from onset_limit import step_at_mean_delay, clt_coefficient


@pytest.mark.parametrize("order", [1, 2, 12])
def test_closed_form_against_independent_integral_and_state_propagation(order):
    K, tau, theta, command = .78, 150., 18., 50.
    integral = quad(lambda d: K*command*(-np.expm1(-(theta-d)/tau))*gamma.pdf(d, order, scale=theta/order),
                    0., theta, epsabs=1e-11)[0]
    time = np.linspace(0., theta, 7)
    data = Experiment("step", time, np.full(len(time), 25.), np.full(len(time), command), 25.)
    simulated = predict(data, [K, tau, theta], ModelSpec(order=order))[-1]-25.
    closed = step_at_mean_delay(order)
    assert abs(closed-integral) < 1e-11
    assert abs(closed-simulated) < 1e-11


def test_scaled_onset_converges_to_clt_coefficient():
    scaled = np.sqrt(65536)*step_at_mean_delay(65536)
    assert abs(scaled/clt_coefficient()-1) < .002
