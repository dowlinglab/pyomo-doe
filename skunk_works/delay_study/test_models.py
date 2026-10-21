"""Independent checks of model physics, identifiability and propagation."""
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.special import gammainc

from data import Experiment, load, audit
from models import (chain_matrices, simulate_chain, to_physics, to_io,
                    simulate_original, original_transfer, equivalent_original,
                    simulate_pure_delay)


def synthetic(input_value=50.0):
    t = np.linspace(0, 100, 101)
    return Experiment("constant", t, np.full(len(t), 23.0), np.full(len(t), input_value), 23.0)


def test_coordinates_roundtrip_and_prediction_invariance():
    p = np.array([0.8, 165.0, 16.0])
    assert_allclose(to_io(to_physics(p)), p, rtol=2e-15)
    data = load("sine")
    for n in (1, 2, 3, 8, 12):
        a = simulate_chain(data, p, n)
        b = simulate_chain(data, to_physics(p), n, coordinates="physics")
        assert_allclose(a, b, atol=2e-12, rtol=2e-14)


def test_repeated_pole_has_known_erlang_step_response():
    data = synthetic()
    for n in (1, 2, 3, 8):
        # n delay states and the thermal state have the same time constant.
        p = np.array([0.8, 5.0, n*5.0])
        actual = simulate_chain(data, p, n)
        expected = data.ambient+40*gammainc(n+1, data.time/5)
        assert_allclose(actual, expected, atol=2e-11, rtol=1e-12)


def test_irregular_time_linear_hold_matches_independent_integrator():
    t = np.array([0.0, 0.7, 2.2, 3.0, 4.8, 8.1, 10.0])
    data = Experiment("irregular", t, np.full(len(t), 23.0), np.array([0, 20, 80, 25, 100, 60, 0]), 23.0)
    parameters = np.array([0.8, 15.0, 7.0])
    A, B = chain_matrices(parameters, 3)
    state = np.zeros(4)
    reference = [23.0]
    for left, right, uleft, uright in zip(t[:-1], t[1:], data.input[:-1], data.input[1:]):
        def rhs(tt, yy):
            u = uleft+(uright-uleft)*(tt-left)/(right-left)
            return A@yy+B*u
        solution = solve_ivp(rhs, (left, right), state, method="DOP853", rtol=2e-12, atol=2e-13)
        assert solution.success
        state = solution.y[:, -1]
        reference.append(state[-1]+23)
    assert_allclose(simulate_chain(data, parameters, 3), reference, atol=3e-10)


def test_zero_input_ambient_equilibrium():
    d = synthetic(0)
    for n in (1, 4, 12):
        assert_allclose(simulate_chain(d, [0.8, 165, 16], n), 23, atol=0)


def test_legacy_backward_euler_baseline():
    p = [0.7696155198801705, 165.05213342442886, 16.034829418065147]
    root = Path(__file__).resolve().parent / "baseline"
    for name in ("sine", "step"):
        reference = np.genfromtxt(root/f"{name}_delay2_rep0.csv", delimiter=",", names=True)
        actual = simulate_chain(load(name), p, 2, initialization="input_equilibrated", hold="right", method="backward_euler")
        assert_allclose(actual, reference["Tm"], atol=1e-9, rtol=1e-11)


def test_original_continuum_and_transfer_match():
    data = load("sine")
    physical = np.array([0.0535, 0.0148, 6.911, 0.318])
    K, a, b = original_transfer(physical)
    fast, slow = np.sort(np.roots([1.0, -a, b]))
    reference = simulate_original(data, physical)
    for r in np.linspace(fast*1.1, slow*0.9, 8):
        member = equivalent_original(K, a, b, r)
        assert_allclose(original_transfer(member), [K, a, b], rtol=1e-13)
        assert_allclose(simulate_original(data, member), reference, atol=2e-10)
    assert_allclose(simulate_chain(data, [K, slow, fast], 1), reference, atol=2e-10)


def test_one_stage_pole_interchange_depends_on_initialization():
    data = load("sine")
    p = [0.8, 165, 16]
    swapped = [0.8, 16, 165]
    assert_allclose(simulate_chain(data, p, 1), simulate_chain(data, swapped, 1), atol=2e-10)
    difference = np.max(np.abs(simulate_chain(data, p, 1, initialization="input_equilibrated")-
                               simulate_chain(data, swapped, 1, initialization="input_equilibrated")))
    assert difference > 1.0


def test_pure_delay_step_analytic_and_chain_limit():
    data = synthetic()
    p = [0.8, 15, 7.3]
    expected = 23+40*(-np.expm1(-np.maximum(data.time-p[2], 0)/p[1]))
    assert_allclose(simulate_pure_delay(data, p), expected, atol=5e-13)
    limit = simulate_pure_delay(data, p)
    errors = [np.linalg.norm(simulate_chain(data, p, n)-limit) for n in (2, 8, 32)]
    assert errors[2] < errors[1] < errors[0]


@pytest.mark.parametrize("hold", ["left", "right"])
@pytest.mark.parametrize("initialization", ["cold", "input_equilibrated"])
def test_pure_delay_piecewise_constant_holds_against_convolution(hold, initialization):
    time = np.array([0., .7, 2.2, 3., 4.8, 8.1, 10.])
    command = np.array([15., 20., 80., 25., 100., 60., 0.])
    data = Experiment("irregular_hold", time, np.full(len(time), 23.), command, 22.)
    K, tau, theta = .8, 3., 1.3
    expected = []
    for t in time:
        value = 22+np.exp(-t/tau)
        # Independent convolution of delayed constant forcing segments.
        prehistory = 0. if initialization == "cold" else command[0]
        segments = [(0., theta, prehistory)]
        segments.extend((a+theta, b+theta, command[i+int(hold == "right")])
                        for i, (a, b) in enumerate(zip(time[:-1], time[1:])))
        for a, b, u in segments:
            b = min(b, t)
            if b > a:
                value += K*u*(np.exp(-(t-b)/tau)-np.exp(-(t-a)/tau))
        expected.append(value)
    assert_allclose(simulate_pure_delay(data, [K, tau, theta], initialization, hold), expected, atol=2e-13)


def test_pure_delay_does_not_silently_ignore_a_numerical_method():
    from inference import ModelSpec, predict
    with pytest.raises(ValueError, match="exact shifted-knot"):
        predict(synthetic(), [.8, 15., 7.3], ModelSpec(family="pure_delay", order=0, method="backward_euler"))


def test_pure_delay_linear_hold_preserves_completed_baseline():
    import pandas as pd
    root = Path(__file__).resolve().parent
    record = json.loads((root/"results/fits/joint__pure_delay_n0_io_cold_linear_exact.json").read_text())
    parameters = [record["fit"]["parameters"][name] for name in ("K", "tau", "theta")]
    frame = pd.read_csv(root/"results"/record["prediction_file"])
    for name in ("sine", "step"):
        saved = frame[(frame.role == "calibration") & (frame.experiment == name)]
        assert_allclose(simulate_pure_delay(load(name), parameters), saved.predicted_C, atol=2e-12)


def test_complex_step_sensitivity_matches_central_difference():
    data = load("sine")
    p = np.array([0.8, 165, 16])
    for j in range(3):
        delta = np.zeros(3)
        delta[j] = 1e-4*p[j]
        central = (simulate_chain(data, p+delta, 3)-simulate_chain(data, p-delta, 3))/(2*delta[j])
        z = p.astype(complex)
        z[j] += 1e-20j
        exact = simulate_chain(data, z, 3).imag/1e-20
        assert_allclose(central, exact, atol=1e-8, rtol=2e-6)


def test_audit_detects_calibration_duplicates():
    groups = audit()["exact_numeric_duplicates"]
    assert any("data/tclab_step_test.csv" in group and "data/validation_experiment_env_2_step_50_run_1.csv" in group for group in groups)
    assert any("data/tclab_sine_test_5min_period.csv" in group and "data/validation_experiment_env_2_sin_5_50_run_1.csv" in group for group in groups)
