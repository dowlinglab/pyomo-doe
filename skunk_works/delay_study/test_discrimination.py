"""Check covariance weighting, gradients and a false discrimination example."""
import json
from pathlib import Path

import numpy as np

from design import build_scenario, interpolation_basis
from discrimination import (fixed_gaussian_error, refitted_separation,
                            separation_map, squared_distance_and_gradient)
from inference import ModelSpec
from run_discrimination import tasks


def test_declared_discrimination_matrix_has_no_duplicate_or_missing_jobs():
    config = json.loads((Path(__file__).parent/"discrimination_config.json").read_text())
    designs, refits = tasks(config, "designs"), tasks(config, "refits")
    assert len({row[0] for row in designs}) == len(designs) == 12
    assert len({row[0] for row in refits}) == len(refits) == 72
    assert {row[4] for row in refits} == {2, 12, 0}


def scenario_pair(initialization="cold", noise_length=10.):
    time = np.arange(0., 205., 5.)
    knots = np.arange(0., 201., 25.)
    basis = interpolation_basis(time, knots)
    first = build_scenario("n1", np.array([.75, 40., 8.]), ModelSpec(order=1, initialization=initialization),
                           time, basis, noise_length=noise_length)
    second = build_scenario("n2", np.array([.79, 50., 12.]), ModelSpec(order=2, initialization=initialization),
                            time, basis, noise_length=noise_length)
    return time, knots, basis, first, second


def test_separation_agrees_with_dense_covariance_and_finite_difference_gradient():
    time, knots, basis, first, second = scenario_pair()
    amplitudes = 50+20*np.sin(knots/30)
    matrix = separation_map(first, second, time)
    value, gradient = squared_distance_and_gradient(matrix, amplitudes)
    delta = ((first.response_map-second.response_map)@amplitudes)[1:]
    covariance = first.sigma**2*np.exp(-abs(time[1:, None]-time[None, 1:])/10.)
    assert abs(value-delta@np.linalg.solve(covariance, delta)) < 1e-9
    for j in range(len(amplitudes)):
        h = np.eye(len(amplitudes))[j]*1e-3
        numerical = (squared_distance_and_gradient(matrix, amplitudes+h)[0]-
                     squared_distance_and_gradient(matrix, amplitudes-h)[0])/.002
        assert abs(numerical-gradient[j]) < 1e-7


def test_equilibrated_constant_input_cannot_discriminate_orders_after_refitting():
    time, knots, basis, first, second = scenario_pair(initialization="input_equilibrated")
    result = refitted_separation(first, second, np.full(len(knots), 50.), time, basis, starts=2)
    assert result["fixed_parameter_distance_squared"] > 1
    assert result["best"]["distance_squared"] < 1e-20
    assert not result["requires_revisit"]


def test_pole_label_exchange_has_zero_fixed_mean_separation():
    time, knots, basis, first, second = scenario_pair(noise_length=None)
    swapped = build_scenario("exchanged", first.parameters[[0, 2, 1]], first.spec, time, basis)
    value, _ = squared_distance_and_gradient(separation_map(first, swapped, time), 50+10*np.cos(knots/20))
    assert value < 1e-21
    assert fixed_gaussian_error(0.) == .5
