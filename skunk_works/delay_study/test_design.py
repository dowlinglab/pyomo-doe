"""Check design linearization, gradients and actual resource constraints."""
import numpy as np
import pytest

from data import Experiment
from inference import ModelSpec, predict, sensitivities
from design import (interpolation_basis, energy_weights, build_scenario, information,
                    information_gradient, score_and_gradient, candidates, optimize)


def example(initialization="cold"):
    time = np.arange(0., 601., 5.)
    knots = np.linspace(0., 600., 13)
    basis = interpolation_basis(time, knots)
    parameters = np.array([.78, 150., 18.])
    spec = ModelSpec(order=2, initialization=initialization)
    scenario = build_scenario("synthetic", parameters, spec, time, basis)
    return time, knots, basis, parameters, spec, scenario


@pytest.mark.parametrize("initialization", ["cold", "input_equilibrated"])
def test_input_and_sensitivity_maps_reproduce_direct_simulation(initialization):
    time, knots, basis, parameters, spec, scenario = example(initialization)
    amplitudes = candidates(knots)["multisine"]
    experiment = Experiment("synthetic", time, np.full(len(time), 25.), basis@amplitudes, 25.)
    assert np.max(abs(scenario.response_map@amplitudes+25-predict(experiment, parameters, spec))) < 1e-10
    direct = sensitivities(parameters, [experiment], spec)
    mapped = np.einsum("tpk,k->tp", scenario.sensitivity_map, amplitudes)
    assert np.max(abs(direct-mapped)) < 1e-9


@pytest.mark.parametrize("criterion", ["D", "E", "theta", "prediction"])
def test_utility_gradient_matches_independent_finite_difference(criterion):
    _, knots, _, _, _, scenario = example()
    amplitudes = candidates(knots)["multisine"]
    weight = np.diag([1., 2., 3.])
    value, gradient = score_and_gradient(scenario, amplitudes, criterion, prediction_information=weight)
    finite = []
    for j in range(len(amplitudes)):
        change = np.zeros(len(amplitudes))
        change[j] = 1e-3
        hi = score_and_gradient(scenario, amplitudes+change, criterion, prediction_information=weight)[0]
        lo = score_and_gradient(scenario, amplitudes-change, criterion, prediction_information=weight)[0]
        finite.append((hi-lo)/2e-3)
    assert np.isfinite(value)
    assert np.allclose(gradient, finite, atol=2e-7, rtol=3e-4)


def test_candidate_energy_and_slew_are_actual_constraints():
    _, knots, _, _, _, _ = example()
    for amplitudes in candidates(knots).values():
        assert np.min(amplitudes) >= -1e-12
        assert np.max(amplitudes) <= 100+1e-12
        assert abs(energy_weights(knots)@amplitudes-50) < 1e-12
        assert np.max(abs(np.diff(amplitudes))/np.diff(knots)) <= 1+1e-12


def test_constant_equilibrated_input_is_rejected_as_singular_design():
    _, knots, _, _, _, scenario = example("input_equilibrated")
    value, _ = score_and_gradient(scenario, np.full(len(knots), 50.))
    assert not np.isfinite(value)


def test_optimized_input_obeys_temperature_and_energy_constraints():
    _, knots, _, _, _, scenario = example()
    starts = {"multisine": candidates(knots)["multisine"]}
    result = optimize([scenario], knots, starts, maximum_temperature=75., maxiter=150)
    assert result["best"] is not None
    amplitudes = np.array(result["best"]["amplitudes"])
    assert abs(energy_weights(knots)@amplitudes-50) < 1e-7
    assert np.max(scenario.response_map@amplitudes+25) <= 75+1e-6
    initial = score_and_gradient(scenario, starts["multisine"])[0]
    final = score_and_gradient(scenario, amplitudes)[0]
    assert final >= initial-1e-7
