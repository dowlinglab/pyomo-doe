"""Check design linearization, gradients and actual resource constraints."""
from dataclasses import replace
import numpy as np
import pytest

from data import Experiment
from inference import ModelSpec, predict, sensitivities
from design import (interpolation_basis, energy_weights, build_scenario, information,
                    information_gradient, score_and_gradient, candidates, optimize)


def example(initialization="cold", noise_length=None):
    time = np.arange(0., 601., 5.)
    knots = np.linspace(0., 600., 13)
    basis = interpolation_basis(time, knots)
    parameters = np.array([.78, 150., 18.])
    spec = ModelSpec(order=2, initialization=initialization)
    scenario = build_scenario("synthetic", parameters, spec, time, basis, noise_length=noise_length)
    return time, knots, basis, parameters, spec, scenario


def test_corresponding_physical_design_preserves_D_delay_and_prediction_utilities():
    _, knots, _, _, _, base = example()
    M = np.array([[-1., 0., 0.], [-1., 1., 0.], [0., 0., 1.]])
    amplitudes = candidates(knots)["multisine"]
    prior = np.diag([200., 100., 3.])
    io = replace(base, prior_information=prior)
    physical = replace(base, name="same_response_physical_log_metric",
                       sensitivity_map=np.einsum("tpk,pq->tqk", base.sensitivity_map, M),
                       prior_information=M.T@prior@M)
    A = np.diag([1., 2., 3.])
    for historical in (False, True):
        for criterion in ("D", "theta", "prediction"):
            value, gradient = score_and_gradient(io, amplitudes, criterion, historical, A)
            alternate, alternate_gradient = score_and_gradient(physical, amplitudes, criterion, historical, M.T@A@M)
            assert abs(value-alternate) < 1e-10
            assert np.allclose(gradient, alternate_gradient, rtol=1e-10, atol=1e-12)
        first = score_and_gradient(io, amplitudes, "E", historical)[0]
        second = score_and_gradient(physical, amplitudes, "E", historical)[0]
        assert abs(first-second) > .01


@pytest.mark.parametrize("initialization", ["cold", "input_equilibrated"])
@pytest.mark.parametrize("noise_length", [None, 10.])
def test_input_and_sensitivity_maps_reproduce_direct_simulation(initialization, noise_length):
    from noise import whiten_ou
    time, knots, basis, parameters, spec, scenario = example(initialization, noise_length)
    amplitudes = candidates(knots)["multisine"]
    experiment = Experiment("synthetic", time, np.full(len(time), 25.), basis@amplitudes, 25.)
    assert np.max(abs(scenario.response_map@amplitudes+25-predict(experiment, parameters, spec))) < 1e-10
    direct = sensitivities(parameters, [experiment], spec)
    if noise_length is not None:
        direct, _ = whiten_ou(direct, time[1:], noise_length)
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


def test_prediction_target_can_be_estimable_with_a_null_delay():
    _, knots, _, _, _, scenario = example("input_equilibrated")
    amplitudes = np.full(len(knots), 50.)
    J = np.einsum("tpk,k->tp", scenario.sensitivity_map, amplitudes)
    target = J.T@J
    assert np.linalg.matrix_rank(information(scenario, amplitudes), tol=1e-8) == 2
    value, gradient = score_and_gradient(scenario, amplitudes, "prediction", prediction_information=target)
    delay_value, _ = score_and_gradient(scenario, amplitudes, "theta")
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert not np.isfinite(delay_value)


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


def test_scenario_specific_prediction_weight_matches_single_target():
    _, knots, _, _, _, scenario = example()
    starts = {"multisine": candidates(knots)["multisine"]}
    target = np.diag([1., 2., 3.])
    common = dict(criterion="prediction", maximum_temperature=75., maxiter=150)
    direct = optimize([scenario], knots, starts, prediction_information=target, **common)
    mapped = optimize([scenario], knots, starts, prediction_information={scenario.name: target}, **common)
    assert direct["best"] is not None and mapped["best"] is not None
    assert np.allclose(direct["best"]["amplitudes"], mapped["best"]["amplitudes"], atol=1e-9)
    assert abs(direct["best"]["objective"]-mapped["best"]["objective"]) < 1e-12
