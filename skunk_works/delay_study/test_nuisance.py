"""Check nuisance derivatives and history-dependent global ambiguity."""
from dataclasses import replace
import numpy as np
import pytest

from data import Experiment
from inference import ModelSpec
from models import chain_matrices, propagate_lti
from nuisance import prediction, mean_derivatives, exchanged_history, fit_nuisance


def data():
    t = np.linspace(0., 120., 121)
    return Experiment("nuisance", t, np.full(len(t), 25.), 40+20*np.cos(t/10), 24.5)


def test_linear_nuisance_representation_against_direct_state_initialization():
    experiment = data()
    p, offset, fraction = np.array([.8, 25., 4.]), .4, .37
    spec = ModelSpec(order=3)
    A, B = chain_matrices(p, 3)
    initial = np.r_[np.full(3, fraction*experiment.input[0]), experiment.temperature[0]-experiment.ambient-offset]
    actual = propagate_lti(A, B, initial, experiment.time, experiment.input)[:, -1]+experiment.ambient+offset
    assert np.max(abs(prediction(experiment, p, spec, offset, fraction)-actual)) < 1e-12


def test_all_nuisance_mean_derivatives_against_finite_differences():
    experiment, spec = data(), ModelSpec(order=2)
    q = np.r_[np.log([.8, 25., 4.]), .4, .37]
    analytic = mean_derivatives(experiment, np.exp(q[:3]), spec, q[3], q[4])
    finite = []
    for j in range(5):
        delta = np.zeros(5)
        delta[j] = 1e-5
        hi, lo = q+delta, q-delta
        finite.append((prediction(experiment, np.exp(hi[:3]), spec, hi[3], hi[4])-prediction(experiment, np.exp(lo[:3]), spec, lo[3], lo[4]))[1:]/2e-5)
    assert np.allclose(analytic, np.column_stack(finite), atol=3e-8, rtol=1e-6)


def test_free_history_can_restore_one_stage_pole_interchange():
    experiment, spec = data(), ModelSpec(order=1)
    p, offset, fraction = np.array([.8, 25., 4.]), .4, .37
    exchanged = exchanged_history([experiment], p, [offset], [fraction])
    assert exchanged is not None
    first = prediction(experiment, p, spec, offset, fraction)
    second = prediction(experiment, p[[0, 2, 1]], spec, offset, exchanged[0])
    assert np.max(abs(first-second)) < 2e-12


@pytest.mark.parametrize("order", [1, 2, 12, 0])
def test_constant_input_has_gain_ambient_history_equivalence(order):
    experiment = replace(data(), input=np.full(121, 50.))
    spec = ModelSpec(order=order, family="chain" if order else "pure_delay")
    p, offset, fraction = np.array([.8, 25., 4.]), .4, .3
    new_p = np.array([.82, 25., 4.])
    new_fraction = 1-p[0]*(1-fraction)/new_p[0]
    new_offset = offset+50*(p[0]-new_p[0])
    assert 0 < new_fraction < 1 and -2 < new_offset < 2
    first = prediction(experiment, p, spec, offset, fraction)
    second = prediction(experiment, new_p, spec, new_offset, new_fraction)
    assert np.max(abs(first-second)) < 3e-12


def test_joint_nuisance_fit_recovers_noiseless_mean_from_perturbed_start():
    first = data()
    second = replace(first, name="second", input=35+25*np.sin(first.time/7))
    spec, truth = ModelSpec(order=2), np.array([.8, 25., 4.])
    datasets = [replace(d, temperature=prediction(d, truth, spec, a, g))
                for d, a, g in zip([first, second], [.4, -.3], [.2, .8])]
    result = fit_nuisance(datasets, spec, [.77, 28., 6.], True, True,
                          history_fraction=.5, n_starts=4, max_nfev=500)
    assert result["sse"] < 1e-13
    assert np.allclose(list(result["parameters_io"].values()), truth, rtol=2e-6)
    assert np.allclose(result["ambient_offsets_C"], [.4, -.3], atol=2e-7)
    assert np.allclose(result["history_fractions"], [.2, .8], atol=2e-7)
