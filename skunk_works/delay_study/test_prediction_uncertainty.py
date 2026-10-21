"""Prediction-space invariance and failures hidden by a parameter pseudoinverse."""
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

from data import Experiment, load
from inference import ModelSpec, sensitivities, predict
from models import original_transfer
from prediction_uncertainty import (estimable_variance, gaussian_mixture_quantiles,
                                     constant_equilibrated_target, local_covariances)


def quotient(J, sigma=.25):
    _, s, Vh = np.linalg.svd(J, full_matrices=False)
    keep = s > s[0]*1e-8
    return (Vh[keep].T/s[keep]**2)@Vh[keep]*sigma**2, Vh[~keep].T


def test_original_gauge_preserves_prediction_variance_of_identifiable_transfer():
    p = [.0535, .0148, 6.911, .318]
    K, a, b = original_transfer(p)
    fast, slow = np.sort(np.roots([1., -a, b]))
    original, chain = ModelSpec(family="original", order=0, coordinates="physics"), ModelSpec(order=1)
    data = [load("sine"), load("step")]
    future = load("validation_experiment_env_2_sin_2_50_run_1.csv")
    Cphysical, null = quotient(sensitivities(p, data, original))
    Cio, _ = quotient(sensitivities([K, slow, fast], data, chain))
    assert null.shape == (4, 1)
    first, defined = estimable_variance(sensitivities(p, [future], original), Cphysical, null)
    second, _ = estimable_variance(sensitivities([K, slow, fast], [future], chain), Cio)
    assert defined.all()
    assert np.allclose(first, second, atol=1e-12, rtol=2e-10)


def test_step_null_delay_allows_step_prediction_but_not_varying_input_prediction():
    p, spec = [.78, 150., 18.], ModelSpec(order=2, initialization="input_equilibrated")
    step = load("step")
    C, null = quotient(sensitivities(p, [step], spec))
    assert null.shape == (3, 1)
    step_variance, step_defined = estimable_variance(sensitivities(p, [step], spec), C, null)
    varying_variance, varying_defined = estimable_variance(sensitivities(p, [load("sine")], spec), C, null)
    assert step_defined.all() and np.isfinite(step_variance).all()
    assert not varying_defined.all()
    assert np.isnan(varying_variance[~varying_defined]).all()


def test_gaussian_mixture_quantiles_reduce_to_known_gaussian_and_handle_two_modes():
    means = np.tile([20., 30.], (4, 1))
    result, error = gaussian_mixture_quantiles(means, .25)
    assert np.allclose(result, means[0][None, :]+.25*norm.ppf([.025, .975])[:, None], atol=1e-9)
    assert error < 1e-9
    means = np.array([[-3., 2.], [3., 4.]])
    result, error = gaussian_mixture_quantiles(means, .25)
    assert error < 1e-9
    assert np.allclose(ndtr((result[:, None, :]-means[None, :, :])/.25).mean(axis=1), np.array([.025, .975])[:, None], atol=1e-9)


def test_zero_local_delay_derivative_does_not_prove_global_prediction_identification():
    p, spec = [.78, 150., 18.], ModelSpec(family="pure_delay", order=0, initialization="input_equilibrated")
    C, null = quotient(sensitivities(p, [load("step")], spec))
    time = np.arange(11.)
    future = Experiment("short_varying_target", time, np.full(11, 25.), 50+10*np.sin(time), 25.)
    _, locally_defined = estimable_variance(sensitivities(p, [future], spec), C, null)
    assert locally_defined.all()  # the chosen delay exceeds the target horizon
    assert np.max(abs(predict(future, p, spec)-predict(future, [.78, 150., 1.], spec))) > .01
    assert not constant_equilibrated_target(future, spec)
    assert constant_equilibrated_target(load("step"), spec)


def test_effective_rank_dof_and_hac_prediction_are_invariant_to_original_gauge():
    physical = [.0535, .0148, 6.911, .318]
    K, a, b = original_transfer(physical)
    fast, slow = np.sort(np.roots([1., -a, b]))
    data = [load("sine"), load("step")]
    target = load("validation_experiment_env_2_sin_2_50_run_1.csv")
    original, chain = ModelSpec(family="original", order=0, coordinates="physics"), ModelSpec(order=1)
    residual = np.concatenate([(predict(d, physical, original)-d.temperature)[1:] for d in data])
    sizes = [len(d.time)-1 for d in data]
    first, null, meta = local_covariances(sensitivities(physical, data, original), residual, sizes)
    second, _, other = local_covariances(sensitivities([K, slow, fast], data, chain), residual, sizes)
    assert meta["residual_dof"] == other["residual_dof"] == len(residual)-3
    for method in first:
        v1, allowed = estimable_variance(sensitivities(physical, [target], original), first[method], null)
        v2, _ = estimable_variance(sensitivities([K, slow, fast], [target], chain), second[method])
        assert allowed.all()
        assert np.allclose(v1, v2, atol=2e-9, rtol=2e-10)
