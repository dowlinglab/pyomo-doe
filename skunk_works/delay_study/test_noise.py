"""Verify the irregular-time likelihood against dense covariance calculations."""
import numpy as np
import pytest

from data import load
from inference import ModelSpec
from noise import whiten_ou, profile_ou, objective


def test_ou_whitening_matches_dense_gaussian_likelihood():
    time = np.array([0., .7, 1.1, 3.2, 3.9, 7.4])
    residual = np.array([.1, -.3, .2, .15, -.08, .4])
    length = 1.7
    R = np.exp(-abs(time[:, None]-time[None, :])/length)
    w, logdet = whiten_ou(residual, time, length)
    assert np.isclose(w@w, residual@np.linalg.solve(R, residual), rtol=1e-12)
    assert np.isclose(logdet, np.linalg.slogdet(R)[1], atol=1e-12)
    n = len(time)
    sigma = np.sqrt(residual@np.linalg.solve(R, residual)/n)
    dense = .5*(n*np.log(2*np.pi*sigma*sigma)+n+np.linalg.slogdet(R)[1])
    value = profile_ou([residual], [time], length)
    assert np.isclose(value["negative_log_likelihood"], dense, atol=1e-12)


def test_small_correlation_length_recovers_iid_and_resets_between_experiments():
    groups = [np.array([1., 2., 3.]), np.array([-.5, .3])]
    times = [np.array([0., 1., 2.]), np.array([0., 1.])]
    value = profile_ou(groups, times, .001, known_sigma=.5)
    rss = sum(x@x for x in groups)
    expected = .5*(5*np.log(2*np.pi*.25)+rss/.25)
    assert np.isclose(value["negative_log_likelihood"], expected)
    # The first residual of each experiment has its own stationary distribution.
    finite = profile_ou(groups, times, 10.)
    assert finite["whitened_residuals"][1][0] == -.5


@pytest.mark.parametrize("known_sigma", [None, .25])
def test_likelihood_gradient_matches_finite_difference(known_sigma):
    data = [load("sine"), load("step")]
    spec = ModelSpec(order=2)
    point = np.log([.78, 150., 16., 12.])
    value, gradient = objective(point, data, spec, known_sigma)
    finite = []
    for j in range(len(point)):
        change = np.zeros(len(point))
        change[j] = 1e-5
        hi = objective(point+change, data, spec, known_sigma)[0]
        lo = objective(point-change, data, spec, known_sigma)[0]
        finite.append((hi-lo)/2e-5)
    assert np.isfinite(value)
    assert np.allclose(gradient, finite, rtol=2e-6, atol=2e-4)


def test_whitened_sensitivity_columns_match_dense_triangular_solve():
    time = np.array([0., 1., 2.2, 4.])
    X = np.arange(12., dtype=float).reshape(4, 3)
    R = np.exp(-abs(time[:, None]-time[None, :])/2.)
    white, _ = whiten_ou(X, time, 2.)
    assert np.allclose(white, np.linalg.solve(np.linalg.cholesky(R), X), atol=1e-12)
