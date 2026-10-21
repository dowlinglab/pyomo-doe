"""Independent covariance and interval-geometry checks for the coverage study."""
import numpy as np

from coverage_study import sample_errors, simulate_experiments, profile_interval_summary, wilson_interval
from data import load
from inference import ModelSpec, predict


def test_synthetic_irregular_ou_covariance_matches_theory():
    time = np.array([.1, .8, 1.2, 3.4])
    noise = {"kind": "ou", "sigma_C": .25, "length_s": 2.}
    rng = np.random.default_rng(704)
    draws = np.array([sample_errors(time, noise, rng) for _ in range(10000)])
    expected = .25**2*np.exp(-abs(time[:, None]-time[None, :])/2.)
    assert np.max(abs(np.cov(draws, rowvar=False)-expected)) < .0035
    assert np.max(abs(draws.mean(axis=0))) < .01


def test_synthetic_experiment_preserves_anchor_and_source_arrays():
    data = load("sine")
    original = data.temperature.copy()
    spec = ModelSpec(order=2)
    truth = [.78, 150., 18.]
    noise = {"kind": "iid", "sigma_C": .25}
    first = simulate_experiments([data], truth, spec, noise, 571)[0]
    second = simulate_experiments([data], truth, spec, noise, 571)[0]
    assert np.array_equal(first.temperature, second.temperature)
    assert np.array_equal(data.temperature, original)
    assert first.temperature[0] == predict(data, truth, spec)[0]
    assert not np.array_equal(first.temperature[1:], original[1:])


def test_profile_coverage_does_not_fill_a_disconnected_gap():
    components = [{"lower": 1., "upper": 2., "lower_status": "crossing", "upper_status": "crossing"},
                  {"lower": 8., "upper": 9., "lower_status": "crossing", "upper_status": "crossing"}]
    result = profile_interval_summary(components, 5.)
    assert not result["covered"]
    assert result["total_component_width"] == 2.
    assert result["envelope_width"] == 8.


def test_wilson_intervals_retain_binomial_uncertainty_at_extremes():
    assert wilson_interval(0, 0) is None
    zero, all_ = wilson_interval(0, 500), wilson_interval(500, 500)
    assert zero[0] == 0. and 0. < zero[1] < .01
    assert all_[1] == 1. and .99 < all_[0] < 1.
