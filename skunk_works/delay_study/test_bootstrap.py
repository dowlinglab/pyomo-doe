"""Check resampling assumptions without replacing the fitted forward model."""
import numpy as np
from numpy.testing import assert_allclose

from bootstrap import circular_indices, replicate_seed, resample_experiments, summarize_samples
from data import load
from inference import ModelSpec


def test_circular_blocks_preserve_internal_order():
    indices = circular_indices(73, 10, np.random.default_rng(41))
    assert len(indices) == 73
    for start in range(0, 70, 10):
        assert np.all(np.diff(indices[start:start+10]) % 73 == 1)


def test_replicate_randomness_is_independent_of_execution_order():
    expected = [replicate_seed("case", "block30", k, 532) for k in range(20)]
    actual = {k: replicate_seed("case", "block30", k, 532) for k in reversed(range(20))}
    assert expected == [actual[k] for k in range(20)]
    assert len(set(expected)) == 20


def test_resampling_preserves_conditioned_initial_reading_and_inputs():
    data = load("sine")
    a = resample_experiments([data], [.8, 150, 16], ModelSpec(), 532, block_length=30)[0]
    b = resample_experiments([data], [.8, 150, 16], ModelSpec(), 532, block_length=30)[0]
    assert_allclose(a.temperature, b.temperature, atol=0)
    assert a.temperature[0] == data.temperature[0]
    assert_allclose(a.input, data.input, atol=0)
    assert_allclose(a.time, data.time, atol=0)


def test_unidentified_parameter_never_gets_bootstrap_interval():
    samples = np.array([[.7, 140, 2.], [.8, 150, 16.], [.9, 160, 150.]])
    result = summarize_samples(samples, [True, True, True], ["theta"])
    assert result["io"]["theta"]["lower"] is None
    assert result["physics"]["theta"]["upper"] is None
    assert result["cov_log_io"] is None
    assert result["physics"]["Ua"]["lower"] > 0
