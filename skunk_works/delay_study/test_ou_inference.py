"""Check stochastic recovery without requiring an exact noisy-data estimate."""
import numpy as np

from data import Experiment, load
from inference import ModelSpec, predict
from ou_inference import fit_ou, local_uncertainty


def test_recovers_synthetic_ou_mean_and_noise_with_independent_experiments():
    rng = np.random.default_rng(400)
    spec = ModelSpec(order=2)
    truth = np.array([.78, 150., 18.])
    datasets = []
    for name in ("sine", "step"):
        data = load(name)
        mean = predict(data, truth, spec)
        residual = np.empty(len(mean)-1)
        residual[0] = .2*rng.normal()
        for i, delta in enumerate(np.diff(data.time[1:]), 1):
            phi = np.exp(-delta/3.)
            residual[i] = phi*residual[i-1]+.2*np.sqrt(-np.expm1(-2*delta/3.))*rng.normal()
        observed = mean.copy()
        observed[1:] += residual
        datasets.append(Experiment(name, data.time, observed, data.input, data.ambient))
    result = fit_ou(datasets, spec, truth, n_starts=3, seed=25, maxiter=200)
    fitted = np.array([result["parameters"][name] for name in spec.names])
    assert np.allclose(fitted, truth, rtol=.05)
    assert 1.5 < result["correlation_length_s"] < 6.
    assert .15 < result["sigma_stationary_ML_C"] < .25
    assert not result["requires_revisit"]
    uncertainty = local_uncertainty(result, datasets, spec)
    assert uncertainty["mean_rank_relative_1e-8"] == 3
    assert np.linalg.eigvalsh(uncertainty["mean_cov_log_conditional_noise"]).min() > 0
    assert uncertainty["hessian_relative_step_disagreement"] < 1e-4
