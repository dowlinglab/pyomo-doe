"""Check estimation against synthetic truth and uncertainty transformations."""
import numpy as np
from numpy.testing import assert_allclose

from data import load, Experiment
from models import simulate_chain, to_physics
from inference import fit, uncertainty, ModelSpec, sensitivities


def synthetic():
    source = load("sine")
    truth = np.array([.8, 100., 10.])
    temperature = simulate_chain(source, truth, 3)
    data = Experiment("synthetic", source.time, temperature, source.input, source.ambient)
    return data, truth


def test_multistart_recovers_synthetic_parameters_in_both_coordinates():
    data, truth = synthetic()
    for coordinates in ("io", "physics"):
        spec = ModelSpec(order=3, coordinates=coordinates)
        result = fit([data], spec, n_starts=3, seed=41)
        assert_allclose(result["canonical_io"], truth, rtol=2e-7)
        assert result["sse"] < 1e-14
        assert all("success" in row and "message" in row for row in result["starts"])


def test_log_fisher_transforms_between_coordinates():
    data, truth = synthetic()
    io = ModelSpec(order=3)
    physics = ModelSpec(order=3, coordinates="physics")
    Ji = sensitivities(truth, [data], io)
    Jp = sensitivities(to_physics(truth), [data], physics)
    M = np.array([[-1., 0., 0.], [-1., 1., 0.], [0., 0., 1.]])
    assert_allclose(Jp, Ji@M, atol=3e-11, rtol=1e-10)
    Ci = np.linalg.inv(Ji.T@Ji)
    Cp = np.linalg.inv(Jp.T@Jp)
    assert_allclose(Cp, M@Ci@M.T, atol=2e-13, rtol=1e-9)
    assert_allclose(Ji@Ci@Ji.T, Jp@Cp@Jp.T, atol=2e-11, rtol=1e-8)


def test_original_model_sensitivity_has_structural_null_direction():
    data = load("sine")
    p = np.array([.0535, .0148, 6.911, .318])
    J = sensitivities(p, [data], ModelSpec(family="original", coordinates="physics"))
    singular = np.linalg.svd(J, compute_uv=False)
    assert singular[-1]/singular[0] < 1e-10
    assert singular[-2]/singular[0] > 1e-4


def test_uncertainty_excludes_conditioning_measurement():
    data = load("sine")
    result = fit([data], ModelSpec(order=2), n_starts=1)
    assert result["n"] == len(data.time)-1
    output = uncertainty(result, [data])
    assert output["rank_relative_1e-8"] == 3
    assert np.all(np.linalg.eigvalsh(output["cov_log_iid_unknown_sigma"]) > 0)
    for cov in output["cov_log_hac"].values():
        assert np.linalg.eigvalsh(cov).min() > -1e-12
