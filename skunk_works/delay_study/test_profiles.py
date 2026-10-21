"""Check nuisance profiling and preservation of disconnected confidence sets."""
import numpy as np

from data import load
from inference import ModelSpec, fit, uncertainty
from profiles import profile, interval_components, profile_grid


def test_disconnected_intervals_do_not_bridge_excluded_region():
    q = np.log([1., 2., 3., 4., 5., 6., 7.])
    statistic = [10., 0., 10., 10., 0., 0., 10.]
    components = interval_components(q, statistic, 3.84, q[0], q[-1])
    assert len(components) == 2
    assert components[0]["upper"] < components[1]["lower"]


def test_constant_equilibrated_step_has_flat_delay_profile():
    datasets = [load("step")]
    spec = ModelSpec(order=2, initialization="input_equilibrated")
    result = fit(datasets, spec, n_starts=1)
    record = {"fit": result, "uncertainty": uncertainty(result, datasets)}
    output = profile(record, datasets, 2, grid_points=9, n_starts=1, max_nfev=80)
    statistic = np.array([p["lr_unknown_sigma_iid"] for p in output["points"]])
    assert np.max(np.abs(statistic)) < 1e-5
    component = output["intervals_unknown_sigma_iid"][0]
    assert component["lower_status"] == component["upper_status"] == "open_at_exploration_bound"


def test_one_stage_grid_includes_both_pole_labels():
    datasets = [load("sine")]
    spec = ModelSpec(order=1)
    result = fit(datasets, spec, n_starts=1)
    record = {"fit": result, "uncertainty": uncertainty(result, datasets)}
    grid = profile_grid(record, 2)
    assert np.min(abs(grid-np.log(result["parameters"]["theta"]))) < 1e-12
    assert np.min(abs(grid-np.log(result["parameters"]["tau"]))) < 1e-12


def test_original_recovery_retains_failed_attempt_and_uses_complex_jacobian(monkeypatch):
    import profiles
    datasets = [load("step")]
    spec = ModelSpec(family="original", order=0, coordinates="physics")
    result = fit(datasets, spec, n_starts=1)
    record = {"fit": result, "uncertainty": uncertainty(result, datasets)}
    center = np.log(result["parameters"]["Ua"])
    monkeypatch.setattr(profiles, "profile_grid", lambda *a, **kw: np.array([center]))
    original_solver = profiles.least_squares
    calls = []

    def interrupted_first_attempt(*args, **kwargs):
        calls.append(kwargs)
        solution = original_solver(*args, **kwargs)
        if len(calls) == 1:
            # Simulate an exhausted attempt without discarding its diagnostics.
            solution.success = False
            solution.status = 0
        return solution

    monkeypatch.setattr(profiles, "least_squares", interrupted_first_attempt)
    output = profiles.profile(record, datasets, 0, n_starts=1, recover_original=True)
    attempts = output["points"][0]["attempts"]
    assert [a["phase"] for a in attempts] == ["initial", "recovery"]
    assert not attempts[0]["success"] and attempts[1]["success"]
    assert calls[1]["jac"] == "cs" and calls[1]["x_scale"] == "jac"
    assert calls[1]["max_nfev"] == 10*calls[0]["max_nfev"]
    assert abs(output["points"][0]["sse"]-result["sse"]) < 1e-5
