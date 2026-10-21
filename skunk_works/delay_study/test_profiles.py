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
