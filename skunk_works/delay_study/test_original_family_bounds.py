"""Challenge profile endpoints with exact bounded physical-family witnesses."""
import json
from pathlib import Path

import numpy as np

from inference import ModelSpec, parameter_box
from models import equivalent_original, original_transfer
from original_family_bounds import feasible_intervals, parameter_extrema, fixed_value_members


def joint_parameters():
    path = Path(__file__).parent/"results/fits/joint__original_n0_physics_cold_linear_exact.json"
    result = json.loads(path.read_text())["fit"]["parameters"]
    return np.array([result[name] for name in ("Ua", "Ub", "CpH", "CpS")])


def test_bounded_family_extrema_contain_independent_dense_feasible_members():
    p = joint_parameters()
    lower, upper = parameter_box(ModelSpec(family="original", order=0, coordinates="physics"))
    K, a, b = original_transfer(p)
    intervals = feasible_intervals(p)
    assert len(intervals) == 1
    dense = np.array([equivalent_original(K, a, b, r) for left, right in intervals for r in np.linspace(left+1e-8, right-1e-8, 1001)])
    assert np.all(dense >= lower[None, :]-1e-10) and np.all(dense <= upper[None, :]+1e-10)
    for j in (1, 2, 3):
        extrema = parameter_extrema(p, j)
        assert extrema["minimum"]["value"] <= dense[:, j].min()+1e-10
        assert extrema["maximum"]["value"] >= dense[:, j].max()-1e-10
        for witness in extrema["witnesses"]:
            assert np.allclose(original_transfer(witness["parameters"]), [K, a, b], atol=2e-10, rtol=1e-12)
        value = (extrema["minimum"]["value"]+extrema["maximum"]["value"])/2
        members = fixed_value_members(p, j, value)
        assert members and all(abs(member[j]-value) < 1e-10 for member in members)
        assert fixed_value_members(p, j, extrema["maximum"]["value"])


def test_rectangular_constraints_can_split_one_continuum_into_two_components():
    p = joint_parameters()
    lower, upper = parameter_box(ModelSpec(family="original", order=0, coordinates="physics"))
    upper[3] = .5*parameter_extrema(p, 3)["maximum"]["value"]
    intervals = feasible_intervals(p, lower, upper)
    assert len(intervals) == 2
    K, a, b = original_transfer(p)
    assert intervals[0][1] < np.sqrt(b) < intervals[1][0]
    for left, right in intervals:
        member = equivalent_original(K, a, b, (left+right)/2)
        assert np.all(member >= lower) and np.all(member <= upper)
