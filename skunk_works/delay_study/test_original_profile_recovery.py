"""Regression for the actual Ua nuisance-profile failure and chart equivalence."""
import json
from pathlib import Path

import numpy as np

from data import load
from original_profile_recovery import chart_from_physical, physical_from_chart, solve_fixed_Ua


def test_physical_chart_roundtrip_preserves_parameters_and_positivity():
    parameters = np.array([.04, .014, 6.5, .3])
    chart = chart_from_physical(parameters)
    assert np.allclose(physical_from_chart(chart, parameters[0]), parameters, rtol=1e-11)
    assert np.all(physical_from_chart(chart, parameters[0]) > 0)


def test_observable_chart_resolves_previously_failed_joint_Ua_point():
    path = Path(__file__).parent/"results/fits/joint__original_n0_physics_cold_linear_exact.json"
    record = json.loads(path.read_text())
    parameters = np.array([record["fit"]["parameters"][name] for name in ["Ua", "Ub", "CpH", "CpS"]])
    result = solve_fixed_Ua(.053508896090370586, parameters, [load("sine"), load("step")], maxiter=600)
    assert result["success"]
    assert result["maximum_log_box_violation"] < 1e-7
    assert result["sse_representation_difference"] < 1e-6
