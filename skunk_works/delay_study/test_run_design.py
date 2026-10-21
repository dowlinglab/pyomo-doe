"""Check design study scope and an independent shutdown-temperature audit."""
import json
from pathlib import Path

import numpy as np

from design import build_scenario, interpolation_basis
from inference import ModelSpec
from run_design import dense_audit, tasks


def test_design_matrix_retains_all_nominal_and_ensemble_cases():
    config = json.loads((Path(__file__).parent/"design_config.json").read_text())
    jobs = tasks(config)
    assert len(jobs) == 112
    assert len({row[0] for row in jobs}) == 112
    assert sum(len(row[1]) == 48 for row in jobs) == 16
    assert sum(len(row[1]) == 1 for row in jobs) == 96


def test_dense_audit_agrees_with_independent_one_stage_step_and_checks_tail():
    # With constant input until 60 s, the exact two-pole step response is known.
    time = np.arange(61.)
    knots = np.array([0., 30., 60.])
    parameters = np.array([.78, 150., 18.])
    scenario = build_scenario("analytic", parameters, ModelSpec(order=1), time,
                              interpolation_basis(time, knots))
    config = {"horizon_s": 60, "audit_tail_s": 300, "audit_interval_s": .25,
              "maximum_slew_percent_per_s": 1., "maximum_temperature_C": 85.}
    audit = dense_audit([scenario], np.full(3, 50.), knots, config)
    row = audit["scenarios"][0]
    K, tau, theta = parameters
    analytic_at_horizon = 25+K*50*(1-(tau*np.exp(-60/tau)-theta*np.exp(-60/theta))/(tau-theta))
    assert row["maximum_C"] >= analytic_at_horizon-1e-10
    assert row["time_of_maximum_s"] > 60  # thermal lag during shutdown matters
    assert not row["exceeds_assumed_limit"]
    config["maximum_temperature_C"] = analytic_at_horizon-.01
    assert dense_audit([scenario], np.full(3, 50.), knots, config)["scenarios"][0]["exceeds_assumed_limit"]
