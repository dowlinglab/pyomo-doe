"""Independent precision calculation and corrupted saved-artifact checks."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from audit_design import DirectStudy
from audit_discrimination import (audit_design_record, audit_refit_record,
                                  declared_tasks, direct_distance, sha)
from data import Experiment
from inference import ModelSpec


def test_discrimination_precision_matches_dense_irregular_gaussian_covariance():
    time = np.array([0., 1., 2., 5., 11., 14.])
    a = np.r_[25., 25+np.sin(time[1:])]
    b = np.r_[100., np.full(5, 25.)]  # The conditioned first observation is omitted.
    for length in [None, 10.]:
        R = .25**2*(np.eye(5) if length is None else np.exp(-abs(time[1:, None]-time[None, 1:])/length))
        delta = (a-b)[1:]
        assert abs(direct_distance(a, b, time, length, .25)-delta@np.linalg.solve(R, delta)) < 1e-10


@pytest.fixture
def short_study(tmp_path, monkeypatch):
    import audit_design as audit
    import run_design as preparation
    import run_discrimination as producer
    here = Path(__file__).parent
    dc = json.loads((here/"design_config.json").read_text())
    dc.update(horizon_s=120, observation_interval_s=10, knot_interval_s=30,
              audit_interval_s=2, audit_tail_s=100, maximum_temperature_C=200,
              ensemble_orders=[1, 2], noise_lengths_s=[None, 10], maximum_iterations=100)
    config = json.loads((here/"discrimination_config.json").read_text())
    config.update(orders=[1, 2], noise_lengths_s=[None, 10], refit_alternative_orders=[2], refit_starts=3)
    time = np.arange(0., 121., 10.)
    sine = Experiment("sine", time, np.full(len(time), 25.), 50+20*np.sin(time/15), 25.)
    step = Experiment("step", time, np.full(len(time), 25.), np.full(len(time), 50.), 25.)
    fake_load = lambda name: {"sine": sine, "step": step}[name]
    monkeypatch.setattr(preparation, "load", fake_load)
    monkeypatch.setattr(audit, "load", fake_load)
    fits = tmp_path/"fits"
    fits.mkdir()
    for order in dc["ensemble_orders"]:
        for initialization in dc["initializations"]:
            spec = ModelSpec(order=order, initialization=initialization)
            parameters = [.8, 25., 4.] if order == 1 else [.78, 35., 6.]
            (fits/("joint__"+spec.identifier+".json")).write_text(json.dumps({"datasets": ["sine", "step"],
                "fit": {"parameters": dict(zip(spec.names, parameters)), "spec": asdict(spec)}}))
    scenarios, _, knots, _ = preparation.prepare(dc, fits)
    basis = preparation.interpolation_basis(time, knots)
    output = tmp_path/"records"
    output.mkdir()
    monkeypatch.setattr(producer, "STATE", (config, dc, scenarios, knots, time, basis, str(output)))
    return producer, DirectStudy(dc, fits), config, output


def test_saved_fixed_mean_design_is_independently_checked(short_study):
    producer, study, config, output = short_study
    task = ("cold__ou10__mean", "cold", 10, "mean")
    _, _, revisit = producer.execute(task, "designs", output, {"fixture": True})
    assert not revisit
    record = json.loads((output/(task[0]+".json")).read_text())
    summary = audit_design_record(record, task, study, config, {})
    assert summary["successful_feasible_starts"] > 0
    corrupted = deepcopy(record)
    corrupted["evaluations"]["selected_local_design"]["pairs"][0]["distance_squared"] *= 1.1
    with pytest.raises(RuntimeError, match="fixed_distance_squared"):
        audit_design_record(corrupted, task, study, config, {})
    corrupted = deepcopy(record)
    corrupted["evaluations"]["constant"]["pairs"][0]["ideal_fixed_gaussian_equal_prior_error"] = .5
    with pytest.raises(RuntimeError, match="ideal_gaussian_error"):
        audit_design_record(corrupted, task, study, config, {})
    corrupted = deepcopy(record)
    corrupted["optimization"]["attempts"][0]["iterations"] = study.config["maximum_iterations"]+1
    with pytest.raises(RuntimeError, match="iteration budget"):
        audit_design_record(corrupted, task, study, config, {})


@pytest.mark.parametrize("initialization,waveform", [("cold", "multisine"), ("input_equilibrated", "constant")])
def test_saved_alternative_refits_are_checked_including_exact_null_case(short_study, initialization, waveform):
    producer, study, config, output = short_study
    task = (initialization+"__ou10__"+waveform+"__alternative2", initialization, 10, waveform, 2)
    _, _, revisit = producer.execute(task, "refits", output, {"fixture": True})
    assert not revisit
    record = json.loads((output/(task[0]+".json")).read_text())
    summary = audit_refit_record(record, task, study, config, {}, output)
    assert summary["closest_found_distance_squared_upper_bound"] <= summary["fixed_distance_squared"]
    if initialization == "input_equilibrated":
        assert summary["fixed_distance_squared"] > 1
        assert summary["closest_found_distance_squared_upper_bound"] < 1e-20
    corrupted = deepcopy(record)
    corrupted["refit"]["attempts"][0]["distance_squared"] += 1.
    with pytest.raises(RuntimeError, match="refitted_distance_squared"):
        audit_refit_record(corrupted, task, study, config, {}, output)
    corrupted = deepcopy(record)
    corrupted["refit"]["attempts"][0]["active_bounds"][0] = True
    with pytest.raises(RuntimeError, match="active-bound"):
        audit_refit_record(corrupted, task, study, config, {}, output)


def test_independent_task_inventory_matches_declared_full_matrix():
    config = json.loads((Path(__file__).parent/"discrimination_config.json").read_text())
    assert len(set(declared_tasks(config, "designs"))) == 12
    assert len(set(declared_tasks(config, "refits"))) == 72


def test_selected_refit_binds_the_actual_source_waveform(short_study):
    producer, study, config, output = short_study
    design_task = ("cold__ou10__mean", "cold", 10, "mean")
    assert not producer.execute(design_task, "designs", output, {"fixture": True})[2]
    source = output/(design_task[0]+".json")
    identity = {"fixture": True, "source_design_sha256": {source.name: sha(source)}}
    task = ("cold__ou10__selected_mean__alternative2", "cold", 10, "selected_mean", 2)
    assert not producer.execute(task, "refits", output, identity)[2]
    record = json.loads((output/(task[0]+".json")).read_text())
    audit_refit_record(record, task, study, config, {}, output)
    changed = json.loads(source.read_text())
    changed["evaluations"]["selected_local_design"]["amplitudes_percent"][0] += 1.
    source.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="source design identity"):
        audit_refit_record(record, task, study, config, {}, output)
