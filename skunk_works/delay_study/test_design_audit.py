"""Independent design audit: analytic targets, dense covariance and artifacts."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from audit_design import (DirectStudy,audit_record,command_resources,dense_command,
                          direct_scores,scenario_label,weighted_gram)
from data import Experiment
from inference import ModelSpec


def test_direct_design_scores_match_analytic_targets_and_reject_null_delay():
    F,A = np.diag([4.,9.,16.]),np.diag([1.,2.,3.])
    scores = direct_scores(F,A)
    assert abs(scores["D"]-np.log(4*9*16)) < 1e-12
    assert abs(scores["E"]-np.log(4)) < 1e-12
    assert abs(scores["theta"]-np.log(16)) < 1e-12
    assert abs(scores["prediction"]+np.log(1/4+2/9+3/16)) < 1e-12
    singular = direct_scores(np.diag([4.,9.,0.]),np.diag([1.,2.,0.]))
    assert singular["D"] is singular["E"] is singular["theta"] is None
    assert abs(singular["prediction"]+np.log(1/4+2/9)) < 1e-12
    assert direct_scores(np.diag([4.,9.,0.]),np.eye(3))["prediction"] is None


def test_design_audit_gram_matches_dense_irregular_ou_precision():
    t = np.array([1.,2.,5.,10.,11.])
    J = np.column_stack([np.ones(5),t,np.sin(t)])
    R = .25**2*np.exp(-abs(t[:,None]-t[None,:])/10.)
    assert np.allclose(weighted_gram(J,t,10.,.25),J.T@np.linalg.solve(R,J),rtol=1e-12,atol=1e-10)


def test_resource_audit_integrates_irregular_knots_and_checks_shutdown_ramp():
    amplitudes,knots = np.array([20.,80.,20.]),np.array([0.,2.,6.])
    resources = command_resources(amplitudes,knots)
    assert resources["mean_input_percent"] == 50.
    assert resources["maximum_slew_percent_per_s"] == 30.
    config = {"horizon_s":6.,"audit_tail_s":4.,"audit_interval_s":.25,"maximum_slew_percent_per_s":10.}
    time,command = dense_command(amplitudes,knots,config)
    assert command[time == 7.] == 10.
    assert np.all(command[time >= 8.] == 0.)
    assert np.allclose(command[time <= 6.],np.interp(time[time <= 6.],knots,amplitudes))


def test_saved_synthetic_design_is_audited_and_information_corruption_detected(tmp_path,monkeypatch):
    import audit_design as audit
    import run_design as producer
    config = json.loads((Path(__file__).parent/"design_config.json").read_text())
    config.update(horizon_s=120,observation_interval_s=10,knot_interval_s=30,
                  audit_interval_s=2,audit_tail_s=100,maximum_temperature_C=200,
                  ensemble_orders=[1,0],nominal_orders=[1],noise_lengths_s=[None,10],maximum_iterations=100)
    t = np.arange(0.,121.,10.)
    sine = Experiment("sine",t,np.full(len(t),25.),50+20*np.sin(t/15),25.)
    step = Experiment("step",t,np.full(len(t),25.),np.full(len(t),50.),25.)
    fake_load = lambda name:{"sine":sine,"step":step}[name]
    monkeypatch.setattr(producer,"load",fake_load)
    monkeypatch.setattr(audit,"load",fake_load)
    fits = tmp_path/"fits"
    fits.mkdir()
    for order in config["ensemble_orders"]:
        for initialization in config["initializations"]:
            spec = ModelSpec(family="pure_delay" if order == 0 else "chain",order=order,initialization=initialization)
            (fits/("joint__"+spec.identifier+".json")).write_text(json.dumps({"datasets":["sine","step"],
                "fit":{"parameters":dict(zip(spec.names,[.8,25.,4.])),"spec":asdict(spec)}}))
    scenarios,targets,knots,_ = producer.prepare(config,fits)
    for name,value in [("SCENARIOS",scenarios),("TARGETS",targets),("CONFIG",config),("KNOTS",knots)]:
        monkeypatch.setattr(producer,name,value)
    output = tmp_path/"design"
    output.mkdir()
    task = ("synthetic",[scenario_label(1,"cold",None)],"D",False,"mean")
    _,_,revisit = producer.execute(task,output,{"synthetic_fixture":True})
    assert not revisit
    record = json.loads((output/"synthetic.json").read_text())
    study = DirectStudy(config,fits)
    summary = audit_record(record,task,study,{})
    assert summary["successful_feasible_starts"] > 0
    assert summary["dense_maximum_C"] < 200
    corrupted = deepcopy(record)
    corrupted["evaluations"]["selected_local_design"]["scenario_information"][0]["information_log_io"][0][0] *= 1.1
    with pytest.raises(RuntimeError,match="information_log_io"):
        audit_record(corrupted,task,study,{})
    corrupted = deepcopy(record)
    corrupted["evaluations"]["selected_local_design"]["dense_audit"]["scenarios"][0]["maximum_C"] -= 1.
    with pytest.raises(RuntimeError,match="dense_maximum_C"):
        audit_record(corrupted,task,study,{})


@pytest.mark.parametrize("last", [1e-12, .25+1e-12, .25-1e-12])
def test_nearly_coincident_shutdown_is_deduplicated_at_propagator_resolution(last):
    from types import SimpleNamespace
    from run_design import dense_audit
    config = {"horizon_s": 6., "audit_tail_s": 4., "audit_interval_s": .25,
              "maximum_slew_percent_per_s": 1., "maximum_temperature_C": 200.}
    knots = np.array([0., 3., 6.])
    amplitudes = np.array([20., 20., last])
    time, command = dense_command(amplitudes, knots, config)
    assert np.all(np.round(np.diff(time), 9) > 0.)
    scenario = SimpleNamespace(name="near_grid", parameters=np.array([.8, 25., 4.]),
                               spec=ModelSpec(family="chain", order=1), ambient=25.)
    result = dense_audit([scenario], amplitudes, knots, config)
    rounded = amplitudes.copy()
    rounded[-1] = round(last, 9)
    reference = dense_audit([scenario], rounded, knots, config)
    assert abs(result["scenarios"][0]["maximum_C"]-reference["scenarios"][0]["maximum_C"]) < 1e-9
