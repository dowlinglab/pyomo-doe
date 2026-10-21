"""Check independent audit equations and deliberate band corruption."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm

from audit_prediction_uncertainty import (direct_local_covariances, direct_ou_information,
                                          audit_group)
from data import Experiment
from inference import ModelSpec, sensitivities


def test_audit_ou_whitening_matches_dense_irregular_covariance():
    time = np.array([0.,3.,9.,22.,48.,80.])
    data = Experiment("dense",time,np.full(len(time),25.),[50,75,25,80,20,50],25.)
    parameters,spec = [.8,25.,4.],ModelSpec(order=2)
    J = sensitivities(parameters,[data],spec)
    length,sigma = 10.,.25
    R = sigma**2*np.exp(-abs(time[1:,None]-time[None,1:])/length)
    expected = np.linalg.inv(J.T@np.linalg.solve(R,J))
    covariance,null,rank = direct_ou_information(parameters,[data],spec,length,sigma)
    assert rank == 3 and null.shape == (3,0)
    assert np.allclose(covariance,expected,rtol=1e-9,atol=1e-10)


def test_audit_hac_resets_records_and_uses_quotient_degrees_of_freedom():
    x = np.arange(7.)
    J = np.column_stack([np.ones(7),x,np.ones(7)])
    residual = np.array([1.,-.3,.5,-1.,.4,.7,-.2])
    matrices,null,rank,dof,variance = direct_local_covariances(J,residual,[3,4],.25,[2])
    bread = np.linalg.pinv(J.T@J)
    meat = np.zeros((3,3))
    for indices in [range(3),range(3,7)]:
        for i in indices:
            for j in indices:
                lag = abs(i-j)
                if lag <= 2:
                    meat += (1-lag/3)*np.outer(J[i]*residual[i],J[j]*residual[j])
    assert rank == 2 and dof == 5 and null.shape == (3,1)
    assert variance == residual@residual/5
    assert np.allclose(matrices["hac2_quotient_dof"],7/5*bread@meat@bread,atol=1e-12)
    assert np.allclose(matrices["iid_unknown_quotient_dof"],bread*variance,atol=1e-12)


def band_fixture():
    t = np.arange(5.)
    mean = 25+t/10
    observed = mean+.01
    observed[0] = mean[0]
    data = Experiment("bands",t,observed,np.full(5,50.),25.)
    variance = np.array([.1,.2,.3,.4])
    z = norm.ppf(.975)
    frame = pd.DataFrame({"role":"validation","experiment":data.name,"method":"iid_known_sigma",
                          "time_s":t,"input_percent":data.input,"measured_C":data.temperature,
                          "mean_C":mean,"mean_variance_C2":np.r_[np.nan,variance],
                          "estimable":[False,True,True,True,True],
                          "first_temperature_conditions_state":[True,False,False,False,False],
                          "mean_lower_C":np.r_[np.nan,mean[1:]-z*np.sqrt(variance)],
                          "mean_upper_C":np.r_[np.nan,mean[1:]+z*np.sqrt(variance)],
                          "prediction_lower_C":np.r_[np.nan,mean[1:]-z*np.sqrt(variance+.25**2)],
                          "prediction_upper_C":np.r_[np.nan,mean[1:]+z*np.sqrt(variance+.25**2)]})
    return frame,data,mean,variance


def test_band_audit_detects_conditioning_and_undefined_value_corruption():
    frame,data,mean,variance = band_fixture()
    kwargs = dict(frame=frame,data=data,role="validation",method="iid_known_sigma",mean=mean,
                  variance=variance,allowed=np.ones(4,dtype=bool),errors={},future_variance=.25**2)
    summary = audit_group(**kwargs)
    assert summary["estimable_points"] == 4
    frame.loc[0,"estimable"] = True
    with pytest.raises(RuntimeError,match="estimability"):
        audit_group(**kwargs)
    frame.loc[0,"estimable"] = False
    frame.loc[2,"mean_lower_C"] = np.nan
    with pytest.raises(RuntimeError,match="undefined-value"):
        audit_group(**kwargs)


def test_mixture_endpoint_audit_checks_cdf_instead_of_reusing_quantile_helper():
    frame,data,mean,_ = band_fixture()
    means = mean[None,1:]+np.array([-1.,1.])[:,None]
    sigma = .25
    endpoints = np.array([[brentq(lambda y:ndtr((y-means[:,j])/sigma).mean()-q,
                                  means[:,j].min()-3,means[:,j].max()+3)
                           for j in range(means.shape[1])] for q in [.025,.975]])
    frame["method"] = "bootstrap_parametric"
    frame["mean_variance_C2"] = np.r_[np.nan,np.var(means,axis=0,ddof=1)]
    frame["mean_lower_C"],frame["mean_upper_C"] = np.r_[np.nan,means[0]],np.r_[np.nan,means[1]]
    frame["bootstrap_mean_median_C"] = np.r_[np.nan,mean[1:]]
    frame["prediction_lower_C"],frame["prediction_upper_C"] = np.r_[np.nan,endpoints[0]],np.r_[np.nan,endpoints[1]]
    kwargs = dict(frame=frame,data=data,role="validation",method="bootstrap_parametric",mean=mean,
                  variance=np.var(means,axis=0,ddof=1),allowed=np.ones(4,dtype=bool),errors={},
                  lower=means[0],upper=means[1],median=mean[1:],mixture_means=means,mixture_sigma=sigma)
    audit_group(**kwargs)
    frame.loc[1,"prediction_lower_C"] += .02
    with pytest.raises(RuntimeError,match="mixture_cdf"):
        audit_group(**kwargs)


@pytest.mark.parametrize("phase,method",[("local",None),("ou",None),("bootstrap","parametric"),("bootstrap","block60")])
def test_saved_trial_artifact_passes_end_to_end_audit(tmp_path,monkeypatch,phase,method):
    # A short synthetic producer/auditor round trip exercises real file/schema
    # handling without starting a pool or refitting a model.
    import audit_prediction_uncertainty as audit
    import run_prediction_uncertainty as producer
    from inference import predict
    study = Path(__file__).parent
    config = json.loads((study/"prediction_config.json").read_text())
    identity = {"source_sha256":{name:audit.sha(study/name) for name in
        ["data.py","models.py","inference.py","noise.py","prediction_uncertainty.py","run_prediction_uncertainty.py","run_fits.py"]},
        "config_sha256":audit.sha(study/"prediction_config.json")}
    t = np.array([0.,3.,9.,22.,48.,80.])
    calibration = Experiment("sine",t,[25,26,27,28,29,30],[50,75,25,80,20,50],25.)
    step = Experiment("step",t,[25,26,27,28,29,30],np.full(len(t),50.),25.)
    validation = Experiment("heldout",t,[25,26,27,28,29,30],[50,20,80,25,75,50],25.)
    spec,parameters = ModelSpec(order=2),np.array([.8,25.,4.])
    datasets = [calibration] if phase in {"local","ou"} else [calibration,step]
    residual = np.concatenate([(predict(d,parameters,spec)-d.temperature)[1:] for d in datasets])
    record = {"calibration":"sine" if phase in {"local","ou"} else "joint","datasets":[d.name for d in datasets],"validation":[{"dataset":"heldout"}],
              "fit":{"spec":asdict(spec),"parameters":dict(zip(spec.names,parameters.tolist())),"sse":float(residual@residual)},
              "uncertainty":{"nonestimable_native_parameters":[]}}
    fits = tmp_path/"fits"
    fits.mkdir()
    (fits/"fixture.json").write_text(json.dumps(record))
    output = tmp_path/"prediction"
    output.mkdir()
    fake_load = lambda name:{"sine":calibration,"step":step,"heldout":validation}[name]
    monkeypatch.setattr(producer,"load",fake_load)
    monkeypatch.setattr(audit,"load",fake_load)
    key = phase+"__fixture" if phase != "bootstrap" else "bootstrap__fixture__"+method
    if phase == "ou":
        ou = tmp_path/"ou"
        ou.mkdir()
        (ou/"fixture.json").write_text(json.dumps({"fit_sha256":audit.sha(fits/"fixture.json"),
            "fit":{"parameters":dict(zip(spec.names,parameters.tolist())),"correlation_length_s":10.,
                   "sigma_stationary_ML_C":.25,"active_bounds_log":[0,0,0,0]},
            "uncertainty":{"observed_cov_log_mean_and_length":(.01*np.eye(4)).tolist(),
                           "hessian_relative_step_disagreement":0.}}))
    if phase == "bootstrap":
        bootstrap = tmp_path/"bootstrap"
        bootstrap.mkdir()
        arrays = bootstrap/("fixture__"+method+".npz")
        np.savez(arrays,parameters=np.tile(parameters,(400,1)),completed=np.ones(400,dtype=bool),success=np.ones(400,dtype=bool))
        (bootstrap/("fixture__"+method+".json")).write_text(json.dumps({"fit_sha256":audit.sha(fits/"fixture.json"),"arrays_sha256":audit.sha(arrays)}))
    producer.execute((key,phase,"fixture.json",method),config,fits,tmp_path/"ou",tmp_path/"bootstrap",output,identity)
    manifest = {"identity":identity,"status":"complete","failed":{},"filter":key,
                "expected_jobs":1,"completed":{key:{"status":"complete"}}}
    (output/"manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(sys,"argv",["audit_prediction_uncertainty.py","--results",str(tmp_path),
                                   "--prediction",str(output),"--allow-trial"])
    audit.main()
    report = json.loads((output/"audit.json").read_text())
    assert report["status"] == "passed_trial" and report["cases"] == 1
    expected_groups = {"local":10,"ou":4,"bootstrap":3}[phase]
    assert report["target_groups"] == expected_groups
    assert report["saved_points_checked"] == 6*expected_groups
