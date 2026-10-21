"""Independent nuisance audit must detect geometry/covariance corruption."""
from copy import deepcopy

import numpy as np
import pytest

from audit_sensitivity import direct_mean, audit_nuisance_geometry
from data import Experiment
from inference import ModelSpec
from nuisance import prediction, layout, residual_and_jacobian


def fixture(constant=False):
    t = np.linspace(0., 120., 121)
    d = Experiment("audit", t, 25+np.sin(t/11),
                   np.full(len(t), 50.) if constant else 45+20*np.cos(t/15), 24.5)
    p, spec = np.array([.8, 25., 4.]), ModelSpec(order=2)
    q = np.r_[np.log(p), .4, .3]
    names, lower, upper = layout(spec, 1, True, True)
    residual, J = residual_and_jacobian(q, [d], spec, True, True)
    _, s, V = np.linalg.svd(J, full_matrices=False)
    keep = s > 1e-8*s[0]
    bread = (V[keep].T/s[keep]**2)@V[keep]
    full = bool(np.all(keep))
    fit = {"mixed_coordinates":q.tolist(),"ambient_offsets_C":[.4],"history_fractions":[.3],
           "free_ambient":True,"free_history":True,"bounds_mixed":[lower.tolist(),upper.tolist()],
           "singular_values_mixed_metric":s.tolist(),"right_singular_vectors_rows":V.tolist(),
           "rank_relative_1e-8":int(keep.sum()),"sse":float(residual@residual),"n_conditional":len(residual),
           "mixed_coordinate_names":names,
           "cov_mixed_iid_unknown_sigma":(bread*(residual@residual)/(len(residual)-len(q))).tolist() if full else None,
           "cov_mixed_known_sigma_0p25":(bread*.25**2).tolist() if full else None,
           "nonestimable_mixed_coordinates":[name for name, flag in zip(names,np.sum(V[~keep]**2,axis=0)>1e-10) if flag]}
    return fit, [d], spec, p


@pytest.mark.parametrize("spec",[ModelSpec(order=12),ModelSpec(order=0,family="pure_delay")])
def test_direct_nuisance_state_mean_agrees_with_producer(spec):
    _, datasets, _, p = fixture()
    d = datasets[0]
    assert np.max(abs(direct_mean(d,p,spec,.4,.3)-prediction(d,p,spec,.4,.3))) < 2e-12


def test_finite_difference_audit_accepts_regular_geometry_and_detects_changed_covariance():
    fit, datasets, spec, p = fixture()
    gram, cov = audit_nuisance_geometry(fit,datasets,spec,p)
    assert gram < 1e-8 and cov < 1e-12
    corrupt = deepcopy(fit)
    corrupt["cov_mixed_known_sigma_0p25"][0][0] *= 1.1
    with pytest.raises(RuntimeError,match="covariance arithmetic"):
        audit_nuisance_geometry(corrupt,datasets,spec,p)


def test_exact_constant_input_gauge_has_undefined_full_covariance():
    fit, datasets, spec, p = fixture(constant=True)
    assert fit["rank_relative_1e-8"] == 4
    audit_nuisance_geometry(fit,datasets,spec,p)
    fit["cov_mixed_known_sigma_0p25"] = np.eye(5).tolist()
    with pytest.raises(RuntimeError,match="rank-deficient full covariance"):
        audit_nuisance_geometry(fit,datasets,spec,p)
