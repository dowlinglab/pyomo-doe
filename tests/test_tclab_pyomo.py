from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_two_state_data(module):
    return module.TC_Lab_data(
        name="pytest smoke",
        time=np.array([0.0, 60.0, 120.0]),
        T1=np.array([23.0, 23.4, 23.8]),
        u1=np.array([0.0, 50.0, 75.0]),
        P1=200.0,
        TS1_data=np.array([23.0, 23.4, 23.8]),
        T2=None,
        u2=None,
        P2=200.0,
        TS2_data=None,
        Tamb=23.0,
    )


def test_tc_lab_experiment_builds_active_two_state_model(tclab_pyomo_module):
    data = make_two_state_data(tclab_pyomo_module)

    experiment = tclab_pyomo_module.TC_Lab_experiment(data=data)
    model = experiment.get_labeled_model()

    assert model.experiment_inputs[model.U1[data.time[0]]] is None
    assert model.measurement_error[model.Ts1[data.time[0]]] == pytest.approx(0.25)

    extracted = tclab_pyomo_module.extract_results(model)
    assert np.allclose(extracted.time, data.time)
    assert np.allclose(extracted.u1, data.u1)
    assert extracted.TS2_data is None


def test_parameter_round_trip_helpers(tclab_pyomo_module):
    alpha = 0.00016
    p1 = 200.0
    original = {
        "Ua": 0.0535,
        "Ub": 0.0148,
        "inv_CpH": 1 / 6.911,
        "inv_CpS": 1 / 0.318,
    }

    reformulated = tclab_pyomo_module.reformulate_parameters(original, alpha, p1)
    recovered = tclab_pyomo_module.recover_original_parameters(
        reformulated, alpha, p1
    )

    for key, value in original.items():
        assert recovered[key] == pytest.approx(value)


def test_recover_original_covariance_shape(tclab_pyomo_module):
    alpha = 0.00016
    p1 = 200.0
    reformulated = {
        "beta_1": 0.1,
        "beta_2": 0.2,
        "beta_3": 0.3,
        "beta_4": 0.4,
    }
    cov_reform = pd.DataFrame(
        np.eye(4),
        index=["beta_1", "beta_2", "beta_3", "beta_4"],
        columns=["beta_1", "beta_2", "beta_3", "beta_4"],
    )

    cov_orig = tclab_pyomo_module.recover_original_covariance(
        reformulated, cov_reform, alpha, p1
    )

    assert list(cov_orig.index) == ["Ua", "Ub", "Inv_CpH", "inv_CpS"]
    assert list(cov_orig.columns) == ["Ua", "Ub", "Inv_CpH", "inv_CpS"]
    assert cov_orig.shape == (4, 4)
