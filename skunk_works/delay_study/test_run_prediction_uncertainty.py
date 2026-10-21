"""Check interval study scope and exclusion of conditioned/undefined targets."""
import json
from pathlib import Path

import numpy as np

from data import Experiment
from prediction_uncertainty import normal_bands
from run_prediction_uncertainty import tasks, frame_for, descriptive_summary


def test_prediction_job_matrix_includes_original_quotient_and_all_joint_models():
    root = Path(__file__).parent
    config = json.loads((root/"prediction_config.json").read_text())
    jobs = tasks(root/"results/fits", config)
    assert len(jobs) == len({job[0] for job in jobs}) == 131
    assert sum(job[1] == "local" for job in jobs) == 51
    assert sum(job[1] == "ou" for job in jobs) == 48
    assert sum(job[1] == "bootstrap" for job in jobs) == 32
    assert sum("original" in job[0] for job in jobs) == 3


def test_conditioned_initial_temperature_and_undefined_targets_do_not_get_intervals():
    data = Experiment("target", np.arange(3.), np.full(3, 25.), np.full(3, 50.), 25.)
    mean = np.full(3, 25.)
    variance = np.array([.1, np.nan])
    bands = normal_bands(mean[1:], variance, .25**2)
    frame = frame_for(data, "validation", "test", mean, variance, bands, [True, False])
    assert np.isnan(frame.loc[0, "mean_lower_C"]) and np.isnan(frame.loc[0, "prediction_lower_C"])
    assert np.isnan(frame.loc[2, "mean_upper_C"]) and not frame.loc[2, "estimable"]
    summary = descriptive_summary(frame)
    assert summary["conditional_points"] == 2 and summary["estimable_points"] == 1
