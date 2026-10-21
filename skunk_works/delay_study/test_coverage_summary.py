"""Failure accounting and interpolation diagnostics have different meanings."""
import numpy as np
import pytest

from summarize_coverage import coverage_counts, quadratic_grid_statistic, verify_chunk_counts


def test_failed_or_undefined_intervals_remain_in_unconditional_denominator():
    counts = coverage_counts(8, 9, 12)
    assert counts["coverage_when_defined"] == 8/9
    assert counts["unconditional_coverage_lower"] == 8/12
    assert counts["unconditional_coverage_upper"] == 11/12
    assert counts["undefined_or_failed"] == 3
    assert counts["conservative_wilson_lower"] < 8/12
    assert counts["conservative_wilson_upper"] > 11/12


def test_all_failed_run_does_not_return_a_conditional_coverage_estimate():
    counts = coverage_counts(0, 0, 300)
    assert counts["coverage_when_defined"] is None
    assert counts["wilson_when_defined_lower"] is None
    assert counts["unconditional_coverage_lower"] == 0
    assert counts["unconditional_coverage_upper"] == 1


def test_quadratic_diagnostic_exposes_linear_grid_classification_error():
    x = np.array([-2., 0., 2.])
    y = x*x
    truth = 1.9
    assert np.interp(truth, x, y) > 3.7
    assert abs(quadratic_grid_statistic(x, y, truth)-truth**2) < 1e-12
    assert quadratic_grid_statistic(x, y, truth) < 3.7
    assert quadratic_grid_statistic(x, y, 3.) is None


def test_file_loss_cannot_be_reclassified_as_an_optimizer_failure():
    draw = {"replicate": 0, "status": "complete"}
    manifest = {"completed": {"case__iid__0000_0001": {"complete": 2, "failed": 0}}, "failed": {}}
    with pytest.raises(RuntimeError, match="saved draw files"):
        verify_chunk_counts([draw], "case", "iid", 2, 2, manifest)
    manifest = {"completed": {}, "failed": {"case__iid__0000_0001": "worker exception"}}
    verify_chunk_counts([draw], "case", "iid", 2, 2, manifest)
