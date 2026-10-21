"""Outcome-blind artifact preparation: failed/undefined and disconnected cases."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from audit_coverage_figures import audit_values, direct_statistics, score_interval
from plot_coverage import load_delay_summary


def local_draw(lower=9., upper=11., defined=True):
    interval = {"lower": [1., 1., lower] if defined else None,
                "upper": [1., 1., upper] if defined else None,
                "covered": [True, True, lower <= 10 <= upper] if defined else None}
    return {"status": "complete", "truth_io": [.8, 40., 10.],
            "local_intervals_io": {"iid_known_sigma": interval}}


def test_saved_interval_statistics_retain_all_failure_denominators():
    draws = [local_draw(), local_draw(12., 14.), local_draw(defined=False), {"status": "failed"}]
    row = direct_statistics(draws, "iid_known_sigma", 5)
    assert (row["requested"], row["defined"], row["covered"]) == (5, 2, 1)
    assert (row["completed_draws"], row["failed_draws"], row["missing_draws"]) == (3, 1, 1)
    assert row["undefined_on_completed_draws"] == 1
    assert row["coverage_when_defined"] == .5
    assert row["unconditional_coverage_lower"] == .2 and row["unconditional_coverage_upper"] == .8
    assert row["median_width"] == 2.
    damaged = deepcopy(draws)
    damaged[0]["local_intervals_io"]["iid_known_sigma"]["covered"][2] = False
    with pytest.raises(RuntimeError, match="membership"):
        direct_statistics(damaged, "iid_known_sigma", 5)


def test_score_interval_endpoints_satisfy_independent_score_equations():
    from statistics import NormalDist
    z = NormalDist().inv_cdf(.975)
    for k in [0, 1, 150, 299, 300]:
        lo, hi = score_interval(k, 300)
        assert 0 <= lo <= k/300 <= hi <= 1
        for p in [lo, hi]:
            if 0 < p < 1:
                assert abs((k-300*p)**2/(300*p*(1-p))-z*z) < 1e-10
    assert score_interval(0, 0) == (None, None)


def test_disconnected_profile_widths_and_grid_classification_remain_distinct():
    from statistics import NormalDist
    components = [{"lower": 1., "upper": 2., "lower_status": "open_at_exploration_bound", "upper_status": "crossing"},
                  {"lower": 8., "upper": 9., "lower_status": "crossing", "upper_status": "crossing"}]
    draw = {"status": "complete", "truth_io": [.8, 40., float(np.exp(1.95))],
            "delay_profile": {"intervals_known_sigma_iid": components, "chi2_95_df1": NormalDist().inv_cdf(.975)**2,
                              "points": [{"log_value": x, "lr_known_sigma_iid": x*x} for x in [-2., 0., 2.]]}}
    row = direct_statistics([draw], "profile_known_sigma_iid", 1)
    assert row["covered"] == 0 and row["defined"] == 1
    assert row["median_width"] == 2. and row["mean_envelope_width"] == 8.
    assert row["mean_component_count"] == 2 and row["open_exploration_sets"] == 1
    assert row["quadratic_interpolation_coverage_class_changes"] == 1
    damaged = deepcopy(draw)
    damaged["delay_profile"]["intervals_known_sigma_iid"][1]["lower"] = 1.5
    with pytest.raises(RuntimeError, match="Overlapping"):
        direct_statistics([damaged], "profile_known_sigma_iid", 1)


def test_undefined_coverage_cannot_be_replaced_with_zero_or_finite_width():
    row = direct_statistics([local_draw(defined=False)], "iid_known_sigma", 300)
    assert row["coverage_when_defined"] is row["median_width"] is None
    audit_values(row, row, {})
    changed = dict(row, coverage_when_defined=0.)
    with pytest.raises(RuntimeError, match="Undefined coverage"):
        audit_values(changed, row, {})
    changed = dict(row, defined=1)
    with pytest.raises(RuntimeError, match="defined"):
        audit_values(changed, row, {})


def test_plotting_refuses_live_runs_and_unbound_summary_bytes(tmp_path):
    config = Path(__file__).parent/"coverage_config.json"
    coverage, summary = tmp_path/"coverage", tmp_path/"summary"
    coverage.mkdir()
    summary.mkdir()
    manifest = {"status": "running", "filter": "", "identity": {"replicate_override": None},
                "completed": {"chunk": {}}, "failed": {}, "expected_chunks": 1, "expected_draws": 3600}
    path = coverage/"manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="terminal"):
        load_delay_summary(summary, coverage, config)
    manifest["status"] = "complete"
    path.write_text(json.dumps(manifest))
    (summary/"audit.json").write_text(json.dumps({"status": "passed", "requested_draws": 3600,
        "summary_rows": 384, "input_sha256": {str(path): "unbound-manifest"}}))
    with pytest.raises(RuntimeError, match="arithmetic audit"):
        load_delay_summary(summary, coverage, config)


def test_all_failed_terminal_collection_round_trips_without_fake_coverage(tmp_path, monkeypatch):
    # Every worker chunk failed before writing a draw. This is an explicit
    # negative fixture, not a synthetic successful calibration or real result.
    import sys
    import pandas as pd
    import audit_coverage_figures as audit
    import plot_coverage as plot
    from summarize_coverage import summarize_scenario
    config_path = Path(__file__).parent/"coverage_config.json"
    config = json.loads(config_path.read_text())
    coverage, summary, figures = [tmp_path/name for name in ["coverage", "summary", "figures"]]
    coverage.mkdir()
    summary.mkdir()
    failures = {}
    for case in config["cases"]:
        for noise in config["noise_scenarios"]:
            for first in range(0, 300, 10):
                failures[case+"__"+noise+f"__{first:04d}_{first+9:04d}"] = "Explicit all-failed artifact fixture"
    manifest = {"status": "complete_with_failures", "filter": "",
                "identity": {"replicate_override": None, "config_sha256": plot.sha(config_path)},
                "completed": {}, "failed": failures, "expected_chunks": 360, "expected_draws": 3600}
    manifest_path = coverage/"manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    rows = [row for case in config["cases"] for noise in config["noise_scenarios"]
            for row in summarize_scenario([], case, noise, 300)]
    csv_path = summary/"coverage_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    (summary/"audit.json").write_text(json.dumps({"status": "passed", "requested_draws": 3600,
        "summary_rows": 384, "summary_sha256": plot.sha(csv_path),
        "input_sha256": {str(manifest_path): plot.sha(manifest_path), str(config_path): plot.sha(config_path)}}))
    alias = tmp_path/"byte_identical_config.json"
    alias.write_bytes(config_path.read_bytes())
    assert len(plot.load_delay_summary(summary, coverage, alias)[0]) == 84
    alias.write_text(alias.read_text().replace('"replicates": 300', '"replicates": 299'))
    with pytest.raises(RuntimeError, match="terminal"):
        plot.load_delay_summary(summary, coverage, alias)
    monkeypatch.setattr(plot, "figures", lambda *args: ["coverage_delay_coverage", "coverage_delay_width"])
    arguments = ["--coverage", str(coverage), "--summary", str(summary), "--config", str(config_path)]
    monkeypatch.setattr(sys, "argv", ["plot_coverage.py", *arguments, "--output", str(figures)])
    plot.main()
    data_path = figures/"coverage_delay_data.json"
    data = json.loads(data_path.read_text())
    assert len(data["rows"]) == 84
    assert all(r["coverage_when_defined"] is None and r["missing_draws"] == 300 for r in data["rows"])
    monkeypatch.setattr(sys, "argv", ["audit_coverage_figures.py", *arguments, "--figures", str(figures)])
    audit.main()
    report = json.loads((figures/"coverage_numeric_audit.json").read_text())
    assert report["status"] == "passed" and report["delay_rows"] == 84
    # Bind the edited data bytes in the figure manifest so the independent
    # statistic check, rather than the digest check, must catch the error.
    data["rows"][0]["coverage_when_defined"] = 0.
    data_path.write_text(json.dumps(data))
    figure_manifest = figures/"coverage_figure_manifest.json"
    values = json.loads(figure_manifest.read_text())
    values["plot_data_sha256"] = plot.sha(data_path)
    figure_manifest.write_text(json.dumps(values))
    with pytest.raises(RuntimeError, match="Undefined coverage"):
        audit.main()


def test_shared_width_axes_include_later_noise_panels(tmp_path, monkeypatch):
    import plot_coverage as plot
    from summarize_coverage import coverage_counts
    rows = [dict(case=case, true_noise=noise, method=method,
                 median_width=1.+100*ni+mi+j, **coverage_counts(285, 300, 300))
            for ni, (noise, _) in enumerate(plot.NOISES)
            for mi, (case, *_) in enumerate(plot.MODELS) for j, method in enumerate(plot.METHODS)]
    inspected = []
    def check(figure, name, **kwargs):
        limits = [ax.get_ylim() for ax in figure.axes]
        assert limits == [limits[0]]*3
        for ax in figure.axes:
            for line in ax.lines:
                values = np.asarray(line.get_ydata(), dtype=float)
                assert np.all(values >= ax.get_ylim()[0]) and np.all(values <= ax.get_ylim()[1])
        inspected.append(str(name))
    monkeypatch.setattr(plot.fs, "save_fig", check)
    monkeypatch.setattr(plot.fs, "grayscale_preview", lambda *args: None)
    plot.figures(rows, tmp_path, {})
    assert len(inspected) == 2
