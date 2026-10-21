"""Check the declared sensitivity matrix and its baseline authorities."""
import json
from pathlib import Path

from run_sensitivity import task_matrix, baseline_path


def test_sensitivity_matrix_scope_and_meaningful_numerical_methods():
    config = json.loads((Path(__file__).parent/"sensitivity_config.json").read_text())
    tasks = task_matrix(config)
    assert len(tasks) == 241
    assert len({task["key"] for task in tasks}) == 241
    assert sum(task["phase"] == "reconstruction" for task in tasks) == 49
    assert sum(task["phase"] == "nuisance" for task in tasks) == 192
    assert not any(task["spec"]["family"] == "pure_delay" and task["spec"]["method"] != "exact" for task in tasks)


def test_each_sensitivity_job_points_to_the_correct_completed_baseline():
    root = Path(__file__).parent
    config = json.loads((root/"sensitivity_config.json").read_text())
    for task in task_matrix(config):
        path = baseline_path(task, root/"results/fits")
        record = json.loads(path.read_text())
        assert record["datasets"] == task["datasets"]
        assert record["fit"]["spec"]["family"] == task["spec"]["family"]
        if task.get("mode") == "ambient_equilibrated":
            assert record["fit"]["spec"]["initialization"] == "input_equilibrated"
