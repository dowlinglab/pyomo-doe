"""Prevent stale manifests or partial counts from launching follow-up pools."""
import json
import pytest

from wait_then_trials import dependency_state


def write_phase(folder, status="complete", draws=10):
    (folder/"run.lock").touch()
    (folder/"manifest.json").write_text(json.dumps({"status": status, "expected_chunks": 1, "expected_draws": 10,
                                                   "completed": {"chunk": {"complete": draws, "failed": 0}}, "failed": {}}))


def test_released_successful_dependency_requires_complete_counts(tmp_path):
    write_phase(tmp_path)
    assert dependency_state(tmp_path) == "terminal_complete_owner_released"
    write_phase(tmp_path, draws=9)
    with pytest.raises(RuntimeError, match="draw count"):
        dependency_state(tmp_path)


def test_idle_running_manifest_is_not_a_live_process_or_restart_authorization(tmp_path):
    write_phase(tmp_path, status="running")
    with pytest.raises(RuntimeError, match="inspect the existing run"):
        dependency_state(tmp_path)
