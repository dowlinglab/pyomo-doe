"""Run bounded follow-up trials after the live coverage owner releases its lock.

This is a one-shot local research dependency, not a recurring automation. It
must itself run from frozen_run.py so future trials use this exact source/data
snapshot. It never launches the full studies without a subsequent agent review.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run_fits import save_json

HERE = Path(__file__).resolve().parent


TRIALS = [
    ("design_nominal", "run_design.py", "nominal__n2__cold__iid__D__prior0"),
    ("nuisance_joint", "run_sensitivity.py", "joint__chain_n2_io_cold_linear_exact__ambient_history"),
    ("pure_left_hold", "run_sensitivity.py", "joint__pure_delay_n0_io_cold_left_exact"),
    ("original_prediction", "run_prediction_uncertainty.py", "local__joint__original_n0_physics_cold_linear_exact"),
    ("null_delay_prediction", "run_prediction_uncertainty.py", "local__step__chain_n2_io_input_equilibrated_linear_exact"),
    ("ou_prediction", "run_prediction_uncertainty.py", "ou__joint__chain_n2_io_cold_linear_exact"),
    ("bootstrap_prediction", "run_prediction_uncertainty.py", "bootstrap__joint__chain_n2_io_cold_linear_exact__parametric"),
]


def dependency_state(folder):
    """OS lock confirms liveness; only terminal complete accounting allows work."""
    folder = Path(folder)
    with (folder/"run.lock").open("r") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return "confirmed_live_owner"
        record = json.loads((folder/"manifest.json").read_text())
        if record["status"] != "complete":
            raise RuntimeError("Dependency lock is idle without successful terminal status; inspect the existing run, never restart it blindly")
        if record["failed"] or len(record["completed"]) != record["expected_chunks"]:
            raise RuntimeError("Dependency terminal chunk accounting is incomplete")
        if any(row["failed"] for row in record["completed"].values()):
            raise RuntimeError("Dependency has failed draws requiring review")
        if sum(row["complete"] for row in record["completed"].values()) != record["expected_draws"]:
            raise RuntimeError("Dependency terminal draw count is incomplete")
        return "terminal_complete_owner_released"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = HERE.parents[1]/"snapshot.json"
    if not receipt_path.exists():
        raise RuntimeError("Launch this one-shot queue through frozen_run.py")
    snapshot = json.loads(receipt_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    original_root = Path.cwd()
    study = original_root/"skunk_works/delay_study"
    with (args.output/"run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/"queue.json").exists():
            raise RuntimeError("Inspect the prior queue outcome rather than relaunching this one-shot queue")
        queue = {"status": "checking_dependency", "source_commit": snapshot["commit"],
                 "dependency": str(args.dependency), "started_utc": datetime.now(timezone.utc).isoformat(),
                 "trials": {}, "notice": "Sequential bounded trials only. No full studies start without subsequent agent inspection. Coverage and trial numerical pools never overlap under this owner."}
        save_json(args.output/"queue.json", queue)
        first = True
        while True:
            try:
                state = dependency_state(args.dependency)
            except Exception:
                queue["status"] = "dependency_requires_agent_review"
                queue["dependency_error"] = traceback.format_exc()
                save_json(args.output/"queue.json", queue)
                raise
            queue["last_dependency_state"] = state
            queue["last_observed_utc"] = datetime.now(timezone.utc).isoformat()
            queue["status"] = "waiting_for_live_dependency" if state == "confirmed_live_owner" else "running_trials"
            save_json(args.output/"queue.json", queue)
            if state == "terminal_complete_owner_released":
                print("Coverage owner released after complete accounting; bounded trials start", flush=True)
                break
            if first:
                print("Coverage owner confirmed live by OS lock; waiting without starting another numerical pool", flush=True)
                first = False
            time.sleep(30)
        for name, driver, selector in TRIALS:
            output = args.output/name
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(HERE/driver), "--fits", str(study/"results/fits"), "--output", str(output), "--only", selector]
            if driver == "run_prediction_uncertainty.py":
                command += ["--ou", str(study/"results/ou"), "--bootstrap", str(study/"results/bootstrap")]
            child_receipt = dict(snapshot)
            child_receipt["controller_reproduction_command"] = snapshot["reproduction_command"]
            child_receipt["reproduction_command"] = ["python", "skunk_works/delay_study/frozen_run.py", driver]+command[2:]
            child_receipt["source_policy"] = "This driver and raw data run directly from the controller's immutable source snapshot, at its recorded commit. No editable checkout code is imported."
            save_json(output/"source_snapshot.json", child_receipt)
            queue["trials"][name] = {"status": "running", "driver": driver, "selector": selector,
                                      "started_utc": datetime.now(timezone.utc).isoformat()}
            save_json(args.output/"queue.json", queue)
            print("Starting bounded trial "+name, flush=True)
            with (output/"execution.log").open("w") as log:
                completed = subprocess.run(command, cwd=original_root, stdout=log, stderr=subprocess.STDOUT)
            manifest_path = output/"manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            queue["trials"][name].update(status="finished_pending_agent_review", exit_code=completed.returncode,
                                          phase_status=manifest.get("status"), completed_jobs=len(manifest.get("completed", {})),
                                          failed_jobs=len(manifest.get("failed", {})),
                                          revisit_jobs=[key for key, row in manifest.get("completed", {}).items() if row.get("requires_revisit")],
                                          finished_utc=datetime.now(timezone.utc).isoformat())
            save_json(args.output/"queue.json", queue)
            print("Finished "+name+" exit="+str(completed.returncode), flush=True)
        queue["status"] = "trials_finished_pending_agent_review"
        queue["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(args.output/"queue.json", queue)


if __name__ == "__main__":
    main()
