"""Run the declared six-pair uncertainty/profile cells with a fixed worker cap.

This supervisor imports no numerical implementation. Each child uses the same
read-only source snapshot, one numerical worker, and a separate durable output.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["bootstrap", "profiles", "predictions"], required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only", default="")
    parser.add_argument("--uncertainty", type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not args.references.is_absolute() or not args.output.is_absolute():
        raise ValueError("Absolute paths and one to four numerical workers required")
    if args.phase == "predictions" and (args.uncertainty is None or not args.uncertainty.is_absolute()):
        raise ValueError("Predictions need an absolute uncertainty matrix path")
    parent_receipt = args.output/"source_snapshot.json"
    source = json.loads(parent_receipt.read_text())
    config = json.loads((HERE/"config.json").read_text())
    if [p["id"] for p in config["pairs"]] != ["P"+str(i) for i in range(1, 7)]:
        raise ValueError("This goal permits only the six declared pairs")
    tasks = []
    for pair in config["pairs"]:
        reference = args.references/pair["id"]
        fit_path = reference/"fit.json"
        fit = json.loads(fit_path.read_text())
        if len(fit["record_names"]) != 2 or fit["pair"] != pair["id"]:
            raise ValueError("Invalid restricted calibration input")
        if args.phase == "bootstrap":
            cells = [("gaussian", "gaussian", 400, 60), ("blocks60", "innovation_blocks", 400, 60),
                     ("blocks30", "innovation_blocks", 100, 30), ("blocks120", "innovation_blocks", 100, 120)]
            for label, mechanism, draws, block in cells:
                identity = pair["id"]+"_"+label
                command = [sys.executable, str(HERE/"run_refit_bootstrap.py"), "--reference", str(reference),
                           "--output", str(args.output/identity), "--draws", str(draws), "--workers", "1",
                           "--measurement", mechanism, "--block-length", str(block)]
                tasks.append({"id": identity, "pair": pair["id"], "command": command,
                              "expected_draws": draws, "reference_fit_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest()})
        elif args.phase == "profiles":
            for parameter in ("K", "tau", "theta"):
                identity = pair["id"]+"_"+parameter
                command = [sys.executable, str(HERE/"run_pair_inference.py"), "--reference", str(reference),
                           "--parameter", parameter, "--output", str(args.output/identity)]
                tasks.append({"id": identity, "pair": pair["id"], "command": command,
                              "reference_fit_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest()})
        else:
            identity = pair["id"]
            gaussian = args.uncertainty/(identity+"_gaussian")
            blocks = args.uncertainty/(identity+"_blocks60")
            command = [sys.executable, str(HERE/"run_predictions.py"), "--reference", str(reference),
                       "--gaussian", str(gaussian), "--blocks", str(blocks), "--minimum-draws", "360",
                       "--output", str(args.output/identity)]
            tasks.append({"id": identity, "pair": identity, "command": command,
                          "reference_fit_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
                          "ready_cells": [identity+"_gaussian", identity+"_blocks60"]})
    if args.only:
        selected = set(args.only.split(","))
        if not selected.issubset({t["id"] for t in tasks}):
            raise ValueError("Unknown requested cell")
        tasks = [t for t in tasks if t["id"] in selected]
    manifest = {"phase": args.phase, "status": "running", "pid": os.getpid(),
                "started_unix": time.time(), "maximum_numerical_workers": args.workers,
                "source_commit": source["commit"], "expected_cells": len(tasks),
                "scope": "Only the six declared calibration pairs; one numerical process per active cell",
                "jobs": {t["id"]: dict(t, status="pending") for t in tasks}}
    save(args.output/"manifest.json", manifest)
    save(args.output/"worker_limit.json", {"workers": args.workers,
         "notice": "Change only after reconciling every study worker; the combined study cap remains four."})
    pending = list(tasks)
    running = {}
    while pending or running:
        limit = json.loads((args.output/"worker_limit.json").read_text())["workers"]
        if not isinstance(limit, int) or not 1 <= limit <= 4:
            raise ValueError("Invalid numerical worker limit")
        manifest["maximum_numerical_workers"] = limit
        while pending and len(running) < limit:
            task = pending[0]
            if args.phase == "predictions":
                dependency = json.loads((args.uncertainty/"manifest.json").read_text())
                states = [dependency["jobs"][name]["status"] for name in task["ready_cells"]]
                if any(state == "failed" for state in states):
                    manifest["jobs"][task["id"]].update(status="failed_dependency", dependency_states=states)
                    pending.pop(0)
                    save(args.output/"manifest.json", manifest)
                    continue
                if any(state not in {"complete", "requires_revisit"} for state in states):
                    break
                task["checkpoint_sha256"] = {name: hashlib.sha256((args.uncertainty/name/"checkpoint.json").read_bytes()).hexdigest()
                                                for name in task["ready_cells"]}
            pending.pop(0)
            folder = args.output/task["id"]
            folder.mkdir(exist_ok=False)
            receipt = copy.deepcopy(source)
            receipt.update(command=task["command"], parent_matrix_receipt_sha256=hashlib.sha256(parent_receipt.read_bytes()).hexdigest(),
                           reference_fit_sha256=task["reference_fit_sha256"], source_policy="Read-only committed source shared with the source-bound matrix supervisor; no mutable checkout imports")
            if args.phase == "predictions":
                receipt["checkpoint_sha256"] = task["checkpoint_sha256"]
            save(folder/"source_snapshot.json", receipt)
            log = (folder/"worker.log").open("w")
            child = subprocess.Popen(task["command"], cwd=HERE.parents[2], stdout=log, stderr=subprocess.STDOUT)
            process = {"pid": child.pid, "parent_pid": os.getpid(), "command": task["command"], "started_unix": time.time()}
            save(folder/"process.json", process)
            manifest["jobs"][task["id"]].update(status="running", process=process)
            running[task["id"]] = (child, log, folder)
            save(args.output/"manifest.json", manifest)
        for identity, (child, log, folder) in list(running.items()):
            code = child.poll()
            if code is None:
                continue
            log.close()
            save(folder/"exit.json", {"exit_code": code})
            state = "complete" if code == 0 else "failed"
            if code == 0:
                if args.phase == "bootstrap":
                    summary = json.loads((folder/"summary.json").read_text())
                    if summary["failed"] or summary["requires_revisit"]:
                        state = "requires_revisit"
                elif args.phase == "profiles":
                    profile = json.loads((folder/"profile.json").read_text())
                    if profile["numerical_status"] != "complete":
                        state = "requires_revisit"
                else:
                    audit = json.loads((folder/"audit.json").read_text())
                    if audit["status"] != "passed for declared numerical checks":
                        state = "requires_revisit"
            manifest["jobs"][identity].update(status=state, exit_code=code, ended_unix=time.time())
            del running[identity]
            save(args.output/"manifest.json", manifest)
            print(identity, state, flush=True)
        if running or pending:
            time.sleep(.25)
    issues = sum(t["status"] != "complete" for t in manifest["jobs"].values())
    manifest.update(status="complete" if not issues else "complete_with_issues", issue_cells=issues, ended_unix=time.time())
    save(args.output/"manifest.json", manifest)
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
