"""Run selected full-start audits for every declared uncertainty mechanism."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def save(path, data):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    if not 1 <= a.workers <= 4:
        raise ValueError("Invalid numerical worker cap")
    a.output.mkdir(parents=True, exist_ok=True)
    matrix = json.loads((a.matrix/"manifest.json").read_text())
    if matrix["status"] not in {"complete", "complete_with_issues"} or len(matrix["jobs"]) != 24:
        raise ValueError("Uncertainty matrix must be terminal with all 24 cells")
    source = json.loads((a.output/"source_snapshot.json").read_text())
    tasks = []
    for name, row in matrix["jobs"].items():
        if row["status"] not in {"complete", "requires_revisit"}:
            raise ValueError("A simulation cell failed: "+name)
        cp_path = a.matrix/name/"checkpoint.json"
        checkpoint = json.loads(cp_path.read_text())
        pair = row["pair"]
        reference = a.references/pair
        original = json.loads((reference/"fit.json").read_text())
        baseline = original["best"]
        original_mode = baseline["tau"] >= baseline["theta"]
        draws = checkpoint["draws"]
        wanted = [0, checkpoint["identity"]["draws"]-1] if name.endswith(("_gaussian", "_blocks60")) else [0]
        other = [d["index"] for d in draws if d["status"] == "complete" and
                 (d["fit"]["best"]["tau"] >= d["fit"]["best"]["theta"]) != original_mode]
        if other: wanted.append(min(other))
        flagged = [d["index"] for d in draws if d["status"] == "requires_revisit"]
        wanted.extend(flagged)
        indices = sorted(set(wanted))
        command = [sys.executable, str(HERE/"audit_refit_bootstrap.py"), "--input", str(a.matrix/name),
                   "--reference", str(reference), "--output", str(a.output/name),
                   "--indices", ",".join(str(i) for i in indices)]
        tasks.append({"id": name, "pair": pair, "indices": indices, "command": command,
                      "checkpoint_sha256": hashlib.sha256(cp_path.read_bytes()).hexdigest()})
    manifest = {"status": "running", "source_commit": source["commit"], "maximum_numerical_workers": a.workers,
                "jobs": {t["id"]: dict(t, status="pending") for t in tasks},
                "scope": "Every mechanism and pair: primary first/last draws, sensitivity first draw, first pole switch when present, and every flagged draw. All get independent mean/dense-likelihood checks and 24-start refits; no claim of exhaustive global validation."}
    save(a.output/"manifest.json", manifest)
    pending, running = list(tasks), {}
    while pending or running:
        while pending and len(running) < a.workers:
            task = pending.pop(0); folder = a.output/task["id"]; folder.mkdir(exist_ok=False)
            receipt = copy.deepcopy(source)
            receipt.update(command=task["command"], checkpoint_sha256=task["checkpoint_sha256"],
                           parent_receipt_sha256=hashlib.sha256((a.output/"source_snapshot.json").read_bytes()).hexdigest())
            save(folder/"source_snapshot.json", receipt)
            log = (folder/"worker.log").open("w")
            child = subprocess.Popen(task["command"], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
            save(folder/"process.json", {"pid": child.pid, "command": task["command"], "started_unix": time.time()})
            running[task["id"]] = child, log, folder
            manifest["jobs"][task["id"]].update(status="running", pid=child.pid)
            save(a.output/"manifest.json", manifest)
        for name, (child, log, folder) in list(running.items()):
            code = child.poll()
            if code is None: continue
            log.close();save(folder/"exit.json", {"exit_code": code})
            state = "complete" if code == 0 else "requires_revisit"
            manifest["jobs"][name].update(status=state, exit_code=code)
            del running[name];save(a.output/"manifest.json", manifest)
            print(name, state, flush=True)
        if pending or running: time.sleep(.25)
    issues = sum(t["status"] != "complete" for t in manifest["jobs"].values())
    manifest.update(status="complete" if not issues else "complete_with_issues", issue_cells=issues)
    save(a.output/"manifest.json", manifest)
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
