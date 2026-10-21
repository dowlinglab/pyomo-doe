"""Run a committed analysis from a private, immutable source/data snapshot.

This keeps a long measurement isolated while subsequent manuscript or tooling
work continues in the editable checkout. Inputs/results passed to the child must
be retained unchanged; their own manifests identify their bytes.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py") or Path(sys.argv[1]).name != sys.argv[1]:
        raise SystemExit("Usage: python frozen_run.py STUDY_DRIVER.py [driver arguments]")
    driver, arguments = sys.argv[1], sys.argv[2:]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prefix = str(HERE.relative_to(ROOT))
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision, "--", prefix, "data"], cwd=ROOT, text=True).splitlines()
    selected = [p for p in tracked if (Path(p).parent == Path(prefix) and Path(p).suffix in {".py", ".json", ".txt"}) or (Path(p).parent == Path("data") and p.endswith(".csv"))]
    if prefix+"/"+driver not in selected:
        raise SystemExit("Driver must be committed before a frozen run")
    environment = json.loads(subprocess.check_output(["git", "show", revision+":"+prefix+"/environment.json"], cwd=ROOT))
    actual = {name: importlib.metadata.version(name) for name in environment["packages"]}
    if actual != environment["packages"] or platform.python_version() != environment["python"]:
        raise SystemExit("Runtime differs from the verified numerical environment: "+json.dumps(actual))
    snapshot = Path(tempfile.mkdtemp(prefix="tclab-delay-"+revision[:8]+"-"))
    hashes = {}
    for name in selected:
        content = subprocess.check_output(["git", "show", revision+":"+name], cwd=ROOT)
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        target.chmod(0o444)
        hashes[name] = hashlib.sha256(content).hexdigest()
    receipt = {"commit": revision, "sha256": hashes, "python": platform.python_version(), "packages": actual,
               "reproduction_command": ["python", prefix+"/frozen_run.py", driver]+arguments,
               "source_policy": "Committed code and raw data copied to a read-only-file snapshot; editable checkout is not imported"}
    (snapshot/"snapshot.json").write_text(json.dumps(receipt, indent=2)+"\n")
    if "--output" in arguments:
        output = Path(arguments[arguments.index("--output")+1])
        if not output.is_absolute():
            output = ROOT/output
        output.mkdir(parents=True, exist_ok=True)
        destination = output/"source_snapshot.json"
        if destination.exists():
            previous = json.loads(destination.read_text())
            # A different commit can retain byte-identical scientific sources;
            # child manifests still enforce exact measured source/configuration.
            if previous["packages"] != actual:
                raise SystemExit("Existing run has another runtime")
        else:
            destination.write_text(json.dumps(receipt, indent=2)+"\n")
    print("Frozen source commit "+revision+"; driver "+driver, flush=True)
    return subprocess.call([sys.executable, str(snapshot/prefix/driver)]+arguments, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
