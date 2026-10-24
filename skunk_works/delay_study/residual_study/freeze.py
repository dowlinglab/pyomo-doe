"""Execute a phase driver from committed source, without importing the checkout."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    if len(sys.argv) < 4 or Path(sys.argv[1]).name != sys.argv[1] or "--output" not in sys.argv:
        raise SystemExit("Usage: python freeze.py DRIVER.py --output ABSOLUTE_DIRECTORY [arguments]")
    driver, args = sys.argv[1], sys.argv[2:]
    output = Path(args[args.index("--output")+1])
    if not output.is_absolute():
        raise SystemExit("Output must be absolute")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prefix = HERE.relative_to(ROOT)
    parent = prefix.parent
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision,
                                       "--", str(parent), "data"], cwd=ROOT, text=True).splitlines()
    selected = [p for p in tracked if
                (Path(p).parent == prefix and (p.endswith(".py") or Path(p).name in {"config.json", "structure_selection.json"})) or
                (Path(p).parent == parent and (p.endswith(".py") or Path(p).name == "environment.json")) or
                (Path(p).parent == Path("data") and p.endswith(".csv"))]
    if str(prefix/driver) not in selected:
        raise SystemExit("Commit the driver before measuring")
    environment = json.loads(subprocess.check_output(["git", "show", revision+":"+str(parent/"environment.json")], cwd=ROOT))
    packages = {name: importlib.metadata.version(name) for name in environment["packages"]}
    if packages != environment["packages"] or platform.python_version() != environment["python"]:
        raise SystemExit("Runtime does not match the verified environment")
    # Never overwrite a measured run or retry over an uncertain terminal state.
    output.mkdir(parents=True, exist_ok=False)
    snapshot = Path(tempfile.mkdtemp(prefix="tclab-residual-"+revision[:8]+"-"))
    hashes = {}
    for name in selected:
        content = subprocess.check_output(["git", "show", revision+":"+name], cwd=ROOT)
        destination = snapshot/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        destination.chmod(0o444)
        hashes[name] = hashlib.sha256(content).hexdigest()
    receipt = {"commit": revision, "sha256": hashes, "packages": packages,
               "python": platform.python_version(), "snapshot": str(snapshot),
               "command": [sys.executable, str(snapshot/prefix/driver)]+args,
               "source_policy": "Read-only copies of committed source/data; working checkout not imported"}
    (output/"source_snapshot.json").write_text(json.dumps(receipt, indent=2)+"\n")
    env = dict(os.environ)
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[name] = "1"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(snapshot/"runtime_cache"/"matplotlib")
    env["XDG_CACHE_HOME"] = str(snapshot/"runtime_cache")
    print("Frozen commit "+revision+"; output "+str(output), flush=True)
    child = subprocess.Popen(receipt["command"], cwd=snapshot, env=env)
    (output/"process.json").write_text(json.dumps({"pid": child.pid, "parent_pid": os.getpid(), "command": receipt["command"]}, indent=2)+"\n")
    status = child.wait()
    (output/"exit.json").write_text(json.dumps({"exit_code": status})+"\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
