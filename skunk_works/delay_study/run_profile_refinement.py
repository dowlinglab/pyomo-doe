"""Resolve selected observed profile crossings without replacing first-pass files."""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import traceback

from data import load
from profile_refinement import ProfileRefiner
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_names():
    names = []
    for calibration in ("sine", "step", "joint"):
        names.extend(calibration+"__original_n0_physics_cold_linear_exact__"+parameter+".json"
                     for parameter in ("Ub", "CpH", "CpS"))
        names.extend(calibration+"__chain_n1_"+coordinates+"_cold_linear_exact__theta.json"
                     for coordinates in ("io", "physics"))
    names.extend("joint__"+model+"_io_cold_linear_exact__theta.json"
                 for model in ("chain_n2", "chain_n12", "pure_delay_n0"))
    return sorted(names)


def execute(task):
    name, root_name, output_name, config, identity, entry = task
    root, output = Path(root_name), Path(output_name)
    profile_path = root/entry["path"]
    fit_path = root/"fits"/(name.rsplit("__", 1)[0]+".json")
    inputs = {"fit": sha(fit_path), "profile": sha(profile_path)}
    if inputs["profile"] != entry["sha256"]:
        raise RuntimeError("Cataloged input profile bytes changed: "+name)
    destination = output/name
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["identity"] != identity or saved["inputs_sha256"] != inputs:
            raise RuntimeError("Refinement reuse identity changed: "+name)
        return name, "reused", saved["profile"]["refinement_requires_revisit"], saved["profile"]["baseline_requires_revisit"]
    record = json.loads(fit_path.read_text())
    previous = json.loads(profile_path.read_text())
    if previous["fit_sha256"] != inputs["fit"]:
        raise RuntimeError("Profile fit input bytes changed: "+name)
    refiner = ProfileRefiner(record, previous["profile"], [load(item) for item in record["datasets"]],
                             starts=config["starts"], seed=config["seed"], max_nfev=config["max_nfev"])
    result = refiner.refine(config["statistic_tolerance"], config["log_width_tolerance"], config["maximum_steps"])
    save_json(destination, {"identity": identity, "inputs_sha256": inputs, "source_profile": entry["path"],
                            "fit_file": fit_path.name, "fit_sha256": inputs["fit"], "profile": result})
    return name, "complete", result["refinement_requires_revisit"], result["baseline_requires_revisit"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--output", type=Path, default=HERE/"results/profile_refinement")
    parser.add_argument("--config", type=Path, default=HERE/"profile_refinement_config.json")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    catalog_path = args.results/"profile_index.json"
    catalog = json.loads(catalog_path.read_text())["profiles"]
    sources = ("models.py", "data.py", "inference.py", "profiles.py", "original_family_bounds.py",
               "profile_refinement.py", "run_profile_refinement.py", "run_fits.py")
    identity = {"sources": {name: sha(HERE/name) for name in sources},
                "config_sha256": sha(args.config), "source_catalog_sha256": sha(catalog_path)}
    names = [name for name in selected_names() if args.only in name]
    if not names or any(name not in catalog for name in names):
        raise RuntimeError("Refinement selection is empty or missing an input profile")
    with (args.output/"run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = {"identity": identity, "started_utc": datetime.now(timezone.utc).isoformat(),
                    "expected_jobs": len(names), "completed": {}, "failed": {}, "status": "running",
                    "filter": args.only, "selection": config["selection"],
                    "notice": "No global-minimum or missed-mode certificate. First-pass profiles remain immutable; refined endpoints require a separate arithmetic and exact-family inclusion audit."}
        save_json(args.output/"manifest.json", manifest)
        tasks = [(name, str(args.results), str(args.output), config, identity, catalog[name]) for name in names]
        with ProcessPoolExecutor(max_workers=min(config["workers"], 4)) as pool:
            futures = {pool.submit(execute, task): task[0] for task in tasks}
            for future in as_completed(futures):
                try:
                    name, status, revisit, baseline = future.result()
                    manifest["completed"][name] = {"status": status, "refinement_requires_revisit": revisit,
                                                   "baseline_requires_revisit": baseline}
                    print(str(len(manifest["completed"]))+"/"+str(len(names))+" "+name, flush=True)
                except Exception:
                    manifest["failed"][futures[future]] = traceback.format_exc()
                    print("FAILED "+futures[future], flush=True)
                save_json(args.output/"manifest.json", manifest)
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "complete" if not manifest["failed"] else "complete_with_failures"
        save_json(args.output/"manifest.json", manifest)
        return bool(manifest["failed"] or any(row["refinement_requires_revisit"] or row["baseline_requires_revisit"] for row in manifest["completed"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
