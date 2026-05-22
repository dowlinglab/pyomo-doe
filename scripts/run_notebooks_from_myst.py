#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


NOTEBOOK_FILE_RE = re.compile(r"^-?\s*file:\s*['\"]?([^'\"]+\.ipynb)['\"]?\s*$")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MYST_FILE = REPO_ROOT / "myst.yml"
IDAES_BIN_DIR = Path("~/.idaes/bin").expanduser()


def parse_active_notebooks(myst_file: Path) -> list[Path]:
    notebooks: list[Path] = []
    seen: set[Path] = set()

    for raw_line in myst_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = NOTEBOOK_FILE_RE.match(line)
        if not match:
            continue

        notebook_ref = match.group(1).strip()
        notebook_path = Path(notebook_ref)
        if not notebook_path.is_absolute():
            notebook_path = (myst_file.parent / notebook_path).resolve()

        if notebook_path not in seen:
            notebooks.append(notebook_path)
            seen.add(notebook_path)

    return notebooks


def is_student_exercise_notebook(path: Path) -> bool:
    return path.parent.name == "notebooks" and path.name in {
        "parmest_exercise.ipynb",
        "doe_exercise.ipynb",
    }


def execute_notebook(path: Path, timeout: int, kernel_name: str | None) -> None:
    with path.open(encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    resolved_kernel_name = (
        kernel_name
        or notebook.get("metadata", {}).get("kernelspec", {}).get("name")
        or "python3"
    )

    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=resolved_kernel_name,
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    client.execute()


def prepend_idaes_bin_to_path() -> None:
    if not IDAES_BIN_DIR.exists():
        return

    current_path = os.environ.get("PATH", "")
    idaes_bin = str(IDAES_BIN_DIR)
    path_entries = current_path.split(os.pathsep) if current_path else []
    if idaes_bin in path_entries:
        return
    os.environ["PATH"] = (
        f"{idaes_bin}{os.pathsep}{current_path}" if current_path else idaes_bin
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute active notebooks listed in a MyST TOC file."
    )
    parser.add_argument(
        "myst_file",
        nargs="?",
        default=str(DEFAULT_MYST_FILE),
        help=f"Path to myst.yml (default: {DEFAULT_MYST_FILE}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Cell execution timeout in seconds (default: 1800).",
    )
    parser.add_argument(
        "--kernel-name",
        default=None,
        help="Kernel name override (default: use notebook metadata).",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Do not fail when no active notebook entries are found.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List resolved notebook paths and exit without executing.",
    )
    parser.add_argument(
        "--max-notebooks",
        type=int,
        default=None,
        help="Optional limit for how many notebooks to execute, in TOC order.",
    )
    parser.add_argument(
        "--include-exercises",
        action="store_true",
        help="Include student exercise notebooks in the execution set.",
    )
    args = parser.parse_args()

    prepend_idaes_bin_to_path()

    myst_file = Path(args.myst_file).resolve()
    if not myst_file.exists():
        print(f"[ERROR] MyST file not found: {myst_file}", file=sys.stderr)
        return 2

    notebooks = parse_active_notebooks(myst_file)
    if not args.include_exercises:
        notebooks = [nb for nb in notebooks if not is_student_exercise_notebook(nb)]

    if not notebooks:
        message = f"[INFO] No active notebook entries found in {myst_file}"
        if args.allow_empty:
            print(message)
            return 0
        print(f"[ERROR] {message}", file=sys.stderr)
        return 2

    missing = [path for path in notebooks if not path.exists()]
    if missing:
        print("[ERROR] Some notebooks listed in the TOC do not exist:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2

    if args.max_notebooks is not None:
        if args.max_notebooks <= 0:
            print("[ERROR] --max-notebooks must be positive.", file=sys.stderr)
            return 2
        notebooks = notebooks[: args.max_notebooks]

    if args.list_only:
        print(f"[INFO] Found {len(notebooks)} active notebooks in {myst_file}")
        for notebook in notebooks:
            print(notebook)
        return 0

    print(f"[INFO] Running {len(notebooks)} notebooks from {myst_file}")
    failures: list[tuple[Path, float, str]] = []
    started_all = time.time()

    for idx, notebook_path in enumerate(notebooks, start=1):
        started = time.time()
        print(f"[RUN {idx}/{len(notebooks)}] {notebook_path}")
        try:
            execute_notebook(
                path=notebook_path,
                timeout=args.timeout,
                kernel_name=args.kernel_name,
            )
        except Exception as err:  # pragma: no cover
            duration = time.time() - started
            failures.append((notebook_path, duration, repr(err)))
            print(f"[FAIL] {notebook_path} ({duration:.1f}s): {err}")
            continue

        duration = time.time() - started
        print(f"[PASS] {notebook_path} ({duration:.1f}s)")

    total = time.time() - started_all
    print(f"[DONE] Completed in {total:.1f}s")

    if failures:
        print(f"[SUMMARY] {len(failures)} notebook(s) failed:", file=sys.stderr)
        for path, duration, err_text in failures:
            print(f"  - {path} ({duration:.1f}s): {err_text}", file=sys.stderr)
        return 1

    print("[SUMMARY] All notebooks executed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
