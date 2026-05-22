from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
MYST_FILE = ROOT / "myst.yml"


def load_tclab_pyomo():
    module_path = NOTEBOOK_DIR / "tclab_pyomo.py"
    sys.modules.pop("tclab_pyomo", None)
    spec = importlib.util.spec_from_file_location("tclab_pyomo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def active_notebook_paths():
    notebooks = []
    for raw_line in MYST_FILE.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            continue
        if line.startswith("- file:") and line.endswith(".ipynb"):
            notebooks.append(ROOT / line.split(":", 1)[1].strip())
    return notebooks


@pytest.fixture(scope="session")
def tclab_pyomo_module():
    return load_tclab_pyomo()
