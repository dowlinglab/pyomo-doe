from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager


def _load_test_conftest():
    conftest_path = Path(__file__).with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("tests_conftest", conftest_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load test configuration from {conftest_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_conftest = _load_test_conftest()
ROOT = _conftest.ROOT
active_notebook_paths = _conftest.active_notebook_paths


def test_active_notebooks_match_site_toc():
    notebooks = active_notebook_paths()

    assert notebooks, "No active notebooks were found in myst.yml"
    assert ROOT / "notebooks" / "doe_exploratory_analysis.ipynb" not in notebooks

    for path in notebooks:
        assert path.exists(), f"Missing notebook listed in myst.yml: {path}"


def test_active_notebooks_code_cells_compile():
    transformer = TransformerManager()

    for notebook_path in active_notebook_paths():
        with notebook_path.open() as notebook_file:
            notebook = nbformat.read(notebook_file, as_version=4)
        for cell_index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue

            transformed = transformer.transform_cell(cell.source)
            compile(
                transformed,
                filename=f"{notebook_path}:{cell_index}",
                mode="exec",
            )


def test_site_build_smoke():
    completed = subprocess.run(
        ["myst", "build", "--html"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert (ROOT / "_build" / "html" / "index.html").exists()
