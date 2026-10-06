"""Regression tests for the standalone package boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_installed_package_imports_outside_source_tree(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "true",
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        }
    )
    script = """
import importlib
import pkgutil
import sys
import deepilg
import deepilg.models as models
import deepilg.training as training

for module in pkgutil.walk_packages(deepilg.__path__, deepilg.__name__ + "."):
    importlib.import_module(module.name)
assert training.ConductanceModel is models.ConductanceModel
assert not any(name.startswith("reports.neurips_paper.code") for name in sys.modules)
print(deepilg.__file__)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "deepilg/__init__.py" in result.stdout.replace("\\", "/")
