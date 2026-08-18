"""Packaging and import-time behaviour."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_subpackage_is_importable():
    # Without these __init__.py files a built wheel shipped only the three
    # top-level modules and `import amnis.memory.store` failed after install.
    for package in ("amnis.memory", "amnis.rag", "amnis.wiki", "amnis.server"):
        __import__(package)


def test_importing_amnis_is_cheap():
    """Import must not construct ChromaDB or load an embedding model."""
    code = (
        "import sys, time; "
        "start = time.perf_counter(); import amnis; "
        "print(time.perf_counter() - start); "
        "print('chromadb' in sys.modules); "
        "print('sentence_transformers' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert float(out[0]) < 1.0
    assert out[1] == "False"
    assert out[2] == "False"


def test_lazy_attributes_still_resolve():
    import amnis

    assert amnis.memory_store.store is not None
    assert amnis.__version__


def test_declared_dependencies_cover_direct_imports():
    text = (ROOT / "pyproject.toml").read_text()
    for dependency in (
        "chromadb",
        "mcp",
        "fastapi",
        "uvicorn",
        "pydantic-settings",
        "sqlalchemy",
        "sentence-transformers",
        "numpy",
    ):
        assert f'"{dependency}' in text, f"{dependency} missing from dependencies"


def test_license_file_matches_the_readme_claim():
    assert (ROOT / "LICENSE").exists()
    assert "MIT" in (ROOT / "LICENSE").read_text()
