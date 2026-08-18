"""Shared fixtures.

Every test runs against a throwaway data directory. This is only possible
because the layers are lazy singletons now — in 0.1, importing ``amnis``
constructed a ChromaDB client against the real ``~/amnis/data`` before a test
could redirect anything.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _reset_all() -> None:
    from amnis import database
    from amnis.memory import working
    from amnis.rag import engine as rag_engine
    from amnis.wiki import compiler as wiki_compiler

    database.reset_engine()
    rag_engine.reset_engine()
    wiki_compiler.reset_compiler()
    working.reset_working_memory()


@pytest.fixture()
def amnis_env(tmp_path, monkeypatch):
    """Point every path at a temp directory and reset cached singletons."""
    monkeypatch.setenv("AMNIS_DATA_DIR", str(tmp_path))

    import amnis.config as config_module

    importlib.reload(config_module)
    fresh = config_module.config

    # Modules captured `config` by value at import time; rebind it so they all
    # see the temp directory.
    for name in (
        "amnis.database",
        "amnis.rag.keyword",
        "amnis.rag.engine",
        "amnis.memory.store",
        "amnis.memory.episodic",
        "amnis.memory.pruning",
        "amnis.memory.consolidation",
        "amnis.memory.working",
        "amnis.wiki.compiler",
    ):
        module = importlib.import_module(name)
        if hasattr(module, "config"):
            monkeypatch.setattr(module, "config", fresh, raising=False)

    from amnis.memory import store

    monkeypatch.setattr(store.memory_index, "_db_path", fresh.keyword_db, raising=False)
    monkeypatch.setattr(store.memory_index, "_initialised", False, raising=False)

    _reset_all()
    for directory in (fresh.data_dir, fresh.notes_dir, fresh.wiki_dir, fresh.wiki_facts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    yield fresh

    _reset_all()


@pytest.fixture()
def no_embeddings(monkeypatch, amnis_env):
    """Stub the vector layer so tests stay fast and offline.

    sentence-transformers would otherwise download an 80MB model on a cold CI
    runner, and ChromaDB startup dominates the runtime of every test.
    """
    from amnis.memory import store
    from amnis.rag import engine as rag_engine

    monkeypatch.setattr(store, "_write_wiki_entry", lambda *a, **k: None)
    monkeypatch.setattr(store, "_semantic_memory_ids", lambda *a, **k: [])
    monkeypatch.setattr(
        rag_engine,
        "get_engine",
        lambda: (_ for _ in ()).throw(rag_engine.RagError("vector layer disabled in tests")),
    )
    return amnis_env
