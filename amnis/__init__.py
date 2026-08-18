"""Amnis — Persistent Memory, RAG, and Wiki Compilation for AI agents.

Three layers:
  - Memory store: SQLite-backed facts that survive across sessions
  - RAG engine:   ChromaDB + sentence-transformers semantic search
  - Wiki compiler: structured markdown knowledge compiled from both

Submodules are resolved lazily. Importing this package used to construct a
ChromaDB client and load an 80MB embedding model as an import side effect,
so `amnis --help` took seconds and every test touched the real data
directory. Attribute access below keeps the old spellings working.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.2.0"

_LAZY = {
    "config": "amnis.config",
    "database": "amnis.database",
    "memory_store": "amnis.memory.store",
    "memory_episodic": "amnis.memory.episodic",
    "memory_pruning": "amnis.memory.pruning",
    "memory_consolidation": "amnis.memory.consolidation",
    "rag_engine": "amnis.rag.engine",
    "wiki_compiler": "amnis.wiki.compiler",
}

__all__ = [*_LAZY, "__version__"]

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    # Re-exported for type checkers only; runtime resolution is __getattr__.
    from . import config, database  # noqa: F401
    from .memory import consolidation as memory_consolidation  # noqa: F401
    from .memory import episodic as memory_episodic  # noqa: F401
    from .memory import pruning as memory_pruning  # noqa: F401
    from .memory import store as memory_store  # noqa: F401
    from .rag import engine as rag_engine  # noqa: F401
    from .wiki import compiler as wiki_compiler  # noqa: F401


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(__all__)
