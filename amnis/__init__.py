"""Amnis — Persistent Memory, RAG, and Wiki Compilation for Hermes Agent.

Three layers:
  - Memory Store: SQLite-backed persistent fact storage (cross-session facts, prefs)
  - RAG Engine: ChromaDB + sentence-transformers semantic search over indexed documents
  - Wiki Compiler: Karpathy-style structured knowledge compilation into markdown
"""
from . import config, database
from .memory import store as memory_store
from .rag import engine as rag_engine
from .wiki import compiler as wiki_compiler

__version__ = "0.1.0"
