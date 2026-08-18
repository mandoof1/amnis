"""MCP server — exposes the memory, RAG, and wiki layers as agent tools.

Run with::

    python -m amnis.server.mcp      # or: amnis server

The 0.1 version of this file did not import at all: the ``amnis_prune``
schema contained the JavaScript literal ``false`` inside a Python dict, a
hard SyntaxError. Every client therefore saw zero tools. The hand-written
JSON Schemas that caused it are gone — schemas are now generated from the
type hints, so a signature and its schema cannot drift apart.
"""

from __future__ import annotations

import json
from typing import Any, Literal

# mcp 2.0 renamed the decorator-style server; the tool()/run() surface used
# here is identical in both, so one shim covers the whole supported range.
try:  # pragma: no cover - depends on installed mcp version
    from mcp.server import MCPServer as _ServerClass  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP as _ServerClass  # type: ignore[assignment]

from ..config import config, unknown_env_vars

server = _ServerClass("amnis")

Category = Literal[
    "preference",
    "fact",
    "event",
    "procedure",
    "concept",
    "theme",
    "meta",
    "general",
]
Role = Literal["user", "assistant"]
Outcome = Literal["success", "failure", "neutral"]


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


# ─── Memory ────────────────────────────────────────────────────────────


@server.tool()
def amnis_remember(
    fact: str,
    category: Category = "general",
    importance: int = 5,
    tags: list[str] | None = None,
    context: str | None = None,
) -> dict:
    """Store a fact so it survives across sessions.

    Use for anything the user says about themselves, their preferences, their
    setup, or anything that should never be forgotten. importance is 1-10,
    where 10 means critical.
    """
    from ..memory import store

    return store.store(
        fact=fact,
        category=category,
        importance=importance,
        source="agent",
        tags=tags,
        context=context,
    )


@server.tool()
def amnis_recall(
    query: str = "",
    category: Category | None = None,
    limit: int = 10,
    min_importance: int = 0,
    tags: list[str] | None = None,
) -> dict:
    """Recall stored memories by meaning and keyword.

    Call this before answering anything that depends on what the user told you
    in an earlier session, or to check whether something is already known.
    """
    from ..memory import store

    results = store.recall(
        query=query,
        category=category,
        limit=limit,
        min_importance=min_importance,
        tags=tags,
    )
    return {"count": len(results), "memories": results}


@server.tool()
def amnis_update_memory(
    memory_id: str,
    fact: str | None = None,
    category: Category | None = None,
    importance: int | None = None,
    tags: list[str] | None = None,
    context: str | None = None,
) -> dict:
    """Edit an existing memory in place, keeping its ID and access history."""
    from ..memory import store

    updated = store.update(
        memory_id,
        fact=fact,
        category=category,
        importance=importance,
        tags=tags,
        context=context,
    )
    if updated is None:
        return {"updated": False, "reason": f"No memory with id {memory_id}"}
    return {"updated": True, "memory": updated}


@server.tool()
def amnis_forget(memory_id: str) -> dict:
    """Delete a memory and everything derived from it (wiki page, vectors)."""
    from ..memory import store

    return {"deleted": store.forget(memory_id), "memory_id": memory_id}


@server.tool()
def amnis_memory_stats() -> dict:
    """Counts, category breakdown, and average importance of stored memories."""
    from ..memory import store

    return store.stats()


@server.tool()
def amnis_consolidate(limit: int = 50) -> dict:
    """Extract durable facts from recent conversation logs, then reflect."""
    from ..memory import consolidation

    return consolidation.run_pipeline()


@server.tool()
def amnis_reindex_memories() -> dict:
    """Rebuild the memory keyword index. Run once after upgrading from 0.1."""
    from ..memory import store

    return store.reindex_keywords()


# ─── RAG ───────────────────────────────────────────────────────────────


@server.tool()
def amnis_search(query: str, limit: int = 5, where: dict | None = None) -> dict:
    """Semantic search over indexed documents (notes, markdown, code)."""
    from ..rag.engine import RagError, get_engine

    try:
        results = get_engine().search(query=query, limit=limit, where=where)
    except RagError as exc:
        return {"error": str(exc), "results": []}
    return {"count": len(results), "results": results}


@server.tool()
def amnis_hybrid_search(query: str, limit: int = 5, where: dict | None = None) -> dict:
    """Semantic + keyword search fused with Reciprocal Rank Fusion.

    Best when you need both meaning and exact term matches — identifiers,
    error strings, file names.
    """
    from ..rag.engine import get_engine

    results = get_engine().hybrid_search(query=query, limit=limit, where=where)
    return {"count": len(results), "results": results}


@server.tool()
def amnis_index_file(file_path: str) -> dict:
    """Index one file into the vector store so it becomes searchable."""
    from ..rag.engine import RagError, get_engine

    try:
        return get_engine().index_file(file_path)
    except RagError as exc:
        return {"error": str(exc), "indexed": 0}


@server.tool()
def amnis_index_notes() -> dict:
    """Index (or re-index) every note in the notes directory."""
    from ..rag.engine import get_engine

    return get_engine().index_notes()


@server.tool()
def amnis_rag_stats() -> dict:
    """Chunk count, source count, embedding model, keyword index size."""
    from ..rag.engine import get_engine

    return get_engine().stats()


@server.tool()
def amnis_list_sources(origin: str | None = None, limit: int = 100) -> dict:
    """List indexed documents. origin is one of note, wiki, memory, compiled."""
    from ..rag.engine import get_engine

    sources = get_engine().sources(origin=origin, limit=limit)
    return {"count": len(sources), "sources": sources}


# ─── Wiki ──────────────────────────────────────────────────────────────


@server.tool()
def amnis_compile_wiki(topics: list[str] | None = None) -> dict:
    """Compile wiki pages from indexed notes and memory. Omit topics for all."""
    from ..wiki.compiler import get_compiler

    return get_compiler().compile(topics=topics)


@server.tool()
def amnis_wiki_query(question: str) -> dict:
    """Ask the compiled wiki a question — searches pages and the RAG index."""
    from ..wiki.compiler import get_compiler

    return get_compiler().query(question)


@server.tool()
def amnis_wiki_lint() -> dict:
    """Report stale pages, pages without sources, and duplicate titles."""
    from ..wiki.compiler import get_compiler

    return get_compiler().lint()


@server.tool()
def amnis_wiki_stats() -> dict:
    """Page count, distinct source count, and last compile time."""
    from ..wiki.compiler import get_compiler

    return get_compiler().stats()


# ─── Episodic ──────────────────────────────────────────────────────────


@server.tool()
def amnis_episodic_log(
    session_id: str,
    role: Role,
    content: str,
    summary: str | None = None,
    topics: list[str] | None = None,
    outcome: Outcome | None = None,
) -> dict:
    """Log a conversation turn, auto-extracting topics and a summary."""
    from ..memory import episodic

    return episodic.log_episode(
        session_id=session_id,
        role=role,
        content=content,
        summary=summary,
        topics=topics,
        outcome=outcome,
    )


@server.tool()
def amnis_episodic_recall(
    session_id: str | None = None,
    topic: str | None = None,
    role: Role | None = None,
    limit: int = 20,
) -> dict:
    """Recall past interactions, filtered by session, topic, or speaker."""
    from ..memory import episodic

    results = episodic.recall_episodes(session_id=session_id, topic=topic, role=role, limit=limit)
    return {"count": len(results), "episodes": results}


@server.tool()
def amnis_episodic_stats() -> dict:
    """Total episodes, unique sessions, and the date range covered."""
    from ..memory import episodic

    return episodic.stats()


@server.tool()
def amnis_episodic_prune(days: int = 30) -> dict:
    """Delete episodes older than `days`. Semantic memory is untouched."""
    from ..memory import episodic

    return {"pruned": episodic.prune_old_episodes(days=days), "retention_days": days}


# ─── Maintenance ───────────────────────────────────────────────────────


@server.tool()
def amnis_prune_memory(dry_run: bool = False) -> dict:
    """Decay confidence, drop expired, merge duplicates, prune stale memories.

    Pass dry_run=true to preview without deleting anything.
    """
    from ..memory import pruning

    return pruning.run_pipeline(dry_run=dry_run)


@server.tool()
def amnis_status() -> dict:
    """Health and statistics for all three layers.

    Each layer is probed independently, so one broken subsystem reports as
    degraded instead of failing the whole call.
    """
    report: dict[str, Any] = {"status": "ok", "errors": {}}

    from ..memory import episodic, store

    for name, fn in (
        ("memory", store.stats),
        ("episodic", episodic.stats),
    ):
        try:
            report[name] = fn()
        except Exception as exc:  # noqa: BLE001 - degraded reporting is the point
            report[name] = {}
            report["errors"][name] = str(exc)

    try:
        from ..rag.engine import get_engine

        report["rag"] = get_engine().stats()
    except Exception as exc:  # noqa: BLE001
        report["rag"] = {}
        report["errors"]["rag"] = str(exc)

    try:
        from ..wiki.compiler import get_compiler

        report["wiki"] = get_compiler().stats()
    except Exception as exc:  # noqa: BLE001
        report["wiki"] = {}
        report["errors"]["wiki"] = str(exc)

    if report["errors"]:
        report["status"] = "degraded"

    report["config"] = {
        "data_dir": str(config.data_dir),
        "notes_dir": str(config.notes_dir),
        "wiki_dir": str(config.wiki_dir),
        "embedding_model": config.embedding_model,
    }
    unknown = unknown_env_vars()
    if unknown:
        report["config"]["ignored_env_vars"] = unknown
    return report


def main() -> None:
    """Entry point for `amnis server` and `python -m amnis.server.mcp`."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
