"""Amnis web dashboard — FastAPI backend for the browser UI.

Run with::

    amnis web                       # or
    uvicorn amnis.server.web:app --host 127.0.0.1 --port 8799

Notable changes from 0.1:

* **Handlers are synchronous.** Every endpoint was ``async def`` while its
  body did blocking SQLite, ChromaDB, and sentence-transformers work, so a
  single search froze the event loop for every other request. Sync handlers
  are dispatched to FastAPI's threadpool, which is the correct home for
  blocking I/O.
* **``/api/index-file`` is confined** to the notes and wiki directories.
  It previously accepted any absolute path and would happily embed
  ``/etc/passwd`` into the vector store on request.
* **Mutations are POST/PATCH/DELETE.** Consolidation ran on GET, so any
  crawler, prefetcher, or refresh could rewrite the memory store.
* **Requests are validated by Pydantic models** rather than ``data: dict``
  plus ``data["fact"]``, which turned a missing field into a 500.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import config, unknown_env_vars

logger = logging.getLogger(__name__)

_UI_PATH = Path(__file__).parent / "ui.html"
_STATIC_DIR = Path(__file__).parent / "static"
_ui_cache: str | None = None


# ─── Lazy layer accessors ──────────────────────────────────────────────
# Imported inside functions so `uvicorn --reload` and `--help` stay fast and
# an unavailable optional layer degrades one endpoint, not the whole app.


def _memory():
    from ..memory import store

    return store


def _rag():
    from ..rag.engine import get_engine

    return get_engine()


def _wiki():
    from ..wiki.compiler import get_compiler

    return get_compiler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the embedding model so the first search isn't a 10s stall."""
    for var in unknown_env_vars():
        logger.warning("Ignoring unknown environment variable %s", var)
    try:
        _rag().embedder.encode(["warmup"])
        logger.info("Embedding model ready (%s)", config.embedding_model)
    except Exception as exc:  # noqa: BLE001 - startup must not hard-fail
        logger.warning("Could not pre-load embedding model: %s", exc)
    yield


app = FastAPI(title="Amnis", version="0.2.0", lifespan=lifespan)


# ─── Auth ──────────────────────────────────────────────────────────────


def require_token(x_amnis_token: str | None = Header(default=None)) -> None:
    """Guard mutating endpoints when AMNIS_API_TOKEN is configured.

    The dashboard binds to localhost by default, but people put it behind a
    tunnel; without this, anything that could reach the port could rewrite the
    memory store.
    """
    if not config.api_token:
        return
    if x_amnis_token != config.api_token:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Amnis-Token")


Auth = Depends(require_token)


# ─── Request models ────────────────────────────────────────────────────

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


class MemoryIn(BaseModel):
    fact: str = Field(min_length=1, max_length=5000)
    category: Category = "general"
    importance: int = Field(default=5, ge=1, le=10)
    source: str = "web-ui"
    tags: list[str] | None = None
    context: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class MemoryPatch(BaseModel):
    fact: str | None = Field(default=None, min_length=1, max_length=5000)
    category: Category | None = None
    importance: int | None = Field(default=None, ge=1, le=10)
    tags: list[str] | None = None
    context: str | None = None


class IndexFileIn(BaseModel):
    path: str = Field(min_length=1)


class CompileWikiIn(BaseModel):
    topics: list[str] | None = None


class PruneIn(BaseModel):
    dry_run: bool = True


# ─── Status ────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/api/status")
def api_status() -> dict:
    from ..memory import episodic

    report: dict[str, Any] = {"errors": {}}

    try:
        report["memory"] = _memory().stats()
    except Exception as exc:  # noqa: BLE001
        report["memory"] = {"total_memories": 0}
        report["errors"]["memory"] = str(exc)

    try:
        report["rag"] = _rag().stats()
    except Exception as exc:  # noqa: BLE001
        report["rag"] = {"total_chunks": 0, "unique_sources": 0}
        report["errors"]["rag"] = str(exc)

    try:
        report["wiki"] = _wiki().stats()
    except Exception as exc:  # noqa: BLE001
        report["wiki"] = {"total_pages": 0}
        report["errors"]["wiki"] = str(exc)

    try:
        report["episodic"] = episodic.stats()
    except Exception as exc:  # noqa: BLE001
        report["episodic"] = {"total_episodes": 0}
        report["errors"]["episodic"] = str(exc)

    report["status"] = "degraded" if report["errors"] else "ok"
    report["config"] = {
        "data_dir": str(config.data_dir),
        "notes_dir": str(config.notes_dir),
        "wiki_dir": str(config.wiki_dir),
        "embedding_model": config.embedding_model,
        "auth_required": bool(config.api_token),
    }
    return report


# ─── Memories ──────────────────────────────────────────────────────────


@app.get("/api/memories")
def api_memories(
    query: str = "",
    category: str = "",
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    min_importance: int = Query(default=0, ge=0, le=10),
    tags: str = "",
) -> dict:
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] or None
    store = _memory()
    if query or category or min_importance or tag_list:
        results = store.recall(
            query=query,
            category=category or None,
            limit=limit,
            min_importance=min_importance,
            tags=tag_list,
        )
    else:
        results = store.all_memories(limit=limit, offset=offset)
    return {"memories": results, "total": len(results), "grand_total": store.count()}


@app.get("/api/memories/{memory_id}")
def api_get_memory(memory_id: str) -> dict:
    memory = _memory().get_by_id(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@app.post("/api/memories", status_code=201, dependencies=[Auth])
def api_create_memory(payload: MemoryIn) -> dict:
    return _memory().store(**payload.model_dump())


@app.patch("/api/memories/{memory_id}", dependencies=[Auth])
def api_update_memory(memory_id: str, payload: MemoryPatch) -> dict:
    fields = payload.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    updated = _memory().update(memory_id, **fields)
    if updated is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return updated


@app.delete("/api/memories/{memory_id}", dependencies=[Auth])
def api_delete_memory(memory_id: str) -> dict:
    if not _memory().forget(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True, "id": memory_id}


# ─── Search ────────────────────────────────────────────────────────────


@app.get("/api/search")
def api_search(
    query: str = "",
    limit: int = Query(default=10, ge=1, le=100),
    mode: Literal["semantic", "keyword", "hybrid"] = "hybrid",
) -> dict:
    if not query.strip():
        return {"results": [], "query": query, "mode": mode}

    from ..rag.engine import RagError

    engine = _rag()
    try:
        if mode == "keyword":
            results = engine.keyword_search(query, limit=limit)
        elif mode == "semantic":
            results = engine.search(query, limit=limit)
        else:
            results = engine.hybrid_search(query, limit=limit)
    except RagError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"results": results, "query": query, "mode": mode}


@app.get("/api/sources")
def api_sources(origin: str = "", limit: int = Query(default=200, ge=1, le=1000)) -> dict:
    sources = _rag().sources(origin=origin or None, limit=limit)
    return {"sources": sources, "total": len(sources)}


# ─── Wiki ──────────────────────────────────────────────────────────────


@app.get("/api/wiki")
def api_wiki(search: str = "") -> dict:
    return {"pages": _wiki().pages(search=search)}


@app.get("/api/wiki/{page_id}")
def api_wiki_page(page_id: str) -> dict:
    page = _wiki().get_page(page_id)
    if page is None:
        raise HTTPException(status_code=404, detail="Wiki page not found")
    return page


@app.get("/api/wiki-stats")
def api_wiki_stats() -> dict:
    return _wiki().stats()


@app.get("/api/wiki-lint")
def api_wiki_lint() -> dict:
    return _wiki().lint()


@app.post("/api/compile-wiki", dependencies=[Auth])
def api_compile_wiki(payload: CompileWikiIn | None = None) -> dict:
    topics = payload.topics if payload else None
    return _wiki().compile(topics=topics)


# ─── Episodic ──────────────────────────────────────────────────────────


@app.get("/api/episodes")
def api_episodes(
    session_id: str = "",
    topic: str = "",
    role: str = "",
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    from ..memory import episodic

    episodes = episodic.recall_episodes(
        session_id=session_id or None,
        topic=topic or None,
        role=role or None,
        limit=limit,
    )
    return {"episodes": episodes, "total": len(episodes)}


@app.get("/api/sessions")
def api_sessions(limit: int = Query(default=25, ge=1, le=200)) -> dict:
    from ..memory import episodic

    return {"sessions": episodic.list_sessions(limit=limit)}


# ─── Maintenance ───────────────────────────────────────────────────────


def _resolve_indexable(raw_path: str) -> Path:
    """Resolve a user-supplied path, confined to the notes/wiki directories.

    ``allow_index_outside_notes`` opts out for people who genuinely want to
    index arbitrary files from a trusted local UI.
    """
    path = Path(raw_path).expanduser().resolve()
    if config.allow_index_outside_notes:
        return path

    roots = [Path(config.notes_dir).resolve(), Path(config.wiki_dir).resolve()]
    for root in roots:
        if path == root or root in path.parents:
            return path

    raise HTTPException(
        status_code=403,
        detail=(
            "Path is outside the notes and wiki directories. "
            "Set AMNIS_ALLOW_INDEX_OUTSIDE_NOTES=true to permit it."
        ),
    )


@app.post("/api/index-file", dependencies=[Auth])
def api_index_file(payload: IndexFileIn) -> dict:
    from ..rag.engine import RagError

    path = _resolve_indexable(payload.path)
    try:
        return _rag().index_file(str(path))
    except RagError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/index-notes", dependencies=[Auth])
def api_index_notes() -> dict:
    return _rag().index_notes()


@app.post("/api/index-wiki", dependencies=[Auth])
def api_index_wiki() -> dict:
    return _rag().index_wiki()


@app.post("/api/reindex-memories", dependencies=[Auth])
def api_reindex_memories() -> dict:
    return _memory().reindex_keywords()


@app.post("/api/consolidate", dependencies=[Auth])
def api_consolidate() -> dict:
    from ..memory import consolidation

    return consolidation.run_pipeline()


@app.post("/api/prune", dependencies=[Auth])
def api_prune(payload: PruneIn | None = None) -> dict:
    from ..memory import pruning

    return pruning.run_pipeline(dry_run=payload.dry_run if payload else True)


# ─── Graph ─────────────────────────────────────────────────────────────

_CATEGORY_COLORS = {
    "preference": "#f59e0b",
    "fact": "#10b981",
    "event": "#3b82f6",
    "procedure": "#8b5cf6",
    "concept": "#ec4899",
    "theme": "#f43f5e",
    "meta": "#6b7280",
    "general": "#94a3b8",
}


@app.get("/api/graph")
def api_graph(
    limit: int = Query(default=120, ge=1, le=600),
    min_importance: int = Query(default=0, ge=0, le=10),
) -> dict:
    """Nodes and edges for the knowledge graph.

    Edges come from real relations — shared tags between memories, tag/title
    overlap with wiki pages, and a wiki page's recorded sources. The first
    version linked anything whose words happened to overlap, which produced
    a dense mesh that carried no information.
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    memories = _memory().all_memories(limit=limit)
    if min_importance:
        memories = [m for m in memories if m["importance"] >= min_importance]

    for m in memories:
        nodes.append(
            {
                "id": f"mem_{m['id']}",
                "label": m["fact"][:70] + ("…" if len(m["fact"]) > 70 else ""),
                "tooltip": m["fact"],
                "group": "memory",
                "color": _CATEGORY_COLORS.get(m["category"], "#94a3b8"),
                "weight": m["importance"],
                "meta": {
                    "category": m["category"],
                    "importance": m["importance"],
                    "id": m["id"],
                    "tags": m["tags"],
                },
            }
        )

    wiki_pages = _wiki().pages()
    for p in wiki_pages:
        nodes.append(
            {
                "id": f"wiki_{p['id']}",
                "label": p["title"][:44],
                "tooltip": f"{p['title']} — {len(p['sources'])} sources, v{p['version']}",
                "group": "wiki",
                "color": "#8b5cf6",
                "weight": 6,
                "meta": {"id": p["id"], "title": p["title"], "version": p["version"]},
            }
        )

    documents = _rag().sources(limit=limit)
    doc_by_path: dict[str, str] = {}
    for d in documents:
        node_id = f"doc_{d['path']}"
        doc_by_path[d["path"]] = node_id
        nodes.append(
            {
                "id": node_id,
                "label": Path(d["path"]).stem[:44],
                "tooltip": d["path"],
                "group": "document",
                "color": "#0ea5e9",
                "weight": 4,
                "meta": {"path": d["path"], "chunks": d["chunks"], "origin": d["origin"]},
            }
        )

    # ── memory ↔ memory: shared tags ────────────────────────────────
    # A tag is an explicit, user-authored relation, unlike incidental word
    # overlap. Very common tags are skipped: a tag on half the corpus is a
    # category, not a link, and connecting all of it produces a hairball.
    by_tag: dict[str, list[str]] = {}
    for m in memories:
        for tag in m["tags"] or []:
            by_tag.setdefault(tag.lower(), []).append(f"mem_{m['id']}")

    seen: set[tuple[str, str]] = set()
    tag_cap = max(2, len(memories) // 4)
    for tag, members in by_tag.items():
        if len(members) < 2 or len(members) > tag_cap:
            continue
        # Connect as a ring rather than a clique: same connectivity, O(n)
        # edges instead of O(n²), and the cluster still reads as a cluster.
        for a, b in zip(members, members[1:] + members[:1], strict=True):
            pair = tuple(sorted((a, b)))
            if a != b and pair not in seen:
                seen.add(pair)
                edges.append({"from": a, "to": b, "kind": "tag", "label": tag})

    # ── memory ↔ wiki: the memory's tags name the page ──────────────
    wiki_by_word: dict[str, str] = {}
    for p in wiki_pages:
        for word in p["title"].lower().replace("-", " ").split():
            if len(word) > 3:
                wiki_by_word.setdefault(word, f"wiki_{p['id']}")

    for m in memories:
        source = f"mem_{m['id']}"
        for tag in m["tags"] or []:
            target = wiki_by_word.get(tag.lower())
            if not target:
                continue
            pair = tuple(sorted((source, target)))
            if pair not in seen:
                seen.add(pair)
                edges.append({"from": source, "to": target, "kind": "topic"})

    # ── wiki → document: recorded sources ───────────────────────────
    for p in wiki_pages:
        for src in p["sources"]:
            target = doc_by_path.get(src)
            if target:
                edges.append({"from": f"wiki_{p['id']}", "to": target, "kind": "source"})

    return {
        "nodes": nodes,
        "edges": edges,
        "counts": {
            "memories": len(memories),
            "wiki": len(wiki_pages),
            "documents": len(documents),
            "edges": len(edges),
        },
    }


# ─── UI ────────────────────────────────────────────────────────────────

# Optional: drop files into amnis/server/static/ to have them served at /static.
# The UI itself needs nothing external — it is a single self-contained file.
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    global _ui_cache
    if _ui_cache is None:
        _ui_cache = _UI_PATH.read_text(encoding="utf-8")
    return HTMLResponse(_ui_cache)


def main() -> None:
    """Entry point for `amnis web`."""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"Amnis dashboard → http://{config.host}:{config.port}")
    if not config.api_token:
        print("  (no AMNIS_API_TOKEN set — mutating endpoints are unauthenticated)")
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
