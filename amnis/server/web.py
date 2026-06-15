"""Amnis Web UI — FastAPI server for the interactive knowledge dashboard.

Run with:
  uvicorn amnis.server.web:app --host 127.0.0.1 --port 8799

Or:
  python -m amnis.server.web
"""
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path.home() / "amnis"))

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from ..memory import store as memory_store
from ..memory import consolidation as memory_consolidation
from ..rag.engine import engine as rag_engine
from ..wiki.compiler import compiler as wiki_compiler
from ..config import config

app = FastAPI(title="Amnis Web UI", version="0.1.0")


@app.on_event("startup")
async def startup():
    """Pre-load the embedding model so first search isn't slow."""
    try:
        # Warm up the embedding model
        _ = rag_engine.embedder.encode(["warmup"])
        print("✅ Embedding model loaded")
    except Exception as e:
        print(f"⚠️ Could not pre-load model: {e}")


# ─── API Endpoints ─────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    """Overall system status."""
    mem_stats = memory_store.stats()
    try:
        rag_stats = rag_engine.stats()
    except Exception:
        rag_stats = {"total_chunks": 0, "unique_sources": 0}
    wiki_stats = wiki_compiler.stats()
    return {
        "memory": mem_stats,
        "rag": rag_stats,
        "wiki": wiki_stats,
        "config": {
            "vault": str(config.vault_path),
            "data_dir": str(config.data_dir),
            "embedding_model": config.embedding_model,
        },
    }


@app.get("/api/memories")
async def api_memories(
    query: str = "",
    category: str = "",
    limit: int = 50,
    offset: int = 0,
    min_importance: int = 0,
):
    """Get memories with optional filters."""
    if query:
        results = memory_store.recall(query=query, limit=limit, min_importance=min_importance)
    elif category:
        results = memory_store.recall(category=category, limit=limit, min_importance=min_importance)
    else:
        results = memory_store.all_memories(limit=limit, offset=offset)
    return {"memories": results, "total": len(results)}


@app.post("/api/memories")
async def api_create_memory(data: dict):
    """Create a new memory."""
    required = memory_store.store(
        fact=data["fact"],
        category=data.get("category", "general"),
        importance=data.get("importance", 5),
        source=data.get("source", "web-ui"),
        tags=data.get("tags"),
        context=data.get("context"),
    )
    return required


@app.delete("/api/memories/{memory_id}")
async def api_delete_memory(memory_id: str):
    """Delete a memory."""
    success = memory_store.forget(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True}


@app.get("/api/search")
async def api_search(query: str = "", limit: int = 10):
    """RAG search over indexed documents."""
    if not query:
        return {"results": []}
    results = rag_engine.search(query=query, limit=limit)
    return {"results": results, "query": query}


@app.get("/api/wiki")
async def api_wiki(search: str = ""):
    """Get wiki pages, optionally filtered by search."""
    from ..database import get_session, WikiPage
    session = get_session()
    try:
        if search:
            q = session.query(WikiPage).filter(WikiPage.title.ilike(f"%{search}%"))
        else:
            q = session.query(WikiPage).order_by(WikiPage.title)
        pages = []
        for p in q.all():
            pages.append({
                "id": p.id,
                "title": p.title,
                "summary": p.summary[:200] if p.summary else "",
                "sources": p.sources or [],
                "related": p.related or [],
                "last_compiled": p.last_compiled.isoformat() if p.last_compiled else None,
                "version": p.version,
            })
        return {"pages": pages}
    finally:
        session.close()


@app.get("/api/wiki/{page_id}")
async def api_wiki_page(page_id: str):
    """Get a single wiki page by ID."""
    from ..database import get_session, WikiPage
    session = get_session()
    try:
        page = session.query(WikiPage).filter(WikiPage.id == page_id).first()
        if not page:
            raise HTTPException(status_code=404, detail="Wiki page not found")
        return {
            "id": page.id,
            "title": page.title,
            "content": page.content,
            "summary": page.summary,
            "sources": page.sources or [],
            "related": page.related or [],
            "tags": page.tags or [],
            "last_compiled": page.last_compiled.isoformat() if page.last_compiled else None,
            "version": page.version,
        }
    finally:
        session.close()


@app.get("/api/graph")
async def api_graph():
    """Build graph data for vis.js visualization.

    Nodes: memories, wiki pages, indexed documents
    Edges: relationships between them
    """
    nodes = []
    edges = []
    seen_ids = set()

    # ── Memory Nodes ──
    memories = memory_store.all_memories(limit=100)
    for m in memories:
        nid = f"mem_{m['id'][:8]}"
        if nid in seen_ids:
            continue
        seen_ids.add(nid)
        color = _category_color(m["category"])
        nodes.append({
            "id": nid,
            "label": m["fact"][:40] + ("..." if len(m["fact"]) > 40 else ""),
            "title": f"⭐{m['importance']} [{m['category']}]\n{m['fact']}",
            "group": "memory",
            "color": color,
            "size": 10 + m["importance"] * 2,
            "shape": "dot",
            "font": {"size": 10},
            "data": m,
        })

    # ── Wiki Nodes ──
    from ..database import get_session, WikiPage
    session = get_session()
    try:
        wiki_pages = session.query(WikiPage).all()
        for wp in wiki_pages:
            nid = f"wiki_{wp.id[:8]}"
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            nodes.append({
                "id": nid,
                "label": wp.title[:35],
                "title": f"📝 {wp.title}\nSources: {len(wp.sources or [])}",
                "group": "wiki",
                "color": "#8b5cf6",
                "size": 15,
                "shape": "box",
                "font": {"size": 11, "color": "#e2e8f0"},
                "data": {"title": wp.title, "id": wp.id, "sources": wp.sources},
            })

            # Edge: wiki page → related topics
            for rel in (wp.related or []):
                pass
    finally:
        session.close()

    # ── Document Nodes (from RAG) ──
    try:
        rag_stats = rag_engine.stats()
        # Add document-level nodes from indexed files
        all_data = rag_engine.collection.get(include=["metadatas"])
        doc_sources = set()
        if all_data and all_data.get("metadatas"):
            for m in all_data["metadatas"]:
                src = m.get("source", "")
                if src and src not in doc_sources:
                    doc_sources.add(src)
                    nid = f"doc_{Path(src).stem[:20]}"
                    if nid in seen_ids:
                        continue
                    seen_ids.add(nid)
                    nodes.append({
                        "id": nid,
                        "label": Path(src).stem[:30],
                        "title": f"📄 {src}",
                        "group": "document",
                        "color": "#0ea5e9",
                        "size": 8,
                        "shape": "square",
                        "font": {"size": 9},
                    })
    except Exception:
        pass

    # ── Edges between memories and wiki/docs ──
    # Connect memories to related wiki pages by keyword matching
    for m in memories:
        mem_id = f"mem_{m['id'][:8]}"
        mid = mem_id
        for n in nodes:
            if n["id"] == mid:
                continue
            # Simple: connect if memory fact text appears in wiki/doc label
            kw = m["fact"].lower().split()[:3]
            nlabel = n.get("label", "").lower()
            if any(k in nlabel for k in kw if len(k) > 3):
                edges.append({
                    "from": mid,
                    "to": n["id"],
                    "color": {"color": "rgba(139, 92, 246, 0.3)"},
                    "width": 1,
                    "title": "keyword match",
                })
                break

    # Connect wiki pages to their source documents
    session2 = get_session()
    try:
        wiki_pages2 = session2.query(WikiPage).all()
        for wp in wiki_pages2:
            wid = f"wiki_{wp.id[:8]}"
            for src in (wp.sources or []):
                src_stem = Path(src).stem[:20]
                doc_id = f"doc_{src_stem}"
                if doc_id in seen_ids:
                    edges.append({
                        "from": wid,
                        "to": doc_id,
                        "color": {"color": "rgba(14, 165, 233, 0.3)"},
                        "width": 1,
                        "title": "source",
                    })
    finally:
        session2.close()

    return {"nodes": nodes, "edges": edges}


@app.get("/api/consolidate")
async def api_consolidate(limit: int = 50):
    """Run memory consolidation."""
    result = memory_consolidation.run_pipeline()
    return result


@app.post("/api/index-file")
async def api_index_file(data: dict):
    """Index a file."""
    path = data.get("path", "")
    if not path:
        raise HTTPException(status_code=400, detail="path required")
    result = rag_engine.index_file(path)
    return result


@app.post("/api/index-vault")
async def api_index_vault():
    """Index entire vault."""
    result = rag_engine.index_vault()
    return result


@app.post("/api/compile-wiki")
async def api_compile_wiki(data: dict = None):
    """Compile wiki pages."""
    topics = data.get("topics") if data else None
    result = wiki_compiler.compile(topics=topics)
    return result


@app.get("/api/wiki-stats")
async def api_wiki_stats():
    """Wiki statistics."""
    return wiki_compiler.stats()


# ─── Category Colors ───────────────────────────────────────────────

_CATEGORY_COLORS = {
    "preference": "#f59e0b",
    "fact": "#10b981",
    "event": "#3b82f6",
    "procedure": "#8b5cf6",
    "concept": "#ec4899",
    "meta": "#6b7280",
    "general": "#94a3b8",
}


def _category_color(cat: str) -> str:
    return _CATEGORY_COLORS.get(cat, "#94a3b8")


# ─── Serve the UI ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(_get_ui_html())


_UI_HTML_CACHE: str | None = None

_UI_HTML_PATH = Path(__file__).parent / "ui.html"


def _get_ui_html() -> str:
    """Lazy-load the UI HTML from the embedded file."""
    global _UI_HTML_CACHE
    if _UI_HTML_CACHE is None:
        _UI_HTML_CACHE = _UI_HTML_PATH.read_text(encoding="utf-8")
    return _UI_HTML_CACHE


# ─── Main ──────────────────────────────────────────────────────────

def main():
    """Start the Amnis Web UI server."""
    print(f"☀️ Amnis Web UI starting on http://{config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# UI HTML is loaded from server/ui.html

