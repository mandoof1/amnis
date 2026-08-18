"""Wiki compiler — structured markdown pages built from RAG + memory.

Three correctness fixes over 0.1:

* **No self-citation.** Compiled pages were indexed into the same collection
  the compiler retrieves from, so by the second run a page's top sources were
  previous versions of itself. Retrieval now excludes ``origin == "compiled"``.
* **Exact title match.** Page lookup used ``title ILIKE '%topic%'``, so
  compiling "Rust" would find and overwrite the "Rust Async Runtimes" page,
  and compiling "Go" matched almost everything. Titles are matched exactly.
* **Read and write in one session.** ``existing`` was fetched in a session
  that was then closed; the update branch mutated a detached object, so
  ``version`` never incremented and edits were dropped. Everything now happens
  inside a single ``session_scope()``.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func

from ..config import config
from ..database import IndexedDocument, MemoryFact, WikiPage, session_scope

logger = logging.getLogger(__name__)

_STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "do",
    "does",
    "did",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "of",
    "in",
    "on",
    "for",
    "to",
    "with",
    "about",
    "from",
    "my",
    "your",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
}


def _safe_filename(topic: str) -> str:
    safe = topic.lower().replace(" ", "-").replace("/", "-")
    safe = "".join(c for c in safe if c.isalnum() or c in "-_.")
    return safe[:100].strip("-") or "untitled"


class WikiCompiler:
    """Compiles wiki pages from indexed documents and stored memories."""

    def __init__(self) -> None:
        config.wiki_dir.mkdir(parents=True, exist_ok=True)

    # ─── compilation ───────────────────────────────────────────────────

    def compile(self, topics: list[str] | None = None) -> dict:
        if topics:
            pages = [self._compile_topic(t) for t in topics]
            return {"compiled": len(pages), "pages": pages}
        return self._compile_all()

    def _discover_topics(self) -> list[str]:
        with session_scope() as session:
            docs = session.query(IndexedDocument).filter(IndexedDocument.origin != "compiled").all()
            topics: set[str] = set()
            for doc in docs:
                name = (doc.title or Path(doc.path).stem).replace("-", " ").replace("_", " ")
                if name.lower().strip() in ("index", "readme", ""):
                    continue
                if not any(c.isalpha() for c in name):
                    continue
                topics.add(name.strip())

            for (cat,) in session.query(MemoryFact.category).distinct().all():
                if cat and cat != "general":
                    topics.add(cat)
        return sorted(topics)

    def _compile_all(self) -> dict:
        topics = self._discover_topics()[: config.wiki_max_pages]
        pages = [self._compile_topic(t) for t in topics]
        self._write_index(pages)
        return {"compiled": len(pages), "pages": pages}

    def _retrieve(self, topic: str, limit: int = 10) -> list[dict]:
        """Search source material only — never Amnis's own compiled output."""
        from ..rag.engine import RagError, get_engine

        try:
            return get_engine().search(topic, limit=limit, where={"origin": {"$ne": "compiled"}})
        except RagError as exc:
            logger.warning("RAG unavailable while compiling %r: %s", topic, exc)
            return []

    def _compile_topic(self, topic: str) -> dict:
        title = topic.title()
        rag_results = self._retrieve(topic)

        from ..memory.store import recall

        memories = recall(query=topic, limit=10)

        sources: list[str] = []
        parts = [
            f"# {title}\n",
            "> *Auto-compiled wiki page — last updated: "
            f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}*\n",
        ]

        if rag_results:
            parts.append("## Key Sources\n")
            for r in rag_results[:5]:
                parts.append(f"- **[{r['source']}]({r['source']})** — relevance: {r['score']:.2f}")
                parts.append(f"  > {r['content'][:200]}")
                parts.append("")
                sources.append(r["source"])

        if memories:
            parts.append("## Related Facts\n")
            for m in memories[:5]:
                parts.append(f"- [{m['category']}] {m['fact']}  ")
                parts.append(f"  *Importance: {m['importance']}/10, source: {m['source']}*")
                parts.append("")

        tags: set[str] = set()
        for r in rag_results:
            stem = Path(r["source"]).stem if r.get("source") else ""
            if stem:
                tags.add(stem.replace("-", " ").replace("_", " "))
        for m in memories:
            tags.update(m.get("tags") or [])

        related = sorted(t for t in tags if t.lower() != topic.lower())[:10]
        if related:
            parts.append("## Related Topics\n")
            parts.extend(f"- [[{r.title()}]]" for r in related)
            parts.append("")

        content = "\n".join(parts)
        sources = sorted(set(sources))

        with session_scope() as session:
            existing = session.query(WikiPage).filter(WikiPage.title == title).first()
            if existing is not None:
                existing.content = content
                existing.summary = content[:200]
                existing.sources = sources
                existing.related = related
                existing.last_compiled = datetime.now(UTC)
                existing.version = (existing.version or 1) + 1
                page_id, version = existing.id, existing.version
            else:
                page = WikiPage(
                    id=str(uuid.uuid4()),
                    title=title,
                    content=content,
                    summary=content[:200],
                    sources=sources,
                    related=related,
                    tags=[topic.lower()],
                    last_compiled=datetime.now(UTC),
                    version=1,
                )
                session.add(page)
                page_id, version = page.id, 1

        wiki_file = config.wiki_dir / f"{_safe_filename(topic)}.md"
        try:
            wiki_file.write_text(content, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write %s: %s", wiki_file, exc)

        return {
            "id": page_id,
            "title": title,
            "version": version,
            "sources": len(sources),
            "memories": len(memories),
            "related": len(related),
            "file": str(wiki_file),
        }

    def _write_index(self, pages: list[dict]) -> None:
        lines = [
            "# Amnis Wiki Index\n",
            f"> Auto-generated index — {len(pages)} pages\n",
            f"> Last compiled: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}\n",
            "---\n",
            "## Pages\n",
        ]
        for p in sorted(pages, key=lambda x: x["title"]):
            lines.append(f"- [[{p['title']}]] — {p['sources']} sources, {p['memories']} related memories")
        lines += ["", "---\n", f"*Amnis Wiki Compiler v{_version()}*\n"]
        try:
            (config.wiki_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write wiki index: %s", exc)

    # ─── inspection ────────────────────────────────────────────────────

    def pages(self, search: str = "", limit: int = 500) -> list[dict]:
        from ..memory.store import escape_like

        with session_scope() as session:
            q = session.query(WikiPage)
            if search:
                like = f"%{escape_like(search)}%"
                q = q.filter(WikiPage.title.ilike(like, escape="\\"))
            rows = q.order_by(WikiPage.title).limit(limit).all()
            return [
                {
                    "id": p.id,
                    "title": p.title,
                    "summary": (p.summary or "")[:200],
                    "sources": p.sources or [],
                    "related": p.related or [],
                    "tags": p.tags or [],
                    "last_compiled": p.last_compiled.isoformat() if p.last_compiled else None,
                    "version": p.version,
                }
                for p in rows
            ]

    def get_page(self, page_id: str) -> dict | None:
        with session_scope() as session:
            p = session.query(WikiPage).filter(WikiPage.id == page_id).first()
            if p is None:
                return None
            return {
                "id": p.id,
                "title": p.title,
                "content": p.content,
                "summary": p.summary,
                "sources": p.sources or [],
                "related": p.related or [],
                "tags": p.tags or [],
                "last_compiled": p.last_compiled.isoformat() if p.last_compiled else None,
                "version": p.version,
            }

    def lint(self) -> dict:
        """Report stale pages, sourceless pages, and duplicate titles."""
        issues: list[dict] = []
        now = datetime.now(UTC)

        with session_scope() as session:
            pages = session.query(WikiPage).all()
            for page in pages:
                if page.last_compiled:
                    age = (now - page.last_compiled).days
                    if age > 30:
                        issues.append(
                            {
                                "page": page.title,
                                "issue": "stale",
                                "detail": f"Last compiled {age} days ago",
                            }
                        )
                if not page.sources:
                    issues.append(
                        {
                            "page": page.title,
                            "issue": "no_sources",
                            "detail": "Page has no source references",
                        }
                    )

            duplicates = (
                session.query(WikiPage.title, func.count(WikiPage.id))
                .group_by(WikiPage.title)
                .having(func.count(WikiPage.id) > 1)
                .all()
            )
            for title, n in duplicates:
                issues.append(
                    {
                        "page": title,
                        "issue": "duplicate_title",
                        "detail": f"{n} pages share this title (pre-0.2 data)",
                    }
                )

            checked = len(pages)

        return {"pages_checked": checked, "issues_found": len(issues), "issues": issues}

    def query(self, question: str) -> dict:
        """Answer a question from wiki pages plus RAG hits.

        Matching is per-term. The old implementation searched for the whole
        question as one substring, so anything phrased as a sentence matched
        nothing.
        """
        from ..memory.store import escape_like

        terms = [t for t in re.findall(r"[\w']+", question.lower()) if len(t) > 2 and t not in _STOP_WORDS][
            :8
        ]

        rag_results = self._retrieve(question, limit=5)

        with session_scope() as session:
            q = session.query(WikiPage)
            if terms:
                from sqlalchemy import or_

                clauses = []
                for t in terms:
                    like = f"%{escape_like(t)}%"
                    clauses.append(WikiPage.title.ilike(like, escape="\\"))
                    clauses.append(WikiPage.content.ilike(like, escape="\\"))
                q = q.filter(or_(*clauses))
            matches = q.limit(5).all()
            wiki_pages = [
                {
                    "id": w.id,
                    "title": w.title,
                    "summary": w.summary,
                    "sources": w.sources or [],
                }
                for w in matches
            ]

        return {
            "question": question,
            "terms": terms,
            "rag_results": rag_results[:3],
            "wiki_pages": wiki_pages,
        }

    def stats(self) -> dict:
        """Page count and *distinct* source count.

        0.1 reported ``query(WikiPage.sources).count()`` as "total_sources",
        which is just the number of rows — always equal to total_pages.
        """
        with session_scope() as session:
            total_pages = session.query(func.count(WikiPage.id)).scalar() or 0
            all_sources: set[str] = set()
            for (sources,) in session.query(WikiPage.sources).all():
                all_sources.update(sources or [])
            newest = session.query(func.max(WikiPage.last_compiled)).scalar()
            avg_version = session.query(func.avg(WikiPage.version)).scalar() or 0

        return {
            "total_pages": int(total_pages),
            "total_sources": len(all_sources),
            "avg_version": round(float(avg_version), 1),
            "last_compiled": newest.isoformat() if newest else None,
            "wiki_dir": str(config.wiki_dir),
        }


def _version() -> str:
    from .. import __version__

    return __version__


# ─── Lazy singleton ────────────────────────────────────────────────────

_compiler: WikiCompiler | None = None


def get_compiler() -> WikiCompiler:
    global _compiler
    if _compiler is None:
        _compiler = WikiCompiler()
    return _compiler


def reset_compiler() -> None:
    global _compiler
    _compiler = None


def __getattr__(name: str):
    if name == "compiler":
        return get_compiler()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
