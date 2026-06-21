"""Wiki Compiler — Karpathy-style structured knowledge from indexed sources.

Reads indexed documents and memory, compiles structured markdown wiki pages
with cross-references, summaries, and relationship mapping.
"""
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import config
from ..database import get_session, WikiPage, IndexedDocument, MemoryFact
from ..rag.engine import engine as rag_engine


class WikiCompiler:
    """Compiles structured wiki pages from indexed documents and memory sources."""

    def __init__(self):
        config.wiki_dir.mkdir(parents=True, exist_ok=True)

    def compile(self, topics: Optional[list[str]] = None) -> dict:
        """Compile wiki pages. If topics given, compile specific ones.
        
        For each topic:
          1. Search RAG for relevant content
          2. Search memory for relevant facts
          3. Compile into structured markdown page
          4. Cross-reference with existing pages
        """
        if topics:
            results = []
            for topic in topics:
                page = self._compile_topic(topic)
                results.append(page)
            return {"compiled": len(results), "pages": results}
        else:
            # Auto-detect topics from indexed documents
            return self._compile_all()

    def _compile_all(self) -> dict:
        """Auto-discover topics and compile all wiki pages."""
        # Get all indexed document titles
        session = get_session()
        try:
            docs = session.query(IndexedDocument).all()
            # Extract topic candidates from titles and paths
            topics = set()
            for doc in docs:
                name = doc.title or Path(doc.path).stem
                # Clean up filename to get topic
                name = name.replace("-", " ").replace("_", " ")
                # Skip common non-topics
                if name.lower() in ("index", "readme", ""):
                    continue
                if not any(c.isalpha() for c in name):
                    continue
                topics.add(name)

            # Also get memory categories as topics
            categories = session.query(MemoryFact.category).distinct().all()
            for (cat,) in categories:
                if cat and cat not in ("general",):
                    topics.add(cat)

        finally:
            session.close()

        # Compile each topic
        results = []
        for topic in sorted(topics)[:config.wiki_max_pages]:
            page = self._compile_topic(topic)
            results.append(page)

        # Write all to wiki directory
        self._write_index(results)

        return {"compiled": len(results), "pages": results}

    def _compile_topic(self, topic: str) -> dict:
        """Compile a single wiki page for a topic."""
        # Sanitize topic for filename use
        safe_name = topic.lower().replace(" ", "-").replace("/", "-")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "-_.")
        safe_name = safe_name[:100]  # cap filename length to avoid FS errors
        # Search RAG for content about this topic
        rag_results = rag_engine.search(topic, limit=10)

        # Search memory for facts about this topic
        from ..memory.store import recall
        memories = recall(query=topic, limit=10)

        # Get existing page if any
        session = get_session()
        try:
            existing = (
                session.query(WikiPage)
                .filter(WikiPage.title.ilike(f"%{topic}%"))
                .first()
            )
        finally:
            session.close()

        # Compile content
        sources = []
        content_parts = [f"# {topic.title()}\n"]
        content_parts.append(f"> *Auto-compiled wiki page — last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n")

        # RAG findings section
        if rag_results:
            content_parts.append("## Key Sources\n")
            for r in rag_results[:5]:
                content_parts.append(f"- **[{r['source']}]({r['source']})** — relevance: {r['score']:.2f}")
                content_parts.append(f"  > {r['content'][:200]}")
                sources.append(r["source"])
                content_parts.append("")

        # Memory facts section
        if memories:
            content_parts.append("## Related Facts\n")
            for m in memories[:5]:
                content_parts.append(f"- [{m['category']}] {m['fact']}  ")
                content_parts.append(f"  *Importance: {m['importance']}/10, source: {m['source']}*")
                content_parts.append("")

        # Find related topics
        all_tags = set()
        for r in rag_results:
            src = Path(r["source"]).stem if r.get("source") else ""
            if src:
                all_tags.add(src.replace("-", " ").replace("_", " "))
        for m in memories:
            if m.get("tags"):
                all_tags.update(m["tags"])

        related = [t for t in all_tags if t.lower() != topic.lower()][:10]
        if related:
            content_parts.append("## Related Topics\n")
            for r in related:
                content_parts.append(f"- [[{r.title()}]]")
            content_parts.append("")

        content = "\n".join(content_parts)

        # Save to DB
        session = get_session()
        try:
            if existing:
                existing.content = content
                existing.sources = sources
                existing.related = related
                existing.last_compiled = datetime.now(timezone.utc)
                existing.version += 1
            else:
                page = WikiPage(
                    id=str(uuid.uuid4()),
                    title=topic.title(),
                    content=content,
                    summary=content[:200],
                    sources=sources,
                    related=related,
                    tags=[topic.lower()],
                    last_compiled=datetime.now(timezone.utc),
                )
                session.add(page)
            session.commit()
        finally:
            session.close()

        # Write to wiki directory — use pre-sanitized safe_name from above
        wiki_file = config.wiki_dir / f"{safe_name}.md"
        wiki_file.write_text(content, encoding="utf-8")

        return {
            "title": topic.title(),
            "sources": len(sources),
            "memories": len(memories),
            "related": len(related),
            "file": str(wiki_file),
        }

    def _write_index(self, pages: list[dict]):
        """Write a master index of all wiki pages."""
        lines = [
            "# Amnis Wiki Index\n",
            f"> Auto-generated index — {len(pages)} pages\n",
            f"> Last compiled: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n",
            "---\n",
            "## Pages\n",
        ]
        for p in sorted(pages, key=lambda x: x["title"]):
            safe = p["title"].lower().replace(" ", "-").replace("/", "-")
            lines.append(f"- [[{p['title']}]] — {p['sources']} sources, {p['memories']} related memories")
        lines.append("")
        lines.append("---\n")
        lines.append("*Amnis Wiki Compiler v0.1.0*\n")

        (config.wiki_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")

    def lint(self) -> dict:
        """Check wiki for issues: missing pages, stale content, broken refs."""
        issues = []
        session = get_session()
        try:
            pages = session.query(WikiPage).all()
            for page in pages:
                # Check for stale content (older than 30 days)
                age = (datetime.now(timezone.utc) - page.last_compiled).days
                if age > 30:
                    issues.append({
                        "page": page.title,
                        "issue": "stale",
                        "detail": f"Last compiled {age} days ago",
                    })

                # Check for missing sources
                if not page.sources:
                    issues.append({
                        "page": page.title,
                        "issue": "no_sources",
                        "detail": "Page has no source references",
                    })
        finally:
            session.close()

        return {
            "pages_checked": len(pages) if 'pages' in dir() else 0,
            "issues_found": len(issues),
            "issues": issues,
        }

    def query(self, question: str) -> dict:
        """Ask the wiki a question — searches across all pages."""
        # First try RAG
        rag_results = rag_engine.search(question, limit=5)

        # Then search wiki pages in DB
        session = get_session()
        try:
            like = f"%{question}%"
            wiki_matches = (
                session.query(WikiPage)
                .filter(
                    WikiPage.content.ilike(like) | WikiPage.title.ilike(like)
                )
                .limit(5)
                .all()
            )
        finally:
            session.close()

        return {
            "question": question,
            "rag_results": rag_results[:3],
            "wiki_pages": [
                {"title": w.title, "summary": w.summary, "sources": w.sources}
                for w in wiki_matches
            ],
        }

    def stats(self) -> dict:
        """Get wiki statistics."""
        session = get_session()
        try:
            count = session.query(WikiPage).count()
            total_sources = session.query(WikiPage.sources).count()
            return {
                "total_pages": count,
                "total_sources": total_sources,
                "wiki_dir": str(config.wiki_dir),
            }
        finally:
            session.close()


compiler = WikiCompiler()
