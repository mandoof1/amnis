"""Amnis memory store — CRUD for persistent facts across sessions.

Categories:
  - preference: LO's likes, dislikes, habits
  - fact: things learned about LO, his setup, his work
  - event: things that happened
  - procedure: how to do things
  - concept: domain knowledge
  - meta: about Amnis itself
"""
import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import func, or_, desc

from ..database import get_session, MemoryFact
from ..config import config


def _write_wiki_entry(fact: str, category: str, importance: int, tags: list, memory_id: str):
    """Write a wiki markdown file for a stored memory.

    Creates/updates a page in data/wiki/facts/ so every memory has a
    durable markdown home. Also indexes into ChromaDB immediately so
    the memory is searchable via semantic/keyword search.
    """
    import re
    # Sanitize the fact into a filename fragment
    safe = re.sub(r"[^a-zA-Z0-9\s-]", "", fact.lower())
    safe = re.sub(r"\s+", "-", safe.strip())[:60]
    safe = safe.rstrip("-") or "memory"

    # Use the memory ID suffix to avoid collisions
    short_id = memory_id[:8]
    filename = f"{safe}-{short_id}.md"
    wiki_path = config.wiki_facts_dir / filename

    # Build the wiki content
    tag_str = ", ".join(tags) if tags else "none"
    content = (
        f"# Memory: {fact[:120]}\n\n"
        f"> *Auto-created wiki entry — synced from memory store*\n\n"
        f"## Fact\n"
        f"{fact}\n\n"
        f"## Metadata\n\n"
        f"- **Category:** {category}\n"
        f"- **Importance:** {importance}/10\n"
        f"- **Tags:** {tag_str}\n"
        f"- **Memory ID:** {memory_id}\n"
        f"- **Created:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"## Related Topics\n\n"
        f"- [[Index]]\n"
    )

    try:
        config.wiki_facts_dir.mkdir(parents=True, exist_ok=True)
        wiki_path.write_text(content, encoding="utf-8")

        # Auto-index into ChromaDB for immediate semantic recall
        from ..rag.engine import engine as rag_engine
        rag_engine.index_file(str(wiki_path))
    except Exception:
        pass  # Wiki write + index failure shouldn't block memory storage


def store(
    fact: str,
    category: str = "general",
    importance: int = 5,
    source: str = "manual",
    tags: Optional[list] = None,
    context: Optional[str] = None,
    confidence: float = 1.0,
    expiry: Optional[str] = None,
) -> dict:
    """Store a memory fact. Also creates a wiki entry automatically."""
    session = get_session()
    try:
        entry_id = str(uuid.uuid4())
        entry = MemoryFact(
            id=entry_id,
            fact=fact,
            category=category,
            importance=max(1, min(10, importance)),
            source=source,
            tags=tags or [],
            context=context,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            expiry=datetime.fromisoformat(expiry) if expiry else None,
        )
        session.add(entry)
        session.commit()

        # Auto-create wiki entry for this memory
        _write_wiki_entry(
            fact=fact,
            category=category,
            importance=entry.importance,
            tags=entry.tags or [],
            memory_id=entry_id,
        )

        return _fact_to_dict(entry)
    finally:
        session.close()


def recall(
    query: str = "",
    category: Optional[str] = None,
    limit: int = 10,
    min_importance: int = 0,
    tags: Optional[list] = None,
) -> list[dict]:
    """Recall memories. Supports keyword search, category filter, tag filter."""
    session = get_session()
    try:
        q = session.query(MemoryFact)

        # Text search
        if query:
            like = f"%{query}%"
            q = q.filter(
                or_(
                    MemoryFact.fact.ilike(like),
                    MemoryFact.context.ilike(like),
                )
            )

        # Category filter
        if category:
            q = q.filter(MemoryFact.category == category)

        # Importance filter
        if min_importance > 0:
            q = q.filter(MemoryFact.importance >= min_importance)

        # Exclude expired
        q = q.filter(
            or_(
                MemoryFact.expiry.is_(None),
                MemoryFact.expiry > datetime.now(timezone.utc),
            )
        )

        # Order by importance + recency
        q = q.order_by(
            desc(MemoryFact.importance),
            desc(MemoryFact.timestamp),
        )

        results = q.limit(limit).all()

        # Filter by tags in Python (JSON field)
        if tags:
            filtered = []
            for r in results:
                r_tags = r.tags or []
                if any(t.lower() in [rt.lower() for rt in r_tags] for t in tags):
                    filtered.append(r)
            results = filtered

        # Update access counts
        for r in results:
            r.access_count += 1
            r.last_accessed = datetime.now(timezone.utc)
        session.commit()

        return [_fact_to_dict(r) for r in results]
    finally:
        session.close()


def forget(memory_id: str) -> bool:
    """Delete a specific memory by ID."""
    session = get_session()
    try:
        entry = session.query(MemoryFact).filter(MemoryFact.id == memory_id).first()
        if entry:
            session.delete(entry)
            session.commit()
            return True
        return False
    finally:
        session.close()


def get_by_id(memory_id: str) -> Optional[dict]:
    """Get a single memory by ID."""
    session = get_session()
    try:
        entry = session.query(MemoryFact).filter(MemoryFact.id == memory_id).first()
        if entry:
            return _fact_to_dict(entry)
        return None
    finally:
        session.close()


def clear_expired() -> int:
    """Remove all expired memories. Returns count deleted."""
    session = get_session()
    try:
        count = (
            session.query(MemoryFact)
            .filter(MemoryFact.expiry < datetime.now(timezone.utc))
            .delete()
        )
        session.commit()
        return count
    finally:
        session.close()


def stats() -> dict:
    """Get memory store statistics."""
    session = get_session()
    try:
        total = session.query(MemoryFact).count()
        by_category = (
            session.query(MemoryFact.category, func.count())
            .group_by(MemoryFact.category)
            .all()
        )
        avg_importance = (
            session.query(func.avg(MemoryFact.importance)).scalar() or 0
        )
        expired = (
            session.query(MemoryFact)
            .filter(MemoryFact.expiry < datetime.now(timezone.utc))
            .count()
        )
        return {
            "total_memories": total,
            "by_category": {cat: cnt for cat, cnt in by_category},
            "avg_importance": round(float(avg_importance), 1),
            "expired": expired,
        }
    finally:
        session.close()


def all_memories(limit: int = 50, offset: int = 0) -> list[dict]:
    """List all memories with pagination."""
    session = get_session()
    try:
        results = (
            session.query(MemoryFact)
            .order_by(desc(MemoryFact.importance), desc(MemoryFact.timestamp))
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [_fact_to_dict(r) for r in results]
    finally:
        session.close()


def _fact_to_dict(fact: MemoryFact) -> dict:
    return {
        "id": fact.id,
        "fact": fact.fact,
        "category": fact.category,
        "importance": fact.importance,
        "source": fact.source,
        "tags": fact.tags or [],
        "context": fact.context,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp.isoformat() if fact.timestamp else None,
        "last_accessed": fact.last_accessed.isoformat() if fact.last_accessed else None,
        "access_count": fact.access_count,
        "expiry": fact.expiry.isoformat() if fact.expiry else None,
    }
