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
    """Store a memory fact. Returns the created record."""
    session = get_session()
    try:
        entry = MemoryFact(
            id=str(uuid.uuid4()),
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
