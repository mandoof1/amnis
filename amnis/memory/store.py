"""Persistent fact storage — CRUD plus hybrid recall.

Categories: preference, fact, event, procedure, concept, theme, meta, general.

The headline change from 0.1 is recall. It used to be a single
``fact ILIKE '%<whole query>%'``, so "architecture memory" only matched text
containing that exact phrase — a query of more than one word almost always
returned nothing. Recall now fuses an FTS5 keyword index with ChromaDB
semantic search via RRF, and falls back to (properly escaped) LIKE only when
neither index has been populated yet.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import UTC, datetime

from sqlalchemy import desc, func, or_

from ..config import config
from ..database import MemoryFact, session_scope
from ..rag.keyword import KeywordIndex, rrf_fuse

logger = logging.getLogger(__name__)

# A second FTS5 table in the same file as the chunk index, with its own columns.
memory_index = KeywordIndex(
    table="memory_index",
    columns=("content", "memory_id", "category", "tags"),
)

VALID_CATEGORIES = (
    "preference",
    "fact",
    "event",
    "procedure",
    "concept",
    "theme",
    "meta",
    "general",
)


def escape_like(value: str) -> str:
    """Escape LIKE wildcards so user input cannot become a pattern.

    Without this, recalling ``100%`` or ``some_thing`` silently matched far
    more rows than the user asked for.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# ─── Wiki mirror ───────────────────────────────────────────────────────


def _wiki_filename(fact: str, memory_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9\s-]", "", fact.lower())
    safe = re.sub(r"\s+", "-", safe.strip())[:60].rstrip("-") or "memory"
    return f"{safe}-{memory_id[:8]}.md"


def _write_wiki_entry(
    fact: str,
    category: str,
    importance: int,
    tags: list,
    memory_id: str,
) -> str | None:
    """Mirror a memory to a markdown file and index it. Returns the path.

    Failures are logged rather than swallowed by a bare ``except: pass`` —
    a memory whose wiki mirror silently failed to write looked identical to
    one that succeeded.
    """
    wiki_path = config.wiki_facts_dir / _wiki_filename(fact, memory_id)
    tag_str = ", ".join(tags) if tags else "none"
    content = (
        f"# Memory: {fact[:120]}\n\n"
        f"> *Auto-created wiki entry — synced from the memory store*\n\n"
        f"## Fact\n"
        f"{fact}\n\n"
        f"## Metadata\n\n"
        f"- **Category:** {category}\n"
        f"- **Importance:** {importance}/10\n"
        f"- **Tags:** {tag_str}\n"
        f"- **Memory ID:** {memory_id}\n"
        f"- **Created:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"## Related Topics\n\n"
        f"- [[Index]]\n"
    )

    try:
        config.wiki_facts_dir.mkdir(parents=True, exist_ok=True)
        wiki_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write wiki entry for memory %s: %s", memory_id, exc)
        return None

    try:
        from ..rag.engine import get_engine

        get_engine().index_file(str(wiki_path), origin="memory", extra_metadata={"memory_id": memory_id})
    except Exception as exc:  # noqa: BLE001 - indexing is best-effort
        logger.warning("Could not index wiki entry for memory %s: %s", memory_id, exc)

    return str(wiki_path)


def _index_memory_keywords(entry_id: str, fact: str, category: str, tags: list) -> None:
    memory_index.remove_where("memory_id", entry_id)
    memory_index.add(
        [
            {
                "content": fact,
                "memory_id": entry_id,
                "category": category,
                "tags": " ".join(tags or []),
            }
        ]
    )


def _purge_memory_artifacts(memory_id: str, wiki_path: str | None) -> None:
    """Remove every trace of a memory outside the SQL row itself."""
    try:
        memory_index.remove_where("memory_id", memory_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not de-index memory %s: %s", memory_id, exc)

    if not wiki_path:
        return

    try:
        from ..rag.engine import get_engine

        get_engine().delete_source(wiki_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not remove RAG chunks for %s: %s", wiki_path, exc)

    try:
        from pathlib import Path

        Path(wiki_path).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete wiki file %s: %s", wiki_path, exc)


# ─── CRUD ──────────────────────────────────────────────────────────────


def store(
    fact: str,
    category: str = "general",
    importance: int = 5,
    source: str = "manual",
    tags: list | None = None,
    context: str | None = None,
    confidence: float = 1.0,
    expiry: str | None = None,
) -> dict:
    """Store a fact, mirror it to the wiki, and index it for recall."""
    fact = (fact or "").strip()
    if not fact:
        raise ValueError("fact must not be empty")

    entry_id = str(uuid.uuid4())
    tags = tags or []
    importance = max(1, min(10, importance))

    expiry_dt = None
    if expiry:
        try:
            expiry_dt = datetime.fromisoformat(expiry)
        except ValueError as exc:
            raise ValueError(f"expiry must be ISO-8601, got {expiry!r}") from exc
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=UTC)

    wiki_path = _write_wiki_entry(fact, category, importance, tags, entry_id)

    with session_scope() as session:
        entry = MemoryFact(
            id=entry_id,
            fact=fact,
            category=category,
            importance=importance,
            source=source,
            tags=tags,
            context=context,
            confidence=confidence,
            timestamp=datetime.now(UTC),
            last_accessed=datetime.now(UTC),
            access_count=0,
            expiry=expiry_dt,
            wiki_path=wiki_path,
        )
        session.add(entry)
        session.flush()
        result = _fact_to_dict(entry)

    _index_memory_keywords(entry_id, fact, category, tags)
    return result


def update(memory_id: str, **fields) -> dict | None:
    """Update a memory in place, preserving its ID, history, and counters.

    The web UI used to "edit" by DELETE-then-POST, which minted a new ID and
    reset access_count/timestamp — so editing a typo destroyed the memory's
    provenance and broke every reference to it.
    """
    allowed = {
        "fact",
        "category",
        "importance",
        "source",
        "tags",
        "context",
        "confidence",
        "expiry",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unknown field(s): {', '.join(sorted(unknown))}")

    with session_scope() as session:
        entry = session.query(MemoryFact).filter(MemoryFact.id == memory_id).first()
        if entry is None:
            return None

        for key, value in fields.items():
            if value is None:
                continue
            if key == "importance":
                value = max(1, min(10, int(value)))
            elif key == "expiry":
                value = datetime.fromisoformat(value) if isinstance(value, str) else value
                if value is not None and value.tzinfo is None:
                    value = value.replace(tzinfo=UTC)
            setattr(entry, key, value)

        result = _fact_to_dict(entry)
        fact_text, category, tags = entry.fact, entry.category, entry.tags or []

    _index_memory_keywords(memory_id, fact_text, category, tags)
    return result


def forget(memory_id: str) -> bool:
    """Delete a memory and everything derived from it."""
    with session_scope() as session:
        entry = session.query(MemoryFact).filter(MemoryFact.id == memory_id).first()
        if entry is None:
            return False
        wiki_path = entry.wiki_path
        session.delete(entry)

    _purge_memory_artifacts(memory_id, wiki_path)
    return True


def get_by_id(memory_id: str) -> dict | None:
    with session_scope() as session:
        entry = session.query(MemoryFact).filter(MemoryFact.id == memory_id).first()
        return _fact_to_dict(entry) if entry else None


# ─── Recall ────────────────────────────────────────────────────────────


def _semantic_memory_ids(query: str, limit: int) -> list[str]:
    """Memory IDs from the vector store, best match first."""
    try:
        from ..rag.engine import get_engine

        hits = get_engine().search(query, limit=limit, where={"origin": "memory"}, raise_on_error=False)
    except Exception as exc:  # noqa: BLE001 - semantic leg is optional
        logger.debug("Semantic recall unavailable: %s", exc)
        return []

    ordered: list[str] = []
    for h in hits:
        mid = h.get("memory_id")
        if mid and mid not in ordered:
            ordered.append(mid)
    return ordered


def recall(
    query: str = "",
    category: str | None = None,
    limit: int = 10,
    min_importance: int = 0,
    tags: list | None = None,
    semantic: bool = True,
) -> list[dict]:
    """Recall memories by keyword + meaning, with metadata filters."""
    limit = max(1, limit)
    ranked_ids: list[str] = []

    # Ranking happens before the SQL filters, so when a filter will discard
    # candidates we have to widen the candidate pool first — otherwise a
    # tagged memory that ranks 30th is dropped before the tag is even read.
    post_filtered = bool(tags or category or min_importance)
    candidate_limit = min(limit * (40 if post_filtered else 5), 2000)

    if query:
        keyword_ids = [
            r["memory_id"] for r in memory_index.search(query, limit=candidate_limit) if r.get("memory_id")
        ]
        semantic_ids = _semantic_memory_ids(query, candidate_limit) if semantic else []

        if keyword_ids or semantic_ids:
            scores = rrf_fuse([keyword_ids, semantic_ids])
            ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)

    with session_scope() as session:
        q = session.query(MemoryFact)

        if query:
            if ranked_ids:
                q = q.filter(MemoryFact.id.in_(ranked_ids))
            else:
                # Nothing indexed yet (fresh install, or pre-0.2 database
                # before `amnis reindex`): degrade to a substring match.
                like = f"%{escape_like(query)}%"
                q = q.filter(
                    or_(
                        MemoryFact.fact.ilike(like, escape="\\"),
                        MemoryFact.context.ilike(like, escape="\\"),
                    )
                )

        if category:
            q = q.filter(MemoryFact.category == category)
        if min_importance > 0:
            q = q.filter(MemoryFact.importance >= min_importance)

        q = q.filter(
            or_(
                MemoryFact.expiry.is_(None),
                MemoryFact.expiry > datetime.now(UTC),
            )
        )

        # Over-fetch so the Python-side tag filter still has candidates to
        # work with. Previously the tag filter ran *after* LIMIT, so asking
        # for 10 tagged memories could return 0 while dozens existed.
        fetch = candidate_limit if post_filtered else limit
        q = q.order_by(desc(MemoryFact.importance), desc(MemoryFact.timestamp))
        rows = q.limit(fetch).all()

        if tags:
            wanted = {t.lower() for t in tags}
            rows = [r for r in rows if wanted & {t.lower() for t in (r.tags or [])}]

        if ranked_ids:
            order = {mid: i for i, mid in enumerate(ranked_ids)}
            rows.sort(key=lambda r: order.get(r.id, len(order)))

        rows = rows[:limit]

        now = datetime.now(UTC)
        for r in rows:
            r.access_count = (r.access_count or 0) + 1
            r.last_accessed = now

        return [_fact_to_dict(r) for r in rows]


def all_memories(limit: int = 50, offset: int = 0) -> list[dict]:
    with session_scope() as session:
        rows = (
            session.query(MemoryFact)
            .order_by(desc(MemoryFact.importance), desc(MemoryFact.timestamp))
            .offset(max(0, offset))
            .limit(max(1, limit))
            .all()
        )
        return [_fact_to_dict(r) for r in rows]


def count(category: str | None = None) -> int:
    with session_scope() as session:
        q = session.query(func.count(MemoryFact.id))
        if category:
            q = q.filter(MemoryFact.category == category)
        return int(q.scalar() or 0)


def clear_expired() -> int:
    """Remove expired memories, cleaning up their wiki/RAG artifacts."""
    with session_scope() as session:
        rows = (
            session.query(MemoryFact)
            .filter(MemoryFact.expiry.isnot(None))
            .filter(MemoryFact.expiry < datetime.now(UTC))
            .all()
        )
        doomed = [(r.id, r.wiki_path) for r in rows]
        for r in rows:
            session.delete(r)

    for memory_id, wiki_path in doomed:
        _purge_memory_artifacts(memory_id, wiki_path)
    return len(doomed)


def stats() -> dict:
    with session_scope() as session:
        total = session.query(func.count(MemoryFact.id)).scalar() or 0
        by_category = (
            session.query(MemoryFact.category, func.count(MemoryFact.id)).group_by(MemoryFact.category).all()
        )
        avg_importance = session.query(func.avg(MemoryFact.importance)).scalar() or 0
        avg_confidence = session.query(func.avg(MemoryFact.confidence)).scalar() or 0
        expired = (
            session.query(func.count(MemoryFact.id))
            .filter(MemoryFact.expiry.isnot(None))
            .filter(MemoryFact.expiry < datetime.now(UTC))
            .scalar()
            or 0
        )
        return {
            "total_memories": int(total),
            "by_category": {cat: cnt for cat, cnt in by_category},
            "avg_importance": round(float(avg_importance), 1),
            "avg_confidence": round(float(avg_confidence), 2),
            "expired": int(expired),
            "keyword_indexed": memory_index.count(),
        }


def reindex_keywords() -> dict:
    """Rebuild the memory FTS5 index from scratch.

    Run once after upgrading from 0.1, which had no memory keyword index.
    """
    memory_index.clear()
    with session_scope() as session:
        rows = session.query(MemoryFact).all()
        payload = [
            {
                "content": r.fact,
                "memory_id": r.id,
                "category": r.category,
                "tags": " ".join(r.tags or []),
            }
            for r in rows
        ]
    added = memory_index.add(payload)
    return {"indexed": added}


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
        "access_count": fact.access_count or 0,
        "expiry": fact.expiry.isoformat() if fact.expiry else None,
        "wiki_path": fact.wiki_path,
    }
