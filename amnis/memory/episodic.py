"""Episodic memory — chronological event logging (CoALA episodic layer).

Episodic memory is autobiographical and expiring; semantic memory (the fact
store) is durable. Consolidation promotes the former into the latter.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy import String as SAString
from sqlalchemy import cast, desc, func

from ..config import config
from ..database import ConversationLog, session_scope
from .store import escape_like

# Public: consolidation reuses this list, and it was previously private, so
# the two modules maintained separate near-identical copies.
STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "and",
    "but",
    "or",
    "nor",
    "not",
    "so",
    "yet",
    "both",
    "either",
    "neither",
    "each",
    "every",
    "all",
    "any",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "only",
    "own",
    "same",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
    "if",
    "then",
    "else",
    "about",
    "just",
    "also",
    "very",
    "too",
    "really",
    "actually",
    "basically",
    "here",
    "there",
    "up",
    "down",
    "out",
    "off",
    "over",
    "back",
    "well",
    "get",
    "got",
    "go",
    "going",
    "went",
    "come",
    "came",
    "take",
    "took",
    "make",
    "made",
    "know",
    "knows",
    "think",
    "thinks",
    "say",
    "says",
    "said",
    "see",
    "let",
    "like",
    "want",
    "use",
    "used",
    "using",
    "need",
    "needs",
}

_STOP_WORDS = STOP_WORDS  # backwards-compatible alias


def extract_topics(text: str, max_topics: int = 5) -> list[str]:
    """Extract key topics by frequency, minus stop words."""
    if not text:
        return []
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    meaningful = [w for w in words if w not in STOP_WORDS]
    if not meaningful:
        return []
    return [word for word, _ in Counter(meaningful).most_common(max_topics)]


def generate_summary(text: str, max_words: int = 30) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words]) + "..."


def log_episode(
    session_id: str,
    role: str,
    content: str,
    summary: str | None = None,
    topics: list[str] | None = None,
    outcome: str | None = None,
    results: dict | None = None,
) -> dict:
    """Log a conversation turn, auto-deriving summary and topics if omitted."""
    if not summary:
        summary = generate_summary(content)
    if not topics:
        topics = extract_topics(content)

    with session_scope() as session:
        entry = ConversationLog(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.now(UTC),
            summary=summary,
            topics=topics,
            outcome=outcome,
            results=results,
        )
        session.add(entry)
        # Flush so the new row is counted; otherwise the cap was enforced one
        # episode late and the buffer grew past episodic_max_per_session.
        session.flush()

        count = (
            session.query(func.count(ConversationLog.id))
            .filter(ConversationLog.session_id == session_id)
            .scalar()
            or 0
        )
        excess = count - config.episodic_max_per_session
        if excess > 0:
            oldest = (
                session.query(ConversationLog)
                .filter(ConversationLog.session_id == session_id)
                .order_by(ConversationLog.timestamp)
                .limit(excess)
                .all()
            )
            for old in oldest:
                session.delete(old)

        return _episode_to_dict(entry)


def recall_episodes(
    session_id: str | None = None,
    topic: str | None = None,
    role: str | None = None,
    limit: int = 20,
    since: datetime | None = None,
) -> list[dict]:
    with session_scope() as session:
        q = session.query(ConversationLog)
        if session_id:
            q = q.filter(ConversationLog.session_id == session_id)
        if role:
            q = q.filter(ConversationLog.role == role)
        if since:
            q = q.filter(ConversationLog.timestamp >= since)
        if topic:
            like = f"%{escape_like(topic)}%"
            q = q.filter(cast(ConversationLog.topics, SAString).ilike(like, escape="\\"))

        rows = q.order_by(desc(ConversationLog.timestamp)).limit(max(1, limit)).all()
        return [_episode_to_dict(r) for r in rows]


def get_session_summary(session_id: str) -> dict:
    with session_scope() as session:
        episodes = (
            session.query(ConversationLog)
            .filter(ConversationLog.session_id == session_id)
            .order_by(ConversationLog.timestamp)
            .all()
        )
        if not episodes:
            return {
                "session_id": session_id,
                "episodes": 0,
                "duration_minutes": 0,
                "topics": [],
            }

        all_topics: Counter = Counter()
        for ep in episodes:
            if ep.topics:
                all_topics.update(ep.topics)

        duration = (episodes[-1].timestamp - episodes[0].timestamp).total_seconds() / 60

        return {
            "session_id": session_id,
            "episodes": len(episodes),
            "duration_minutes": round(duration, 1),
            "start_time": episodes[0].timestamp.isoformat(),
            "end_time": episodes[-1].timestamp.isoformat(),
            "topics": [t for t, _ in all_topics.most_common(10)],
        }


def list_sessions(limit: int = 25) -> list[dict]:
    """Most recent sessions with episode counts — powers the web UI timeline."""
    with session_scope() as session:
        rows = (
            session.query(
                ConversationLog.session_id,
                func.count(ConversationLog.id),
                func.max(ConversationLog.timestamp),
            )
            .group_by(ConversationLog.session_id)
            .order_by(desc(func.max(ConversationLog.timestamp)))
            .limit(max(1, limit))
            .all()
        )
        return [
            {
                "session_id": sid,
                "episodes": int(cnt),
                "last_seen": last.isoformat() if last else None,
            }
            for sid, cnt, last in rows
        ]


def prune_old_episodes(days: int | None = None) -> int:
    days = config.episodic_retention_days if days is None else days
    cutoff = datetime.now(UTC) - timedelta(days=days)
    with session_scope() as session:
        return (
            session.query(ConversationLog)
            .filter(ConversationLog.timestamp < cutoff)
            .delete(synchronize_session=False)
        )


def stats() -> dict:
    with session_scope() as session:
        total = session.query(func.count(ConversationLog.id)).scalar() or 0
        if not total:
            return {
                "total_episodes": 0,
                "unique_sessions": 0,
                "oldest_episode": None,
                "newest_episode": None,
            }
        oldest = session.query(func.min(ConversationLog.timestamp)).scalar()
        newest = session.query(func.max(ConversationLog.timestamp)).scalar()
        unique_sessions = session.query(func.count(func.distinct(ConversationLog.session_id))).scalar() or 0
        return {
            "total_episodes": int(total),
            "unique_sessions": int(unique_sessions),
            "oldest_episode": oldest.isoformat() if oldest else None,
            "newest_episode": newest.isoformat() if newest else None,
        }


def _episode_to_dict(ep: ConversationLog) -> dict:
    return {
        "id": ep.id,
        "session_id": ep.session_id,
        "role": ep.role,
        "summary": ep.summary or (ep.content[:100] + "..."),
        "topics": ep.topics or [],
        "outcome": ep.outcome,
        "results": ep.results,
        "timestamp": ep.timestamp.isoformat() if ep.timestamp else None,
    }
