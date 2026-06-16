"""Episodic Memory — chronological event logging for the CoALA episodic layer.

Based on the Memory AI + RAG deep research:
  - Episodic memory = autobiographical recall, past interactions, timestamped events
  - Should be SEPARATE from semantic memory (general facts, preferences)
  - Consolidation runs as a background process: episodic → extract facts → semantic store
  - Episodes auto-expire (retention window) — they're for recent context, not permanent

The existing ConversationLog table already stores raw interactions — this module
adds topic extraction, structured querying, and automatic pruning on top of it.
"""
import uuid
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from collections import Counter

from sqlalchemy import desc, func, cast, String as SAString, Text as SAText

from ..config import config
from ..database import get_session, ConversationLog


# ─── Topic Extraction ──────────────────────────────────────────────────

# Stop words to filter out from topic extraction
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "other", "some", "such",
    "no", "only", "own", "same", "that", "this", "these", "those", "it",
    "its", "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "what", "which", "who", "whom",
    "when", "where", "why", "how", "if", "then", "else", "about", "just",
    "also", "very", "too", "really", "actually", "basically", "here",
    "there", "up", "down", "out", "off", "over", "back", "well", "get",
    "got", "go", "going", "went", "come", "came", "take", "took", "make",
    "made", "know", "knows", "think", "thinks", "say", "says", "said",
    "see", "let", "like", "want", "use", "used", "using", "need", "needs",
}


def extract_topics(text: str, max_topics: int = 5) -> list[str]:
    """Extract key topics from text using frequency analysis.

    Simple but effective: filter meaningful words, count frequency,
    return the top N as topic keywords.
    """
    if not text:
        return []

    # Normalize and tokenize
    lower = text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', lower)

    # Filter stop words and common filler
    meaningful = [w for w in words if w not in _STOP_WORDS and len(w) > 2]

    if not meaningful:
        return []

    # Count frequency
    counter = Counter(meaningful)
    # Return top N most common
    return [word for word, _ in counter.most_common(max_topics)]


def generate_summary(text: str, max_words: int = 30) -> str:
    """Generate a brief summary from text (first N meaningful words)."""
    if not text:
        return ""

    # Clean and truncate
    cleaned = re.sub(r'\s+', ' ', text).strip()
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words]) + "..."


# ─── Logging ───────────────────────────────────────────────────────────


def log_episode(
    session_id: str,
    role: str,
    content: str,
    summary: Optional[str] = None,
    topics: Optional[list[str]] = None,
    outcome: Optional[str] = None,
    results: Optional[dict] = None,
) -> dict:
    """Log a conversation turn as an episode.

    Args:
        session_id: Hermes session ID
        role: 'user' or 'assistant'
        content: Full text of the turn
        summary: Optional short summary (auto-generated if not provided)
        topics: Optional topic tags (auto-extracted if not provided)
        outcome: Optional 'success' / 'failure' / 'neutral'
        results: Optional structured outcome data

    Returns:
        The created episode record
    """
    if not summary:
        summary = generate_summary(content)
    if not topics:
        topics = extract_topics(content)

    session = get_session()
    try:
        entry = ConversationLog(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc),
            summary=summary,
            topics=topics,
            outcome=outcome,
            results=results,
        )
        session.add(entry)

        # Enforce max episodes per session (prune oldest)
        count = (
            session.query(ConversationLog)
            .filter(ConversationLog.session_id == session_id)
            .count()
        )
        if count > config.episodic_max_per_session:
            # Find and delete oldest excess
            excess = count - config.episodic_max_per_session
            oldest = (
                session.query(ConversationLog)
                .filter(ConversationLog.session_id == session_id)
                .order_by(ConversationLog.timestamp)
                .limit(excess)
                .all()
            )
            for old in oldest:
                session.delete(old)

        session.commit()
        return _episode_to_dict(entry)
    finally:
        session.close()


# ─── Querying ──────────────────────────────────────────────────────────


def recall_episodes(
    session_id: Optional[str] = None,
    topic: Optional[str] = None,
    role: Optional[str] = None,
    limit: int = 20,
    since: Optional[datetime] = None,
) -> list[dict]:
    """Recall episodic memories with filters.

    Args:
        session_id: Filter by session
        topic: Filter by topic keyword (matches extracted topics)
        role: Filter by role ('user' or 'assistant')
        limit: Max results
        since: Only episodes after this timestamp

    Returns:
        List of matching episodes
    """
    session = get_session()
    try:
        q = session.query(ConversationLog)

        if session_id:
            q = q.filter(ConversationLog.session_id == session_id)
        if role:
            q = q.filter(ConversationLog.role == role)
        if since:
            q = q.filter(ConversationLog.timestamp >= since)
        if topic:
            # Match against topics JSON field using text representation
            q = q.filter(cast(ConversationLog.topics, SAString).ilike(f"%{topic}%"))

        q = q.order_by(desc(ConversationLog.timestamp)).limit(limit)
        results = q.all()
        return [_episode_to_dict(r) for r in results]
    finally:
        session.close()


def get_session_summary(session_id: str) -> dict:
    """Get a summary of all episodes in a session."""
    session = get_session()
    try:
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

        # Collect all topics
        all_topics = Counter()
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
    finally:
        session.close()


# ─── Pruning ───────────────────────────────────────────────────────────


def prune_old_episodes(days: Optional[int] = None) -> int:
    """Delete episodes older than retention period.

    Args:
        days: Retention period in days (defaults to config.episodic_retention_days)

    Returns:
        Number of episodes deleted
    """
    if days is None:
        days = config.episodic_retention_days

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    session = get_session()
    try:
        count = (
            session.query(ConversationLog)
            .filter(ConversationLog.timestamp < cutoff)
            .delete()
        )
        session.commit()
        return count
    finally:
        session.close()


# ─── Stats ─────────────────────────────────────────────────────────────


def stats() -> dict:
    """Get episodic memory statistics."""
    session = get_session()
    try:
        total = session.query(ConversationLog).count()
        if total == 0:
            return {
                "total_episodes": 0,
                "unique_sessions": 0,
                "oldest_episode": None,
                "newest_episode": None,
            }

        oldest = (
            session.query(func.min(ConversationLog.timestamp)).scalar()
        )
        newest = (
            session.query(func.max(ConversationLog.timestamp)).scalar()
        )
        unique_sessions = (
            session.query(ConversationLog.session_id).distinct().count()
        )

        return {
            "total_episodes": total,
            "unique_sessions": unique_sessions,
            "oldest_episode": oldest.isoformat() if oldest else None,
            "newest_episode": newest.isoformat() if newest else None,
        }
    finally:
        session.close()


# ─── Internal ──────────────────────────────────────────────────────────


def _episode_to_dict(ep: ConversationLog) -> dict:
    return {
        "id": ep.id,
        "session_id": ep.session_id,
        "role": ep.role,
        "summary": ep.summary or ep.content[:100] + "...",
        "topics": ep.topics or [],
        "outcome": ep.outcome,
        "results": ep.results,
        "timestamp": ep.timestamp.isoformat() if ep.timestamp else None,
    }
