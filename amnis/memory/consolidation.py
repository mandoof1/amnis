"""Memory consolidation — extract structured facts from episodic logs.

Uses a simple LLM-free approach: keyword extraction, frequency analysis,
pattern matching. For deeper consolidation, can be wired to an LLM later.
"""
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import func, desc

from ..database import get_session, ConversationLog, MemoryFact
from ..config import config
import uuid


def consolidate_recent(limit: int = 50) -> dict:
    """Scan recent conversation logs and extract actionable facts.
    
    This is a simple extractor. In production you'd pipe through an LLM
    for proper fact extraction. Here we use heuristics and pattern matching.
    """
    session = get_session()
    try:
        logs = (
            session.query(ConversationLog)
            .filter(ConversationLog.role == "assistant")
            .order_by(desc(ConversationLog.timestamp))
            .limit(limit)
            .all()
        )

        extracted = 0
        for log in logs:
            content = log.content
            if not content:
                continue

            # Simple heuristics:
            # Look for declarative statements with personal info
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if not line or len(line) < 30:
                    continue

                # Check if it looks like a fact about LO (has "you" or "your")
                lower = line.lower()
                if any(p in lower for p in ["you ", "your ", "you're", "you've", "you'll"]):
                    # Don't store questions or commands
                    if line.endswith("?") or line.startswith("!"):
                        continue
                    
                    # Check if we already have something similar
                    existing = (
                        session.query(MemoryFact)
                        .filter(MemoryFact.fact.ilike(f"%{line[:50]}%"))
                        .first()
                    )
                    if not existing:
                        fact = MemoryFact(
                            id=str(uuid.uuid4()),
                            fact=line[:500],
                            category="event",
                            importance=3,
                            source="consolidation",
                            confidence=0.5,
                            tags=["auto-extracted", "conversation"],
                            context=f"Extracted from conversation log {log.id}",
                            timestamp=datetime.now(timezone.utc),
                            last_accessed=datetime.now(timezone.utc),
                        )
                        session.add(fact)
                        extracted += 1

        session.commit()
        return {
            "logs_scanned": len(logs),
            "memories_extracted": extracted,
        }
    finally:
        session.close()


def run_pipeline() -> dict:
    """Run the full consolidation pipeline."""
    result = consolidate_recent()
    return result
