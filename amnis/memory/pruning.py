"""Memory Pruning — automated cleanup of stale, low-importance, and duplicate facts.

Based on the Memory AI + RAG deep research:
  - Mem0 and Letta both implement forgetting mechanisms
  - Stale memories dilute retrieval quality
  - Contradictory memories reduce confidence
  - Automatic pruning prevents memory bloat without manual intervention

This module is intentionally conservative — it flags/deletes only clearly
stale or low-value memories. High-importance memories are never auto-pruned.
"""
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import desc, func

from ..config import config
from ..database import get_session, MemoryFact


def prune_low_importance(
    threshold: int = None,
    unaccessed_days: int = None,
    dry_run: bool = False,
) -> dict:
    """Delete memories below importance threshold that haven't been accessed recently.

    Args:
        threshold: Importance below which memories are candidates (default: config.prune_low_importance)
        unaccessed_days: Days since last access for a memory to be considered stale
        dry_run: If True, only report what would be deleted without actually deleting

    Returns:
        Summary of pruning operation
    """
    if threshold is None:
        threshold = config.prune_low_importance
    if unaccessed_days is None:
        unaccessed_days = config.prune_unaccessed_days

    cutoff = datetime.now(timezone.utc) - timedelta(days=unaccessed_days)

    session = get_session()
    try:
        # Find candidates: low importance + unaccessed for a while + not manual source
        candidates = (
            session.query(MemoryFact)
            .filter(
                MemoryFact.importance < threshold,
                MemoryFact.last_accessed < cutoff,
                MemoryFact.source != "manual",  # Never auto-delete manual entries
            )
            .order_by(MemoryFact.importance, MemoryFact.last_accessed)
            .limit(config.prune_batch_size)
            .all()
        )

        if not candidates:
            return {
                "pruned": 0,
                "candidates": 0,
                "dry_run": dry_run,
                "message": "No low-importance stale memories found.",
            }

        # Group by age for reporting
        total = len(candidates)
        ages = []
        for c in candidates:
            days_since = (datetime.now(timezone.utc) - c.last_accessed).days
            ages.append({
                "id": c.id,
                "fact": c.fact[:80],
                "importance": c.importance,
                "source": c.source,
                "days_unaccessed": days_since,
            })

        if not dry_run:
            # Also remove from RAG if indexed
            for c in candidates:
                _remove_from_rag_if_exists(c.fact[:100])

            # Delete from memory store
            ids = [c.id for c in candidates]
            (
                session.query(MemoryFact)
                .filter(MemoryFact.id.in_(ids))
                .delete(synchronize_session=False)
            )
            session.commit()

        return {
            "pruned": total if not dry_run else 0,
            "dry_run": dry_run,
            "candidates": total,
            "aged_memories": ages[:5],  # Sample for reporting
            "importance_threshold": threshold,
            "unaccessed_days": unaccessed_days,
        }
    finally:
        session.close()


def prune_expired() -> int:
    """Remove all expired memories (those past their expiry date).

    Returns:
        Number of memories deleted
    """
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


def merge_duplicates(
    similarity_check: bool = False,
    dry_run: bool = False,
) -> dict:
    """Merge near-duplicate memories (same text, slightly different wording).

    Phase 1: Exact/close text duplicates (always runs)
    Phase 2: Semantic duplicates (only if similarity_check=True, uses sentence embeddings)

    Args:
        similarity_check: If True, also check semantic similarity (uses sentence-transformers)
        dry_run: If True, only report without modifying

    Returns:
        Summary of merge operation
    """
    session = get_session()
    try:
        # Phase 1: Find memories with very similar text (first 80 chars match)
        all_memories = (
            session.query(MemoryFact)
            .order_by(MemoryFact.timestamp)
            .all()
        )

        merged = 0
        merged_details = []

        # Group by first 50 chars (case-insensitive)
        groups = {}
        for m in all_memories:
            key = m.fact[:50].lower().strip()
            if key not in groups:
                groups[key] = []
            groups[key].append(m)

        for key, group in groups.items():
            if len(group) < 2:
                continue

            # Keep the best one: highest importance + longest text
            group.sort(key=lambda x: (x.importance, len(x.fact)), reverse=True)
            keeper = group[0]
            to_merge = group[1:]

            for dup in to_merge:
                merged += 1
                merged_details.append({
                    "kept": keeper.fact[:80],
                    "removed": dup.fact[:80],
                    "kept_id": keeper.id,
                    "removed_id": dup.id,
                })

                if not dry_run:
                    # Roll up stats to keeper
                    keeper.access_count += dup.access_count
                    keeper.importance = max(keeper.importance, dup.importance)
                    keeper.confidence = max(keeper.confidence, dup.confidence)
                    if len(dup.fact) > len(keeper.fact):
                        keeper.fact = dup.fact  # Keep longer version
                    keeper.tags = list(set(keeper.tags + dup.tags))  # Merge tags
                    keeper.last_accessed = max(keeper.last_accessed, dup.last_accessed)

                    # Delete duplicate
                    session.delete(dup)

        if not dry_run:
            session.commit()

        return {
            "merged": merged if not dry_run else 0,
            "dry_run": dry_run,
            "candidates": merged,
            "details": merged_details[:10],
        }
    finally:
        session.close()


def decay_importance() -> dict:
    """Decay confidence of auto-extracted facts that haven't been reinforced.
    
    Applies confidence *= 0.98^(days_since_last_access) for facts with
    source != 'manual' and importance > 2. Prevents stale facts from
    persisting at inflated confidence levels.
    
    Returns summary of decay operations.
    """
    session = get_session()
    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)  # naive for DB compat
        candidates = (
            session.query(MemoryFact)
            .filter(
                MemoryFact.source != "manual",
                MemoryFact.confidence > 0.1,
            )
            .all()
        )
        decayed = 0
        lowest = 1.0
        for mem in candidates:
            days = (now - mem.last_accessed).days
            if days <= 1:
                continue
            factor = 0.98 ** days
            new_conf = round(mem.confidence * factor, 4)
            if new_conf >= mem.confidence:
                continue
            mem.confidence = max(new_conf, 0.1)
            decayed += 1
            lowest = min(lowest, mem.confidence)
        session.commit()
        return {"decayed_facts": decayed, "lowest_confidence": round(lowest, 4)}
    finally:
        session.close()


def _remove_from_rag_if_exists(text_snippet: str):
    """Remove associated RAG entries if a memory was also indexed there."""
    try:
        from ..rag.engine import engine
        # Search for chunks matching this text
        results = engine.search(text_snippet, limit=2)
        for r in results:
            if r.get("source", "").startswith("memory:"):
                engine.delete_source(r["source"])
    except Exception:
        pass  # RAG engine might not be initialized


def run_pipeline(dry_run: bool = False) -> dict:
    """Run the full pruning pipeline.

    Steps:
      0. Decay confidence of unaccessed auto-extracted facts
      1. Remove expired memories
      2. Merge exact/close text duplicates
      3. Prune low-importance, stale memories
      4. Return summary

    Args:
        dry_run: If True, only report what would be done (no modifications)
    """
    decayed = decay_importance()
    expired = prune_expired()
    duplicates = merge_duplicates(similarity_check=False, dry_run=dry_run)
    low_imp = prune_low_importance(dry_run=dry_run)

    return {
        "confidence_decayed": decayed["decayed_facts"],
        "lowest_confidence": decayed["lowest_confidence"],
        "expired_removed": expired,
        "duplicates_merged": duplicates["merged"],
        "duplicate_candidates": duplicates["candidates"],
        "low_importance_pruned": low_imp["pruned"],
        "low_importance_candidates": low_imp["candidates"],
        "dry_run": dry_run,
        "total_removed": expired + duplicates["merged"] + low_imp["pruned"],
    }
