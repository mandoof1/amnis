"""Automated cleanup of stale, low-importance, and duplicate facts.

Conservative by design: manual entries are never auto-deleted, and everything
supports ``dry_run``.

Fixes over 0.1: the age arithmetic here used to mix a naive "now" with aware
column values (``TypeError`` on every run); merged duplicates lost tags when
either side was ``None``; and deletions left the mirrored wiki file and its
ChromaDB chunks behind, so "deleted" memories kept coming back in search.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ..config import config
from ..database import MemoryFact, session_scope
from .store import _purge_memory_artifacts


def prune_low_importance(
    threshold: int | None = None,
    unaccessed_days: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Delete low-importance memories that nothing has touched in a while."""
    threshold = config.prune_low_importance if threshold is None else threshold
    unaccessed_days = config.prune_unaccessed_days if unaccessed_days is None else unaccessed_days
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=unaccessed_days)

    doomed: list[tuple[str, str | None]] = []
    with session_scope() as session:
        candidates = (
            session.query(MemoryFact)
            .filter(
                MemoryFact.importance < threshold,
                MemoryFact.last_accessed < cutoff,
                MemoryFact.source != "manual",
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
                "aged_memories": [],
                "importance_threshold": threshold,
                "unaccessed_days": unaccessed_days,
                "message": "No low-importance stale memories found.",
            }

        ages = [
            {
                "id": c.id,
                "fact": c.fact[:80],
                "importance": c.importance,
                "source": c.source,
                "days_unaccessed": (now - c.last_accessed).days,
            }
            for c in candidates
        ]
        total = len(candidates)

        if not dry_run:
            doomed = [(c.id, c.wiki_path) for c in candidates]
            for c in candidates:
                session.delete(c)

    for memory_id, wiki_path in doomed:
        _purge_memory_artifacts(memory_id, wiki_path)

    return {
        "pruned": total if not dry_run else 0,
        "dry_run": dry_run,
        "candidates": total,
        "aged_memories": ages[:5],
        "importance_threshold": threshold,
        "unaccessed_days": unaccessed_days,
    }


def prune_expired() -> int:
    """Remove memories past their expiry date, artifacts included."""
    from .store import clear_expired

    return clear_expired()


def merge_duplicates(similarity_check: bool = False, dry_run: bool = False) -> dict:
    """Merge memories whose first 50 characters match, keeping the best one."""
    doomed: list[tuple[str, str | None]] = []
    merged_details: list[dict] = []
    merged = 0

    with session_scope() as session:
        all_memories = session.query(MemoryFact).order_by(MemoryFact.timestamp).all()

        groups: dict[str, list[MemoryFact]] = {}
        for m in all_memories:
            groups.setdefault(m.fact[:50].lower().strip(), []).append(m)

        for group in groups.values():
            if len(group) < 2:
                continue

            group.sort(key=lambda x: (x.importance or 0, len(x.fact)), reverse=True)
            keeper, duplicates = group[0], group[1:]

            for dup in duplicates:
                merged += 1
                merged_details.append(
                    {
                        "kept": keeper.fact[:80],
                        "removed": dup.fact[:80],
                        "kept_id": keeper.id,
                        "removed_id": dup.id,
                    }
                )

                if dry_run:
                    continue

                keeper.access_count = (keeper.access_count or 0) + (dup.access_count or 0)
                keeper.importance = max(keeper.importance or 0, dup.importance or 0)
                keeper.confidence = max(keeper.confidence or 0.0, dup.confidence or 0.0)
                if len(dup.fact) > len(keeper.fact):
                    keeper.fact = dup.fact
                # set() over both sides: `keeper.tags + dup.tags` raised
                # TypeError whenever either column was NULL.
                keeper.tags = sorted(set(keeper.tags or []) | set(dup.tags or []))
                if dup.last_accessed and keeper.last_accessed:
                    keeper.last_accessed = max(keeper.last_accessed, dup.last_accessed)

                doomed.append((dup.id, dup.wiki_path))
                session.delete(dup)

    for memory_id, wiki_path in doomed:
        _purge_memory_artifacts(memory_id, wiki_path)

    if not dry_run and merged:
        from .store import reindex_keywords

        reindex_keywords()

    return {
        "merged": merged if not dry_run else 0,
        "dry_run": dry_run,
        "candidates": merged,
        "details": merged_details[:10],
    }


def decay_importance() -> dict:
    """Decay confidence of auto-extracted facts that go unreinforced.

    ``confidence *= rate ** days_since_last_access`` for non-manual facts.
    """
    rate = config.confidence_decay_rate
    now = datetime.now(UTC)
    decayed = 0
    lowest = 1.0

    with session_scope() as session:
        candidates = (
            session.query(MemoryFact).filter(MemoryFact.source != "manual", MemoryFact.confidence > 0.1).all()
        )
        for mem in candidates:
            if not mem.last_accessed:
                continue
            days = (now - mem.last_accessed).days
            if days <= 1:
                continue
            new_conf = round((mem.confidence or 1.0) * (rate**days), 4)
            if new_conf >= (mem.confidence or 1.0):
                continue
            mem.confidence = max(new_conf, 0.1)
            decayed += 1
            lowest = min(lowest, mem.confidence)

    return {"decayed_facts": decayed, "lowest_confidence": round(lowest, 4)}


def run_pipeline(dry_run: bool = False) -> dict:
    """Decay confidence, drop expired, merge duplicates, prune stale."""
    decayed = decay_importance()
    expired = 0 if dry_run else prune_expired()
    duplicates = merge_duplicates(dry_run=dry_run)
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
