"""Pruning arithmetic, duplicate merging, and confidence decay."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from amnis.database import MemoryFact, session_scope
from amnis.memory import pruning


def _make(fact, **kwargs):
    defaults = dict(
        id=str(uuid.uuid4()),
        category="event",
        importance=5,
        source="consolidation",
        confidence=1.0,
        tags=[],
        timestamp=datetime.now(UTC),
        last_accessed=datetime.now(UTC),
        access_count=0,
    )
    defaults.update(kwargs)
    return MemoryFact(fact=fact, **defaults)


def test_pipeline_runs_without_type_errors(no_embeddings):
    old = datetime.now(UTC) - timedelta(days=400)
    with session_scope() as session:
        session.add(_make("A stale low importance memory", importance=1, last_accessed=old))
        session.add(_make("An important manual memory", importance=9, source="manual"))

    # The whole point: 0.1 raised TypeError comparing naive and aware datetimes.
    result = pruning.run_pipeline(dry_run=True)
    assert result["dry_run"] is True
    assert result["low_importance_candidates"] == 1
    assert result["low_importance_pruned"] == 0


def test_low_importance_prune_spares_manual_entries(no_embeddings):
    old = datetime.now(UTC) - timedelta(days=400)
    with session_scope() as session:
        session.add(_make("Auto extracted and long forgotten", importance=1, last_accessed=old))
        session.add(_make("Manual and long forgotten", importance=1, source="manual", last_accessed=old))

    result = pruning.prune_low_importance(dry_run=False)
    assert result["pruned"] == 1
    with session_scope() as session:
        remaining = [m.source for m in session.query(MemoryFact).all()]
    assert remaining == ["manual"]


def test_merge_duplicates_unions_tags_even_when_null(no_embeddings):
    shared = "This exact prefix is shared by both of these memories"
    with session_scope() as session:
        session.add(_make(shared, tags=["alpha"], importance=6))
        session.add(_make(shared + " with a longer tail", tags=None, importance=4))

    result = pruning.merge_duplicates(dry_run=False)
    assert result["merged"] == 1
    with session_scope() as session:
        keeper = session.query(MemoryFact).one()
    # `keeper.tags + dup.tags` raised TypeError against a NULL column in 0.1.
    assert keeper.tags == ["alpha"]
    assert keeper.fact.endswith("longer tail")


def test_decay_only_touches_unreinforced_auto_facts(no_embeddings):
    old = datetime.now(UTC) - timedelta(days=30)
    with session_scope() as session:
        session.add(_make("Auto fact left alone for a month", last_accessed=old))
        session.add(_make("Manual fact left alone for a month", source="manual", last_accessed=old))
        session.add(_make("Auto fact touched today"))

    result = pruning.decay_importance()
    assert result["decayed_facts"] == 1
    with session_scope() as session:
        by_fact = {m.fact: m.confidence for m in session.query(MemoryFact).all()}
    assert by_fact["Auto fact left alone for a month"] < 1.0
    assert by_fact["Manual fact left alone for a month"] == 1.0
    assert by_fact["Auto fact touched today"] == 1.0
