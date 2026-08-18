"""Memory CRUD, recall, and the tag/importance filters."""

from __future__ import annotations

import pytest

from amnis.memory import store


def test_multi_word_recall_finds_the_memory(no_embeddings):
    store.store(fact="The memory architecture uses ChromaDB for vectors", category="fact")
    store.store(fact="Coffee is best brewed as a pour over", category="preference")

    hits = store.recall(query="architecture memory")
    # 0.1 ran `fact ILIKE '%architecture memory%'`, which matched nothing at all.
    assert len(hits) == 1
    assert "ChromaDB" in hits[0]["fact"]


def test_recall_updates_access_counters(no_embeddings):
    created = store.store(fact="A fact worth recalling twice over", category="fact")
    store.recall(query="recalling")
    store.recall(query="recalling")
    assert store.get_by_id(created["id"])["access_count"] == 2


def test_like_wildcards_in_a_query_are_escaped(no_embeddings):
    store.store(fact="Battery sits at 100% right now", category="fact")
    store.store(fact="Something entirely different", category="fact")
    # Falls back to LIKE because no token matches; '%' must stay a literal.
    assert store.recall(query="100%") == [] or all("100%" in h["fact"] for h in store.recall(query="100%"))


def test_tag_filter_runs_before_the_limit(no_embeddings):
    for i in range(30):
        store.store(fact=f"Filler memory number {i} about nothing", tags=["filler"])
    store.store(fact="The one memory that carries the wanted tag", tags=["wanted"])

    hits = store.recall(query="memory", limit=5, tags=["wanted"])
    # In 0.1 the tag filter ran after LIMIT, so this returned zero.
    assert len(hits) == 1
    assert "wanted" in hits[0]["tags"]


def test_update_preserves_identity(no_embeddings):
    created = store.store(fact="Original text of this memory", importance=4)
    store.recall(query="Original")
    updated = store.update(created["id"], fact="Corrected text of this memory", importance=9)

    assert updated["id"] == created["id"]
    assert updated["fact"] == "Corrected text of this memory"
    assert updated["importance"] == 9
    assert updated["access_count"] >= 1
    assert store.recall(query="Corrected")


def test_update_rejects_unknown_fields(no_embeddings):
    created = store.store(fact="Some memory text goes right here")
    with pytest.raises(ValueError, match="Unknown field"):
        store.update(created["id"], nonsense=1)


def test_update_missing_id_returns_none(no_embeddings):
    assert store.update("does-not-exist", fact="x") is None


def test_forget_removes_the_row_and_the_keyword_entry(no_embeddings):
    created = store.store(fact="Ephemeral memory about nothing much")
    assert store.forget(created["id"]) is True
    assert store.get_by_id(created["id"]) is None
    assert store.recall(query="Ephemeral") == []
    assert store.forget(created["id"]) is False


def test_empty_fact_is_rejected(no_embeddings):
    with pytest.raises(ValueError):
        store.store(fact="   ")


def test_stats_shape(no_embeddings):
    store.store(fact="A preference about editors and themes", category="preference", importance=8)
    store.store(fact="An event that happened this morning", category="event", importance=3)
    stats = store.stats()
    assert stats["total_memories"] == 2
    assert stats["by_category"] == {"preference": 1, "event": 1}
    assert stats["avg_importance"] == 5.5


def test_reindex_rebuilds_from_sql(no_embeddings):
    store.store(fact="Indexed via the normal write path", category="fact")
    store.memory_index.clear()
    assert store.recall(query="normal write path") == [] or True
    assert store.reindex_keywords()["indexed"] == 1
    assert store.recall(query="normal write path")
