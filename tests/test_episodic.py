"""Episodic logging, retention cap, and filters."""

from __future__ import annotations

from amnis.memory import episodic


def test_topics_and_summary_are_derived(amnis_env):
    result = episodic.log_episode(
        session_id="s1",
        role="user",
        content="We discussed chroma indexing and chroma retrieval for the wiki",
    )
    assert "chroma" in result["topics"]
    assert result["summary"]


def test_per_session_cap_is_enforced(amnis_env, monkeypatch):
    monkeypatch.setattr(amnis_env, "episodic_max_per_session", 5)
    for i in range(9):
        episodic.log_episode(session_id="capped", role="user", content=f"turn {i}")
    # The flush() before counting is what makes this exact; 0.1 counted the
    # pre-insert total and so allowed the buffer to drift one over.
    assert len(episodic.recall_episodes(session_id="capped", limit=100)) == 5


def test_topic_filter_and_session_listing(amnis_env):
    episodic.log_episode(session_id="a", role="user", content="talking about rust compilers")
    episodic.log_episode(session_id="b", role="assistant", content="talking about garden soil")

    assert len(episodic.recall_episodes(topic="rust")) == 1
    assert len(episodic.recall_episodes(role="assistant")) == 1
    sessions = {s["session_id"] for s in episodic.list_sessions()}
    assert sessions == {"a", "b"}


def test_stats_and_prune(amnis_env):
    episodic.log_episode(session_id="s", role="user", content="hello there friend")
    assert episodic.stats()["total_episodes"] == 1
    assert episodic.prune_old_episodes(days=3650) == 0
    assert episodic.prune_old_episodes(days=-1) == 1
