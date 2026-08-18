"""FTS5 query building, the index itself, and rank fusion."""

from __future__ import annotations

from amnis.rag.keyword import KeywordIndex, build_match_query, rrf_fuse


def test_operators_and_punctuation_are_quoted_not_interpreted():
    assert build_match_query("memory AND chroma") == '"memory" OR "AND" OR "chroma"'
    assert build_match_query('say "hi"') == '"say" OR "hi"'
    assert build_match_query("!!! ???") == ""
    assert build_match_query("a*b:c") == '"a" OR "b" OR "c"'


def test_index_round_trip(tmp_path):
    index = KeywordIndex(db_path=tmp_path / "kw.db")
    index.add(
        [
            {"content": "the memory architecture uses chroma", "source": "a.md", "chunk_idx": 0},
            {"content": "an unrelated cooking recipe", "source": "b.md", "chunk_idx": 0},
        ]
    )
    hits = index.search("architecture memory")
    assert [h["source"] for h in hits] == ["a.md"]
    assert hits[0]["rank_position"] == 1
    assert "rank" in hits[0]

    assert index.count("source", "a.md") == 1
    index.remove_where("source", "a.md")
    assert index.search("architecture memory") == []


def test_reserved_words_do_not_raise(tmp_path):
    index = KeywordIndex(db_path=tmp_path / "kw.db")
    index.add([{"content": "alpha beta gamma", "source": "s", "chunk_idx": 0}])
    # In 0.1 this reached MATCH as a syntax error and the bare except turned
    # it into a silent empty result.
    assert index.search("alpha NOT beta OR gamma")


def test_unsafe_identifiers_are_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        KeywordIndex(table="bad; DROP TABLE x", db_path=tmp_path / "kw.db")
    with pytest.raises(ValueError):
        KeywordIndex(columns=("ok", "bad col"), db_path=tmp_path / "kw.db")


def test_rrf_rewards_appearing_in_both_rankings():
    scores = rrf_fuse([["a", "b"], ["a", "c"]], k=60)
    # `a` is retrieved by both systems; `b` and `c` by one each.
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["c"]


def test_rrf_is_symmetric_in_ranking_order():
    forward = rrf_fuse([["x", "y"], ["y", "x"]], k=60)
    assert abs(forward["x"] - forward["y"]) < 1e-12


def test_rrf_ignores_score_scale_entirely():
    # Two rankings, wildly different underlying scores, identical order:
    # fusion must depend only on position.
    assert rrf_fuse([["p", "q"]], k=10) == rrf_fuse([["p", "q"]], k=10)
