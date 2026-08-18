"""Wiki page lookup, versioning, stats, and linting."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from amnis.database import WikiPage, session_scope


def _page(title, sources, version=1):
    return WikiPage(
        id=str(uuid.uuid4()),
        title=title,
        content=f"# {title}\n",
        summary=title,
        sources=sources,
        related=[],
        tags=[],
        version=version,
        last_compiled=datetime.now(UTC),
    )


def test_stats_counts_distinct_sources_not_rows(amnis_env):
    from amnis.wiki.compiler import get_compiler

    with session_scope() as session:
        session.add(_page("Rust", ["a.md", "b.md"]))
        session.add(_page("Python", ["b.md", "c.md"]))

    stats = get_compiler().stats()
    assert stats["total_pages"] == 2
    # 0.1 returned the row count here, so total_sources always equalled
    # total_pages no matter what the pages actually cited.
    assert stats["total_sources"] == 3


def test_lint_reports_sourceless_and_duplicate_titles(amnis_env):
    from amnis.wiki.compiler import get_compiler

    with session_scope() as session:
        session.add(_page("Duplicated", ["a.md"]))
        session.add(_page("Duplicated", ["a.md"]))
        session.add(_page("Sourceless", []))

    result = get_compiler().lint()
    kinds = {i["issue"] for i in result["issues"]}
    assert result["pages_checked"] == 3
    assert "no_sources" in kinds
    assert "duplicate_title" in kinds


def test_query_matches_individual_terms(amnis_env, monkeypatch):
    from amnis.wiki.compiler import get_compiler

    compiler = get_compiler()
    monkeypatch.setattr(compiler, "_retrieve", lambda *a, **k: [])
    with session_scope() as session:
        session.add(_page("Chroma", ["a.md"]))

    # A whole-sentence LIKE (the 0.1 behaviour) would find nothing here.
    result = compiler.query("How does chroma store its vectors?")
    assert [p["title"] for p in result["wiki_pages"]] == ["Chroma"]
    assert "chroma" in result["terms"]


def test_compile_updates_in_place_and_bumps_version(amnis_env, monkeypatch):
    from amnis.memory import store
    from amnis.wiki.compiler import get_compiler

    compiler = get_compiler()
    monkeypatch.setattr(compiler, "_retrieve", lambda *a, **k: [])
    monkeypatch.setattr(store, "recall", lambda **k: [])

    first = compiler._compile_topic("rust")
    second = compiler._compile_topic("rust")
    assert first["id"] == second["id"]
    # 0.1 mutated a detached object, so version stayed at 1 forever.
    assert (first["version"], second["version"]) == (1, 2)

    with session_scope() as session:
        assert session.query(WikiPage).count() == 1


def test_exact_title_match_does_not_clobber_a_similar_page(amnis_env, monkeypatch):
    from amnis.memory import store
    from amnis.wiki.compiler import get_compiler

    compiler = get_compiler()
    monkeypatch.setattr(compiler, "_retrieve", lambda *a, **k: [])
    monkeypatch.setattr(store, "recall", lambda **k: [])

    with session_scope() as session:
        session.add(_page("Rust Async Runtimes", ["a.md"]))

    compiler._compile_topic("rust")
    with session_scope() as session:
        titles = sorted(p.title for p in session.query(WikiPage).all())
    # `title ILIKE '%rust%'` would have overwritten the longer page.
    assert titles == ["Rust", "Rust Async Runtimes"]
