"""Chunking and fusion — the parts that need no vector store."""

from __future__ import annotations

from amnis.rag.engine import _chunk_text, _split_words


def test_word_splitting_always_advances():
    words = [str(i) for i in range(10)]
    # chunk_size == overlap made range(0, n, 0) raise ValueError in 0.1.
    chunks = _split_words(words, chunk_size=4, overlap=4)
    assert chunks and all(chunks)
    assert len(chunks) == 10


def test_markdown_chunking_is_heading_aware(amnis_env):
    text = "# Title\n\nintro\n\n## Alpha\n\nalpha body\n\n## Beta\n\nbeta body\n"
    chunks = _chunk_text(text, "notes/doc.md")
    headings = [c["heading"] for c in chunks]
    assert "Alpha" in headings and "Beta" in headings
    assert all(c["total_chunks"] == len(chunks) for c in chunks)


def test_plain_text_falls_back_to_word_chunks(amnis_env, monkeypatch):
    monkeypatch.setattr(amnis_env, "chunk_size", 10)
    monkeypatch.setattr(amnis_env, "chunk_overlap", 2)
    chunks = _chunk_text(" ".join(["word"] * 45), "notes/plain.txt")
    assert len(chunks) > 1
    assert all(c["heading"] == "" for c in chunks)
