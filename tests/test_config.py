"""Configuration derivation and validation."""

from __future__ import annotations

import pytest

from amnis.config import AmnisConfig, unknown_env_vars


def test_every_path_derives_from_data_dir(tmp_path):
    cfg = AmnisConfig(data_dir=tmp_path)
    assert cfg.notes_dir == tmp_path / "notes"
    assert cfg.wiki_dir == tmp_path / "wiki"
    assert cfg.memory_db == tmp_path / "memory.db"
    assert cfg.chroma_dir == tmp_path / "chroma"
    assert cfg.wiki_facts_dir == tmp_path / "wiki" / "facts"


def test_explicit_override_wins(tmp_path):
    cfg = AmnisConfig(data_dir=tmp_path, notes_dir=tmp_path / "elsewhere")
    assert cfg.notes_dir == tmp_path / "elsewhere"
    assert cfg.wiki_dir == tmp_path / "wiki"


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        AmnisConfig(chunk_size=64, chunk_overlap=64)


def test_unknown_env_vars_are_reported(monkeypatch):
    monkeypatch.setenv("AMNIS_VAULT_PATH", "/nope")
    monkeypatch.setenv("AMNIS_DATA_DIR", "/tmp")
    unknown = unknown_env_vars()
    assert "AMNIS_VAULT_PATH" in unknown
    assert "AMNIS_DATA_DIR" not in unknown
