"""Amnis configuration — loaded from environment/.env with sensible defaults.

Every path derives from ``data_dir`` unless explicitly overridden, so setting
``AMNIS_DATA_DIR`` alone relocates the whole installation.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AmnisConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AMNIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── Paths ────────────────────────────────────────────────────────
    # Only data_dir has a hard default. Everything else is derived from it
    # by _derive_paths() below unless the operator overrides it explicitly.
    data_dir: Path = Path.home() / "amnis" / "data"
    notes_dir: Path | None = None
    wiki_dir: Path | None = None
    memory_db: Path | None = None
    chroma_dir: Path | None = None
    wiki_facts_dir: Path | None = None

    # ─── Embedding ────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ─── RAG ──────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    default_search_limit: int = 5
    rrf_k: int = Field(default=60, ge=1, description="Reciprocal Rank Fusion constant")

    # ─── Memory ───────────────────────────────────────────────────────
    memory_default_limit: int = 10
    consolidation_batch_size: int = 100
    consolidation_min_line_length: int = 30
    importance_keywords: list[str] = Field(
        default_factory=list,
        description="Domain keywords that add +1 importance to an extracted fact. "
        'Set AMNIS_IMPORTANCE_KEYWORDS=["linux","rust"].',
    )

    # ─── Consolidation ────────────────────────────────────────────────
    dedup_similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    contradiction_distance_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    importance_boost_recent: int = 2
    importance_boost_frequent: int = 1

    # ─── Episodic Memory ──────────────────────────────────────────────
    episodic_retention_days: int = 30
    episodic_max_per_session: int = 50

    # ─── Pruning ──────────────────────────────────────────────────────
    prune_low_importance: int = 3
    prune_unaccessed_days: int = 60
    prune_batch_size: int = 50
    confidence_decay_rate: float = Field(default=0.98, gt=0.0, le=1.0)

    # ─── Wiki ─────────────────────────────────────────────────────────
    wiki_max_pages: int = 100
    wiki_max_tokens: int = 200_000

    # ─── Server ───────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8799
    api_token: str | None = Field(
        default=None,
        description="If set, every mutating web API call must send X-Amnis-Token.",
    )
    allow_index_outside_notes: bool = Field(
        default=False,
        description="Allow /api/index-file to read paths outside notes_dir/wiki_dir.",
    )

    @model_validator(mode="after")
    def _derive_paths(self) -> AmnisConfig:
        if self.notes_dir is None:
            self.notes_dir = self.data_dir / "notes"
        if self.wiki_dir is None:
            self.wiki_dir = self.data_dir / "wiki"
        if self.memory_db is None:
            self.memory_db = self.data_dir / "memory.db"
        if self.chroma_dir is None:
            self.chroma_dir = self.data_dir / "chroma"
        if self.wiki_facts_dir is None:
            self.wiki_facts_dir = self.wiki_dir / "facts"
        return self

    @model_validator(mode="after")
    def _check_chunking(self) -> AmnisConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}) or chunking cannot advance."
            )
        return self

    @property
    def memory_db_url(self) -> str:
        return f"sqlite:///{self.memory_db}"

    @property
    def keyword_db(self) -> Path:
        return self.data_dir / "keyword.db"


def unknown_env_vars() -> list[str]:
    """Return AMNIS_* environment variables that no config field consumes.

    Silently ignored env vars are a classic source of "I set it and nothing
    happened" bug reports, so the CLI and web server surface these on start.
    """
    known = {f"AMNIS_{name.upper()}" for name in AmnisConfig.model_fields}
    return sorted(k for k in os.environ if k.startswith("AMNIS_") and k not in known)


config = AmnisConfig()
