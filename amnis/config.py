"""Amnis configuration — loaded from env/file with sensible defaults."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class AmnisConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AMNIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths — Amnis manages its own data, no external vault dependency
    data_dir: Path = Path.home() / "amnis" / "data"
    notes_dir: Path = Path.home() / "amnis" / "data" / "notes"
    wiki_dir: Path = Path.home() / "amnis" / "data" / "wiki"
    memory_db: Path = Path.home() / "amnis" / "data" / "memory.db"
    chroma_dir: Path = Path.home() / "amnis" / "data" / "chroma"

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    embedding_dimension: int = 384

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 64
    default_search_limit: int = 5
    semantic_weight: float = 0.7      # hybrid search: weight for semantic vs keyword (0.0-1.0)

    # Memory
    memory_default_limit: int = 10
    consolidation_batch_size: int = 100

    # Consolidation
    dedup_similarity_threshold: float = 0.85   # merge memories above this cosine similarity
    contradiction_distance_threshold: float = 0.92  # flag memories this close with opposing sentiment
    importance_boost_recent: int = 2            # extra importance for memories from last session
    importance_boost_frequent: int = 1          # extra importance per 3 access counts

    # Episodic Memory
    episodic_retention_days: int = 30           # auto-prune episodes older than this
    episodic_max_per_session: int = 50          # max episodes stored per session ID

    # Pruning
    prune_low_importance: int = 3               # delete memories below this importance when pruning
    prune_unaccessed_days: int = 60             # delete low-importance memories unaccessed this long
    prune_batch_size: int = 50                  # max memories pruned per run

    # Wiki
    wiki_max_pages: int = 100
    wiki_max_tokens: int = 200_000
    wiki_facts_dir: Path = Path.home() / "amnis" / "data" / "wiki" / "facts"

    # Server
    host: str = "127.0.0.1"
    port: int = 8799

    @property
    def memory_db_url(self) -> str:
        return f"sqlite:///{self.memory_db}"


config = AmnisConfig()
