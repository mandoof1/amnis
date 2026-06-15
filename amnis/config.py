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

    # Paths
    data_dir: Path = Path.home() / "amnis" / "data"
    vault_path: Path = Path.home() / "Documents" / "Obsidian Vault"
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

    # Memory
    memory_default_limit: int = 10
    consolidation_batch_size: int = 100

    # Wiki
    wiki_max_pages: int = 100
    wiki_max_tokens: int = 200_000

    # Server
    host: str = "127.0.0.1"
    port: int = 8799

    @property
    def memory_db_url(self) -> str:
        return f"sqlite:///{self.memory_db}"


config = AmnisConfig()
