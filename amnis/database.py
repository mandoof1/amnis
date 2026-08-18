"""SQLAlchemy models and engine management for the Amnis memory store.

Two things here matter more than the models themselves:

1. ``UTCDateTime`` — SQLite's dialect silently drops ``tzinfo``, so a value
   written as timezone-aware reads back naive. Every later ``now(utc) - value``
   then raises ``TypeError``. This TypeDecorator normalises on the way in and
   re-attaches UTC on the way out, so callers only ever see aware datetimes.

2. One engine per process — the original code built a fresh engine (and ran
   ``create_all``) on every single query, which is both slow and a connection
   leak. ``get_engine()`` memoises it and runs the lightweight migration once.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.types import TypeDecorator

from .config import config

Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """DateTime column that always returns timezone-aware UTC values.

    Stored naive-UTC for SQLite compatibility (and so legacy rows written by
    Amnis 0.1 keep working), but re-hydrated as aware so arithmetic against
    ``datetime.now(timezone.utc)`` never raises.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class MemoryFact(Base):
    """A single persistent fact stored across sessions."""

    __tablename__ = "memory_facts"

    id = Column(String(36), primary_key=True)
    fact = Column(Text, nullable=False)
    category = Column(String(50), nullable=False, default="general", index=True)
    importance = Column(Integer, default=5, index=True)  # 1-10
    source = Column(String(50), default="manual")
    confidence = Column(Float, default=1.0)  # 0.0-1.0
    tags = Column(JSON, default=list)
    context = Column(Text, nullable=True)
    timestamp = Column(UTCDateTime, default=utcnow, index=True)
    last_accessed = Column(UTCDateTime, default=utcnow, index=True)
    access_count = Column(Integer, default=0)
    expiry = Column(UTCDateTime, nullable=True)
    # Path of the auto-generated wiki page, so deletes can clean it up
    # instead of orphaning a markdown file that stays in the RAG index.
    wiki_path = Column(String(500), nullable=True)

    __table_args__ = (Index("ix_memory_facts_importance_timestamp", "importance", "timestamp"),)


class WikiPage(Base):
    """A compiled wiki page."""

    __tablename__ = "wiki_pages"

    id = Column(String(36), primary_key=True)
    # Deliberately indexed but not UNIQUE: pre-0.2 databases can contain
    # duplicate titles (the old compiler matched with ILIKE '%topic%'), and a
    # unique index would fail to create on upgrade. Uniqueness is enforced by
    # exact-title lookup in the compiler instead.
    title = Column(String(200), nullable=False, index=True)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    sources = Column(JSON, default=list)
    related = Column(JSON, default=list)
    tags = Column(JSON, default=list)
    last_compiled = Column(UTCDateTime, default=utcnow)
    version = Column(Integer, default=1)


class IndexedDocument(Base):
    """Tracks which documents have been indexed into ChromaDB."""

    __tablename__ = "indexed_documents"

    id = Column(String(36), primary_key=True)
    path = Column(String(500), nullable=False, unique=True)
    title = Column(String(200), nullable=True)
    doc_type = Column(String(50), default="markdown")
    chunk_count = Column(Integer, default=0)
    last_indexed = Column(UTCDateTime, default=utcnow)
    file_hash = Column(String(64), nullable=True)
    file_size = Column(Integer, default=0)
    # "note" | "wiki" | "memory" | "compiled" — lets retrieval exclude
    # Amnis's own compiled output so the wiki cannot cite itself.
    origin = Column(String(20), default="note", index=True)


class ConversationLog(Base):
    """Episodic memory — records of past interactions."""

    __tablename__ = "conversation_logs"

    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(UTCDateTime, default=utcnow, index=True)
    summary = Column(Text, nullable=True)
    topics = Column(JSON, default=list)
    outcome = Column(String(20), nullable=True)
    results = Column(JSON, nullable=True)


# ─── Engine management ─────────────────────────────────────────────────

_engine: Engine | None = None
_engine_url: str | None = None
_session_factory: sessionmaker | None = None

# Columns added after 0.1.0. SQLite can ALTER TABLE ADD COLUMN cheaply, so
# upgrades are automatic and non-destructive.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "memory_facts": {"wiki_path": "VARCHAR(500)"},
    "indexed_documents": {"origin": "VARCHAR(20) DEFAULT 'note'"},
}


def _migrate(engine: Engine) -> None:
    """Add columns and indexes that ``create_all`` skips on existing tables."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table, columns in _ADDED_COLUMNS.items():
            if table not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table)}
            for name, ddl in columns.items():
                if name not in present:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))

    # create_all(checkfirst=True) skips tables that already exist — including
    # any indexes declared on them. Create those explicitly.
    for table in Base.metadata.tables.values():
        if table.name not in existing_tables:
            continue
        for index in table.indexes:
            try:
                index.create(bind=engine, checkfirst=True)
            except Exception:  # noqa: BLE001 - index creation is best-effort
                pass


def get_engine() -> Engine:
    """Return the process-wide engine, creating schema on first use."""
    global _engine, _engine_url, _session_factory

    url = config.memory_db_url
    if _engine is not None and _engine_url == url:
        return _engine

    if _engine is not None:
        _engine.dispose()

    config.memory_db.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(url, echo=False, future=True)
    _engine_url = url
    Base.metadata.create_all(_engine)
    _migrate(_engine)
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def reset_engine() -> None:
    """Drop the cached engine — used by tests and after a config change."""
    global _engine, _engine_url, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _engine_url = None
    _session_factory = None


def get_session() -> Session:
    """Get a fresh session. Prefer ``session_scope()`` for new code."""
    get_engine()
    assert _session_factory is not None
    return _session_factory()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commits on success, rolls back on exception.

    The original code used ``try/finally: session.close()`` without a rollback,
    so a mid-transaction failure left changes pending and silently discarded
    them at close. Errors now propagate instead of being lost.
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> Session:
    """Backwards-compatible alias for callers that expect a live session."""
    return get_session()
