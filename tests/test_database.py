"""Engine lifecycle, timezone handling, and the 0.1 -> 0.2 migration."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import inspect


def test_datetimes_round_trip_as_aware_utc(amnis_env):
    from amnis.database import MemoryFact, session_scope

    with session_scope() as session:
        session.add(MemoryFact(id=str(uuid.uuid4()), fact="x", timestamp=datetime.now(UTC)))

    with session_scope() as session:
        row = session.query(MemoryFact).one()
        assert row.timestamp.tzinfo is not None
        # The 0.1 schema returned naive values here, so this subtraction was
        # a guaranteed TypeError in prune/lint/decay.
        assert (datetime.now(UTC) - row.timestamp) < timedelta(minutes=1)


def test_engine_is_reused(amnis_env):
    from amnis.database import get_engine

    assert get_engine() is get_engine()


def test_session_scope_rolls_back(amnis_env):
    import pytest

    from amnis.database import MemoryFact, session_scope

    with pytest.raises(RuntimeError):
        with session_scope() as session:
            session.add(MemoryFact(id="rollback", fact="should not persist"))
            raise RuntimeError("boom")

    with session_scope() as session:
        assert session.query(MemoryFact).filter(MemoryFact.id == "rollback").first() is None


def test_migration_adds_columns_and_indexes_to_a_01_database(amnis_env):
    """Build a 0.1-shaped table, then let get_engine() upgrade it."""
    from amnis import database

    database.reset_engine()
    db_path = amnis_env.memory_db
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE memory_facts (
            id VARCHAR(36) PRIMARY KEY, fact TEXT NOT NULL,
            category VARCHAR(50) NOT NULL DEFAULT 'general',
            importance INTEGER DEFAULT 5, source VARCHAR(50) DEFAULT 'manual',
            confidence FLOAT DEFAULT 1.0, tags JSON, context TEXT,
            timestamp DATETIME, last_accessed DATETIME,
            access_count INTEGER DEFAULT 0, expiry DATETIME
        );
        CREATE TABLE indexed_documents (
            id VARCHAR(36) PRIMARY KEY, path VARCHAR(500) NOT NULL UNIQUE,
            title VARCHAR(200), doc_type VARCHAR(50), chunk_count INTEGER,
            last_indexed DATETIME, file_hash VARCHAR(64), file_size INTEGER
        );
        INSERT INTO memory_facts (id, fact, timestamp, last_accessed)
        VALUES ('legacy-1', 'a naive legacy row', '2026-01-01 00:00:00', '2026-01-01 00:00:00');
        """
    )
    conn.commit()
    conn.close()

    engine = database.get_engine()
    inspector = inspect(engine)
    assert "wiki_path" in {c["name"] for c in inspector.get_columns("memory_facts")}
    assert "origin" in {c["name"] for c in inspector.get_columns("indexed_documents")}
    index_names = {i["name"] for i in inspector.get_indexes("memory_facts")}
    assert "ix_memory_facts_category" in index_names
    assert "ix_memory_facts_importance_timestamp" in index_names

    with database.session_scope() as session:
        row = session.query(database.MemoryFact).filter_by(id="legacy-1").one()
        assert row.timestamp.tzinfo is UTC
