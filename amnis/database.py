"""SQLAlchemy models for Amnis memory store."""
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Integer, Float, DateTime, JSON, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import config

Base = declarative_base()


class MemoryFact(Base):
    """A single persistent fact/memory stored across sessions."""
    __tablename__ = "memory_facts"

    id = Column(String(36), primary_key=True)
    fact = Column(Text, nullable=False)
    category = Column(String(50), nullable=False, default="general")
    importance = Column(Integer, default=5)  # 1-10
    source = Column(String(50), default="manual")
    confidence = Column(Float, default=1.0)  # 0.0-1.0
    tags = Column(JSON, default=list)
    context = Column(Text, nullable=True)  # surrounding context
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_accessed = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    access_count = Column(Integer, default=0)
    expiry = Column(DateTime, nullable=True)


class WikiPage(Base):
    """A compiled wiki page — Karpathy-style structured knowledge."""
    __tablename__ = "wiki_pages"

    id = Column(String(36), primary_key=True)
    title = Column(String(200), nullable=False, index=True)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    sources = Column(JSON, default=list)
    related = Column(JSON, default=list)
    tags = Column(JSON, default=list)
    last_compiled = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    version = Column(Integer, default=1)


class IndexedDocument(Base):
    """Track which documents have been indexed in ChromaDB."""
    __tablename__ = "indexed_documents"

    id = Column(String(36), primary_key=True)
    path = Column(String(500), nullable=False, unique=True)
    title = Column(String(200), nullable=True)
    doc_type = Column(String(50), default="markdown")
    chunk_count = Column(Integer, default=0)
    last_indexed = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    file_hash = Column(String(64), nullable=True)  # for change detection
    file_size = Column(Integer, default=0)


class ConversationLog(Base):
    """Episodic memory — records of past interactions."""
    __tablename__ = "conversation_logs"

    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    summary = Column(Text, nullable=True)
    topics = Column(JSON, default=list)
    outcome = Column(String(20), nullable=True)  # success / failure / neutral
    results = Column(JSON, nullable=True)  # structured outcome data


def init_db():
    """Initialize the database and return a session."""
    engine = create_engine(config.memory_db_url, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def get_session():
    """Get a fresh DB session, initializing tables if needed."""
    engine = create_engine(config.memory_db_url, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()
