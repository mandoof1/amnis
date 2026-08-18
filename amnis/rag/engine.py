"""RAG engine — ChromaDB + sentence-transformers over indexed documents.

Notable behaviour changes from 0.1:

* The engine is created lazily. Importing ``amnis`` used to construct a
  ChromaDB client and open the notes directory as a side effect of
  ``import``, which made ``python -m amnis --help`` take seconds.
* ``index_file`` reads its tracking row and writes it back inside a single
  session. The old code opened one session to read, closed it, then opened a
  second to write through the *detached* object from the first — so on the
  update path nothing was ever persisted and files were silently re-embedded
  on every run.
* Errors raise ``RagError`` instead of being returned as ``[{"error": ...}]``,
  a shape every caller treated as a successful result list.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func

from ..config import config
from ..database import IndexedDocument, session_scope
from .keyword import KeywordIndex, rrf_fuse


class RagError(RuntimeError):
    """Raised when indexing or querying fails."""


# ─── File helpers ──────────────────────────────────────────────────────


def _is_markdown(path: Path) -> bool:
    return path.suffix.lower() in (".md", ".markdown", ".txt")


def _split_words(words: list[str], chunk_size: int, overlap: int) -> list[list[str]]:
    """Sliding window over words with a guaranteed forward step.

    ``chunk_size - overlap`` is <= 0 if a user misconfigures the two, which
    made ``range(0, n, step)`` raise ValueError (step 0) or loop backwards.
    Config validation catches this too; this is the belt to that's braces.
    """
    step = max(1, chunk_size - overlap)
    return [words[i : i + chunk_size] for i in range(0, len(words), step) if words[i : i + chunk_size]]


# ─── Chunking ──────────────────────────────────────────────────────────


def _chunk_markdown(text: str, source_path: str) -> list[dict]:
    """Split markdown into heading-aware chunks."""
    chunks: list[dict] = []
    lines = text.split("\n")

    sections: list[tuple[str | None, int, int, list[str]]] = []
    current_heading: str | None = None
    current_level = 0
    current_start = 0
    current_lines: list[str] = []

    for i, line in enumerate(lines):
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            if current_lines:
                sections.append((current_heading, current_level, current_start, current_lines))
            current_level = len(heading_match.group(1))
            current_heading = heading_match.group(2).strip()
            current_start = i
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_level, current_start, current_lines))
    if not sections:
        sections = [("", 0, 0, lines)]

    chunk_size = config.chunk_size
    overlap = config.chunk_overlap

    for heading, level, _start_idx, section_lines in sections:
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            continue

        words = section_text.split()
        if len(words) <= chunk_size:
            chunks.append(
                {
                    "text": section_text,
                    "source": source_path,
                    "heading": heading or "",
                    "heading_level": level,
                    "chunk_index": len(chunks),
                    "total_chunks": 0,
                }
            )
            continue

        for n, chunk_words in enumerate(_split_words(words, chunk_size, overlap)):
            chunk_text = " ".join(chunk_words)
            if heading and n > 0:
                chunk_text = f"[{heading}] {chunk_text}"
            chunks.append(
                {
                    "text": chunk_text,
                    "source": source_path,
                    "heading": heading or "",
                    "heading_level": level,
                    "chunk_index": len(chunks),
                    "total_chunks": 0,
                }
            )

    for c in chunks:
        c["total_chunks"] = len(chunks)
    return chunks


def _chunk_plain_text(text: str, source_path: str) -> list[dict]:
    words = text.split()
    chunks = [
        {
            "text": " ".join(chunk_words),
            "source": source_path,
            "heading": "",
            "heading_level": 0,
            "chunk_index": i,
            "total_chunks": 0,
        }
        for i, chunk_words in enumerate(_split_words(words, config.chunk_size, config.chunk_overlap))
    ]
    for c in chunks:
        c["total_chunks"] = len(chunks)
    return chunks


def _chunk_text(text: str, source_path: str) -> list[dict]:
    path = Path(source_path)
    if _is_markdown(path) and re.search(r"^#{1,6}\s", text, re.MULTILINE):
        return _chunk_markdown(text, source_path)
    return _chunk_plain_text(text, source_path)


# ─── Engine ────────────────────────────────────────────────────────────


class RagEngine:
    """ChromaDB-backed RAG engine with local embeddings + FTS5 hybrid search."""

    def __init__(self) -> None:
        config.data_dir.mkdir(parents=True, exist_ok=True)
        config.chroma_dir.mkdir(parents=True, exist_ok=True)

        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.client = chromadb.PersistentClient(
            path=str(config.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="amnis",
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = None
        self.keyword = KeywordIndex()

    # ─── embeddings ────────────────────────────────────────────────────

    @property
    def embedder(self):
        """Lazy-load the embedding model (~80MB of weights)."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(config.embedding_model)
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts, show_progress_bar=False).tolist()

    # ─── indexing ──────────────────────────────────────────────────────

    def index_file(
        self,
        file_path: str,
        origin: str = "note",
        extra_metadata: dict | None = None,
    ) -> dict:
        """Index one file into ChromaDB + the FTS5 keyword index.

        ``origin`` tags every chunk so retrieval can exclude a class of
        content — notably Amnis's own compiled wiki pages, which otherwise
        become the top sources for the next compile run.
        """
        path = Path(file_path).expanduser()
        if not path.exists():
            raise RagError(f"File not found: {file_path}")
        if not path.is_file():
            raise RagError(f"Not a regular file: {file_path}")

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise RagError(f"Could not read {file_path}: {exc}") from exc

        file_hash = hashlib.sha256(text.encode()).hexdigest()
        file_size = path.stat().st_size
        source = str(path.resolve())

        chunks = _chunk_text(text, source)
        if not chunks:
            return {"file": source, "indexed": 0, "reason": "empty"}

        # Single session for the whole read/decide/write cycle — the object
        # stays attached, so the update branch actually persists.
        with session_scope() as session:
            existing = session.query(IndexedDocument).filter(IndexedDocument.path == source).first()

            if existing is not None and existing.file_hash == file_hash:
                # Unchanged, but the keyword index may predate this file's
                # entries (e.g. index cleared, or upgrading from 0.1).
                if self.keyword.count("source", source) == 0:
                    self.keyword.add(
                        [
                            {
                                "content": c["text"],
                                "source": c["source"],
                                "heading": c.get("heading", ""),
                                "chunk_idx": c["chunk_index"],
                                "file_hash": file_hash,
                            }
                            for c in chunks
                        ]
                    )
                return {
                    "file": source,
                    "indexed": 0,
                    "reason": "unchanged",
                    "chunks": existing.chunk_count,
                    "origin": existing.origin,
                }

            chunk_texts = [c["text"] for c in chunks]
            try:
                embeddings = self._embed(chunk_texts)
            except Exception as exc:  # noqa: BLE001 - surfaced as RagError
                raise RagError(f"Embedding failed for {source}: {exc}") from exc

            ids = [f"{path.stem}-{c['chunk_index']}-{file_hash[:8]}" for c in chunks]
            indexed_at = datetime.now(UTC).isoformat()
            metadatas = []
            for c in chunks:
                meta = {
                    "source": source,
                    "chunk": c["chunk_index"],
                    "total": c["total_chunks"],
                    "heading": c.get("heading", ""),
                    "heading_level": str(c.get("heading_level", 0)),
                    "file_type": path.suffix,
                    "file_hash": file_hash,
                    "origin": origin,
                    "indexed_at": indexed_at,
                }
                if extra_metadata:
                    meta.update({k: str(v) for k, v in extra_metadata.items()})
                metadatas.append(meta)

            self._drop_chroma_source(source)
            self.collection.add(
                ids=ids,
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            self.keyword.remove_where("source", source)
            self.keyword.add(
                [
                    {
                        "content": c["text"],
                        "source": c["source"],
                        "heading": c.get("heading", ""),
                        "chunk_idx": c["chunk_index"],
                        "file_hash": file_hash,
                    }
                    for c in chunks
                ]
            )

            if existing is not None:
                existing.chunk_count = len(chunks)
                existing.last_indexed = datetime.now(UTC)
                existing.file_hash = file_hash
                existing.file_size = file_size
                existing.origin = origin
            else:
                session.add(
                    IndexedDocument(
                        id=str(uuid.uuid4()),
                        path=source,
                        title=path.stem,
                        doc_type=path.suffix.lstrip("."),
                        chunk_count=len(chunks),
                        file_hash=file_hash,
                        file_size=file_size,
                        origin=origin,
                    )
                )

        return {"file": source, "indexed": len(chunks), "chunks": len(chunks), "origin": origin}

    def _index_tree(self, root: Path, origin: str, skip: set[str] | None = None) -> dict:
        files = sorted(set(root.rglob("*.md")) | set(root.rglob("*.txt")))
        total_chunks = 0
        total_files = 0
        errors: list[str] = []

        for f in files:
            rel = f.relative_to(root)
            if any(p.startswith(".") for p in rel.parts):
                continue
            if skip and rel.as_posix() in skip:
                continue
            try:
                result = self.index_file(str(f), origin=origin)
            except RagError as exc:
                errors.append(str(exc))
                continue
            if result.get("indexed", 0) > 0:
                total_chunks += result["indexed"]
                total_files += 1

        return {
            "files_indexed": total_files,
            "chunks_indexed": total_chunks,
            "total_found": len(files),
            "errors": errors,
        }

    def index_notes(self) -> dict:
        """Index every markdown/text file under the notes directory."""
        notes_dir = Path(config.notes_dir)
        if not notes_dir.exists():
            notes_dir.mkdir(parents=True, exist_ok=True)
            return {
                "warning": f"Notes directory created at {notes_dir}",
                "files_indexed": 0,
                "chunks_indexed": 0,
                "total_found": 0,
                "errors": [],
            }
        return self._index_tree(notes_dir, origin="note")

    def index_wiki(self) -> dict:
        """Index the wiki directory. Compiled pages are tagged ``compiled``."""
        wiki_dir = Path(config.wiki_dir)
        if not wiki_dir.exists():
            wiki_dir.mkdir(parents=True, exist_ok=True)
            return {
                "warning": f"Wiki directory created at {wiki_dir}",
                "files_indexed": 0,
                "chunks_indexed": 0,
                "total_found": 0,
                "errors": [],
            }
        return self._index_tree(wiki_dir, origin="compiled", skip={"index.md"})

    # ─── search ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 5,
        where: dict | None = None,
        raise_on_error: bool = True,
    ) -> list[dict]:
        """Semantic search over indexed documents."""
        if not query.strip():
            return []

        try:
            query_embedding = self._embed([query])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, limit),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as RagError
            if raise_on_error:
                raise RagError(f"Search failed: {exc}") from exc
            return []

        hits: list[dict] = []
        documents = (results or {}).get("documents") or [[]]
        for i, doc in enumerate(documents[0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            dist = results["distances"][0][i] if results.get("distances") else 0.0
            hit = {
                "content": doc[:500] + "..." if len(doc) > 500 else doc,
                "source": meta.get("source", "unknown"),
                "heading": meta.get("heading", ""),
                "chunk": meta.get("chunk", 0),
                "score": round(1.0 - dist, 4),
                "file_type": meta.get("file_type", ""),
                "origin": meta.get("origin", "note"),
                "search_type": "semantic",
            }
            if meta.get("memory_id"):
                hit["memory_id"] = meta["memory_id"]
            hits.append(hit)
        return hits

    def keyword_search(self, query: str, limit: int = 5) -> list[dict]:
        """Keyword-only search, shaped like ``search()`` results."""
        rows = self.keyword.search(query, limit=limit)
        return [
            {
                "content": r["content"][:500] + "..." if len(r["content"]) > 500 else r["content"],
                "source": r.get("source", ""),
                "heading": r.get("heading", ""),
                "chunk": int(r.get("chunk_idx") or 0),
                "rank": r["rank"],
                "search_type": "keyword",
            }
            for r in rows
        ]

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        semantic_weight: float | None = None,  # noqa: ARG002 - kept for compat
        where: dict | None = None,
    ) -> list[dict]:
        """Semantic + keyword search merged with Reciprocal Rank Fusion.

        ``semantic_weight`` is accepted and ignored: RRF fuses ranks, so there
        is no shared scale for a weight to act on. It stays in the signature
        because existing MCP clients still pass it.
        """
        semantic = self.search(query, limit=limit * 2, where=where, raise_on_error=False)
        keyword = self.keyword_search(query, limit=limit * 2)

        if not semantic and not keyword:
            return []

        def key(r: dict) -> str:
            return f"{r.get('source', '')}:{r.get('chunk', 0)}"

        by_key: dict[str, dict] = {}
        for r in keyword + semantic:  # semantic last so its richer dict wins
            by_key.setdefault(key(r), {}).update(r)

        scores = rrf_fuse([[key(r) for r in semantic], [key(r) for r in keyword]])
        sem_rank = {key(r): i + 1 for i, r in enumerate(semantic)}
        kw_rank = {key(r): i + 1 for i, r in enumerate(keyword)}

        merged = []
        for k, data in by_key.items():
            data["hybrid_score"] = round(scores.get(k, 0.0), 6)
            data["rrf_sem_rank"] = sem_rank.get(k)
            data["rrf_kw_rank"] = kw_rank.get(k)
            data["search_type"] = (
                "hybrid" if k in sem_rank and k in kw_rank else ("semantic" if k in sem_rank else "keyword")
            )
            merged.append(data)

        merged.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return merged[:limit]

    # ─── maintenance ───────────────────────────────────────────────────

    def _drop_chroma_source(self, source_path: str) -> int:
        try:
            existing = self.collection.get(where={"source": source_path}, include=[])
        except Exception:  # noqa: BLE001 - collection may be empty/new
            return 0
        ids = (existing or {}).get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def delete_source(self, source_path: str) -> int:
        """Remove a source from ChromaDB, FTS5, and the tracking table."""
        count = self._drop_chroma_source(source_path)
        self.keyword.remove_where("source", source_path)
        with session_scope() as session:
            session.query(IndexedDocument).filter(IndexedDocument.path == source_path).delete(
                synchronize_session=False
            )
        return count

    def sources(self, origin: str | None = None, limit: int = 500) -> list[dict]:
        """List indexed documents from the tracking table."""
        with session_scope() as session:
            q = session.query(IndexedDocument)
            if origin:
                q = q.filter(IndexedDocument.origin == origin)
            rows = q.order_by(IndexedDocument.path).limit(limit).all()
            return [
                {
                    "path": r.path,
                    "title": r.title,
                    "chunks": r.chunk_count,
                    "origin": r.origin,
                    "size": r.file_size,
                    "last_indexed": r.last_indexed.isoformat() if r.last_indexed else None,
                }
                for r in rows
            ]

    def stats(self) -> dict:
        """Engine statistics.

        Source counts come from the tracking table. The old implementation
        called ``collection.get(include=["metadatas"])`` — pulling every chunk's
        metadata into memory on every dashboard refresh just to count distinct
        paths.
        """
        try:
            total_chunks = self.collection.count()
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return {"error": str(exc), "total_chunks": 0, "unique_sources": 0}

        with session_scope() as session:
            unique_sources = session.query(func.count(IndexedDocument.id)).scalar() or 0
            by_origin = dict(
                session.query(IndexedDocument.origin, func.count(IndexedDocument.id))
                .group_by(IndexedDocument.origin)
                .all()
            )

        return {
            "total_chunks": total_chunks,
            "unique_sources": int(unique_sources),
            "by_origin": {k or "note": v for k, v in by_origin.items()},
            "embedding_model": config.embedding_model,
            "keyword_index": self.keyword.stats(),
        }


# ─── Lazy singleton ────────────────────────────────────────────────────

_engine: RagEngine | None = None


def get_engine() -> RagEngine:
    """Return the shared engine, constructing it on first use."""
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine


def reset_engine() -> None:
    """Drop the cached engine — used by tests and after a config change."""
    global _engine
    _engine = None


def __getattr__(name: str):
    # Keeps `from amnis.rag.engine import engine` working without paying the
    # ChromaDB construction cost at import time.
    if name == "engine":
        return get_engine()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
