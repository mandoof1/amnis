"""RAG Engine — ChromaDB + sentence-transformers semantic search over documents.

Handles:
  - Indexing documents (markdown, text, code)
  - Semantic + keyword hybrid search (FTS5 keyword index)
  - Heading-aware chunking for markdown files
  - Notes directory indexing
"""
import hashlib
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import config
from ..database import get_session, IndexedDocument


# ─── File Helpers ──────────────────────────────────────────────────────


def _is_markdown(path: Path) -> bool:
    return path.suffix.lower() in (".md", ".markdown", ".txt")


# ─── Heading-Aware Chunking ───────────────────────────────────────────


def _chunk_markdown(text: str, source_path: str) -> list[dict]:
    """Split markdown text into heading-aware chunks.

    Preserves document structure by splitting on ## headings.
    Each section becomes a chunk with its heading hierarchy in metadata.
    Sections longer than chunk_size words are sub-chunked at word boundaries.
    """
    chunks = []
    lines = text.split("\n")

    # Find headings and their content
    sections = []  # [(heading_line, heading_level, start_idx, content_lines)]
    current_heading = None
    current_level = 0
    current_start = 0
    current_lines = []

    for i, line in enumerate(lines):
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            # Save previous section
            if current_lines:
                sections.append((current_heading, current_level, current_start, current_lines))
            # Start new section
            level = len(heading_match.group(1))
            current_heading = heading_match.group(2).strip()
            current_level = level
            current_start = i
            current_lines = [line]
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        sections.append((current_heading, current_level, current_start, current_lines))

    # If no headings found, treat entire document as one section
    if not sections:
        sections = [("", 0, 0, lines)]

    chunk_size = config.chunk_size
    overlap = config.chunk_overlap

    for heading, level, start_idx, section_lines in sections:
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            continue

        words = section_text.split()
        total_words = len(words)

        if total_words <= chunk_size:
            # Small enough to be one chunk
            chunks.append({
                "text": section_text,
                "source": source_path,
                "heading": heading or "",
                "heading_level": level,
                "chunk_index": len(chunks),
                "total_chunks": 0,  # set after
            })
        else:
            # Sub-chunk at word boundaries
            for i in range(0, total_words, chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                if not chunk_words:
                    break
                chunk_text = " ".join(chunk_words)
                # Prepend heading for context
                if heading and i > 0:
                    chunk_text = f"[{heading}] {chunk_text}"
                chunks.append({
                    "text": chunk_text,
                    "source": source_path,
                    "heading": heading or "",
                    "heading_level": level,
                    "chunk_index": len(chunks),
                    "total_chunks": 0,
                })

    total = len(chunks)
    for c in chunks:
        c["total_chunks"] = total

    return chunks


def _chunk_plain_text(text: str, source_path: str) -> list[dict]:
    """Split plain text into word-level chunks (fallback)."""
    chunk_size = config.chunk_size
    overlap = config.chunk_overlap
    words = text.split()
    chunks = []

    for i in range(0, max(1, len(words)), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "text": chunk_text,
            "source": source_path,
            "heading": "",
            "heading_level": 0,
            "chunk_index": len(chunks),
            "total_chunks": 0,
        })

    total = len(chunks)
    for c in chunks:
        c["total_chunks"] = total

    return chunks


def _chunk_text(text: str, source_path: str) -> list[dict]:
    """Split text into chunks, using heading-aware splitting for markdown."""
    path = Path(source_path)
    if _is_markdown(path) and re.search(r'^#{1,6}\s', text, re.MULTILINE):
        return _chunk_markdown(text, source_path)
    return _chunk_plain_text(text, source_path)


# ─── FTS5 Keyword Index (lightweight, zero deps) ─────────────────────
#
# Uses SQLite FTS5 for fast keyword search. The FTS index is stored in
# a separate lightweight SQLite file (keyword.db) to avoid mixing with
# the main memory DB that SQLAlchemy manages.


class KeywordIndex:
    """Lightweight FTS5 keyword search index.

    Zero external dependencies — uses Python's built-in sqlite3 with FTS5.
    FTS5 is included in the standard SQLite compilation (and Python's sqlite3
    module uses the system SQLite).
    """

    def __init__(self):
        self.db_path = config.data_dir / "keyword.db"
        self._init_db()

    def _init_db(self):
        """Create FTS5 virtual table if it doesn't exist."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS keyword_index USING fts5("
                "  content, source, heading, chunk_idx, file_hash,"
                "  tokenize='porter unicode61'"
                ")"
            )
            conn.commit()
        finally:
            conn.close()

    def add_chunks(self, chunks: list[dict], file_hash: str):
        """Add chunk texts to the FTS5 index."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = []
            for c in chunks:
                rows.append((
                    c["text"],
                    c["source"],
                    c.get("heading", ""),
                    c["chunk_index"],
                    file_hash,
                ))
            conn.executemany(
                "INSERT INTO keyword_index (content, source, heading, chunk_idx, file_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def remove_source(self, source_path: str):
        """Remove all entries for a source from FTS5 index."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                "DELETE FROM keyword_index WHERE source = ?",
                (source_path,),
            )
            conn.commit()
        finally:
            conn.close()

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search FTS5 index with BM25 ranking.

        Returns list of {content, source, heading, score} where score is
        the BM25 relevance score normalized to 0-1 range.
        """
        if not query.strip():
            return []

        conn = sqlite3.connect(str(self.db_path))
        try:
            # Clean query for FTS5 (remove special chars, keep boolean operators)
            clean = re.sub(r'[^\w\s"-]', ' ', query)
            clean = ' AND '.join(clean.split()) if clean.strip() else query

            if not clean.strip():
                return []

            cursor = conn.execute(
                "SELECT content, source, heading, chunk_idx, rank "
                "FROM keyword_index "
                "WHERE keyword_index MATCH ? "
                "ORDER BY rank "
                "LIMIT ?",
                (clean, limit),
            )
            results = []
            for row in cursor:
                content, source, heading, chunk_idx, rank = row
                # BM25 rank is negative (0 = perfect match, more negative = worse)
                # Normalize to 0-1 score
                score = max(0.0, 1.0 + rank / 10.0)  # rank is typically -5 to 0
                results.append({
                    "content": content[:500] + "..." if len(content) > 500 else content,
                    "source": source,
                    "heading": heading,
                    "chunk": chunk_idx,
                    "score": round(min(1.0, score), 4),
                    "search_type": "keyword",
                })

            return results
        except Exception as e:
            # FTS5 might fail on complex queries — return empty
            return []
        finally:
            conn.close()

    def stats(self) -> dict:
        """Get keyword index statistics."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM keyword_index")
            count = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT COUNT(DISTINCT source) FROM keyword_index"
            )
            sources = cursor.fetchone()[0]

            return {
                "total_entries": count,
                "unique_sources": sources,
            }
        finally:
            conn.close()

    def count_for_source(self, source_path: str) -> int:
        """Count FTS5 entries for a specific source path."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM keyword_index WHERE source = ?",
                (source_path,),
            )
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def clear(self):
        """Clear the entire keyword index."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("DELETE FROM keyword_index")
            conn.commit()
        finally:
            conn.close()


# ─── RAG Engine ────────────────────────────────────────────────────────


class RagEngine:
    """ChromaDB-backed RAG engine with local embeddings + FTS5 hybrid search."""

    def __init__(self):
        config.data_dir.mkdir(parents=True, exist_ok=True)
        config.chroma_dir.mkdir(parents=True, exist_ok=True)

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

    @property
    def embedder(self):
        """Lazy-load the embedding model (it's ~80MB)."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(config.embedding_model)
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def index_file(self, file_path: str) -> dict:
        """Index a single file into ChromaDB + FTS5 keyword index."""
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}", "indexed": 0}

        # Read file
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"error": str(e), "indexed": 0}

        file_hash = hashlib.sha256(text.encode()).hexdigest()
        file_size = path.stat().st_size

        # Chunk text (fast — just string ops, no embeddings)
        chunks = _chunk_text(text, str(path))
        if not chunks:
            return {"file": file_path, "indexed": 0, "reason": "empty"}

        # Check if already indexed with same hash
        session = get_session()
        try:
            existing = (
                session.query(IndexedDocument)
                .filter(IndexedDocument.path == str(path))
                .first()
            )
            if existing and existing.file_hash == file_hash:
                # File unchanged — but FTS5 may be empty (e.g., indexed before
                # keyword search was added, or index was cleared).
                if self.keyword.count_for_source(str(path)) == 0:
                    self.keyword.add_chunks(chunks, file_hash)
                return {
                    "file": file_path,
                    "indexed": 0,
                    "reason": "unchanged",
                    "chunks": existing.chunk_count,
                }
        finally:
            session.close()

        chunk_texts = [c["text"] for c in chunks]

        # Create unique IDs
        ids = [f"{path.stem}-{c['chunk_index']}-{file_hash[:8]}" for c in chunks]

        # Generate embeddings
        try:
            embeddings = self._embed(chunk_texts)
        except Exception as e:
            return {"error": f"Embedding failed: {e}", "indexed": 0}

        # Metadata for each chunk
        metadatas = [
            {
                "source": str(path),
                "chunk": c["chunk_index"],
                "total": c["total_chunks"],
                "heading": c.get("heading", ""),
                "heading_level": str(c.get("heading_level", 0)),
                "file_type": path.suffix,
                "file_hash": file_hash,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            for c in chunks
        ]

        # Upsert to ChromaDB
        try:
            existing_ids = self.collection.get(
                where={"source": str(path)},
                include=[],
            )
            if existing_ids and existing_ids["ids"]:
                self.collection.delete(ids=existing_ids["ids"])
        except Exception:
            pass

        self.collection.add(
            ids=ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # Update FTS5 keyword index
        self.keyword.remove_source(str(path))
        self.keyword.add_chunks(chunks, file_hash)

        # Update tracking DB
        session = get_session()
        try:
            if existing:
                existing.chunk_count = len(chunks)
                existing.last_indexed = datetime.now(timezone.utc)
                existing.file_hash = file_hash
                existing.file_size = file_size
            else:
                doc = IndexedDocument(
                    id=str(uuid.uuid4()),
                    path=str(path),
                    title=path.stem,
                    doc_type=path.suffix.lstrip("."),
                    chunk_count=len(chunks),
                    file_hash=file_hash,
                    file_size=file_size,
                )
                session.add(doc)
            session.commit()
        finally:
            session.close()

        return {
            "file": file_path,
            "indexed": len(chunks),
            "chunks": len(chunks),
        }

    def index_notes(self) -> dict:
        """Index all markdown files in Amnis's notes directory."""
        notes_dir = Path(config.notes_dir)
        if not notes_dir.exists():
            notes_dir.mkdir(parents=True, exist_ok=True)
            return {"warning": f"Notes directory created at {notes_dir}", "indexed": 0, "files": 0}

        md_files = list(notes_dir.rglob("*.md")) + list(notes_dir.rglob("*.txt"))
        total_indexed = 0
        total_files = 0
        errors = []

        for f in md_files:
            # Skip hidden files
            rel = f.relative_to(notes_dir)
            if any(p.startswith(".") for p in rel.parts):
                continue

            result = self.index_file(str(f))
            if result.get("indexed", 0) > 0:
                total_indexed += result["indexed"]
                total_files += 1
            elif "error" in result:
                errors.append(result["error"])

        return {
            "files_indexed": total_files,
            "chunks_indexed": total_indexed,
            "total_found": len(md_files),
            "errors": errors,
        }

    def index_wiki(self) -> dict:
        """Index all markdown files in the wiki directory into ChromaDB + FTS5.

        This is the vault mechanism — drop .md files into data/wiki/ and they
        become searchable via semantic and keyword search. Run this after
        adding or editing wiki pages.
        """
        wiki_dir = Path(config.wiki_dir)
        if not wiki_dir.exists():
            wiki_dir.mkdir(parents=True, exist_ok=True)
            return {"warning": f"Wiki directory created at {wiki_dir}", "files_indexed": 0, "chunks_indexed": 0}

        # Also index the facts subdirectory (auto-created from memory store)
        facts_dir = Path(config.wiki_facts_dir)
        md_files = list(wiki_dir.rglob("*.md")) + list(wiki_dir.rglob("*.txt"))

        total_indexed = 0
        total_files = 0
        errors = []

        for f in md_files:
            # Skip hidden files and the index (we compile it separately)
            rel = f.relative_to(wiki_dir)
            if any(p.startswith(".") for p in rel.parts):
                continue
            # Skip the auto-generated index page — wiki compiler handles it
            if rel.name == "index.md" and len(rel.parts) == 1:
                continue

            result = self.index_file(str(f))
            if result.get("indexed", 0) > 0:
                total_indexed += result["indexed"]
                total_files += 1
            elif "error" in result:
                errors.append(result["error"])

        return {
            "files_indexed": total_files,
            "chunks_indexed": total_indexed,
            "total_found": len(md_files),
            "errors": errors,
        }

    def search(
        self,
        query: str,
        limit: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Semantic search over indexed documents.

        Args:
            query: Search query
            limit: Max results
            where: Optional ChromaDB metadata filter

        Returns:
            List of search results with content, source, score
        """
        if not query.strip():
            return []

        try:
            query_embedding = self._embed([query])[0]
        except Exception as e:
            return [{"error": f"Query embedding failed: {e}"}]

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            return [{"error": str(e)}]

        hits = []
        if results and results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                dist = results["distances"][0][i] if results.get("distances") else 0
                hits.append({
                    "content": doc[:500] + "..." if len(doc) > 500 else doc,
                    "source": meta.get("source", "unknown"),
                    "heading": meta.get("heading", ""),
                    "chunk": meta.get("chunk", 0),
                    "score": 1.0 - dist,  # cosine similarity
                    "file_type": meta.get("file_type", ""),
                    "search_type": "semantic",
                })

        return hits

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        semantic_weight: Optional[float] = None,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Hybrid search: semantic (ChromaDB) + keyword (FTS5) with RRF fusion.

        Uses Reciprocal Rank Fusion (RRF, k=60) to merge results from both
        retrieval systems. Documents that rank well in BOTH systems get a
        stronger combined score than documents that rank well in only one.

        Args:
            query: Search query
            limit: Max results in final merged list
            semantic_weight: Ignored (RRF replaces weighted fusion).
                             Kept for backward compatibility.
            where: Optional metadata filter for ChromaDB semantic leg.

        Returns:
            Merged and re-ranked search results using RRF
        """
        # Run both searches
        semantic_results = self.search(query, limit=limit * 2, where=where)
        keyword_results = self.keyword.search(query, limit=limit * 2)

        # If one failed or returned nothing, return the other
        if not semantic_results and not keyword_results:
            return []
        if not semantic_results:
            return keyword_results[:limit]
        if not keyword_results:
            return semantic_results[:limit]

        # RRF (Reciprocal Rank Fusion) — standard from the original paper
        # A document that ranks well in BOTH systems beats one that ranks
        # well in only one, even if that one rank is #1.
        k = 60
        score_map = {}

        for rank, r in enumerate(semantic_results):
            key = f"{r['source']}:{r.get('chunk', 0)}"
            if key not in score_map:
                score_map[key] = {"data": r, "sem_rank": None, "kw_rank": None}
            score_map[key]["sem_rank"] = rank + 1  # 1-indexed

        for rank, r in enumerate(keyword_results):
            key = f"{r['source']}:{r.get('chunk', 0)}"
            if key not in score_map:
                score_map[key] = {"data": r, "sem_rank": None, "kw_rank": None}
            score_map[key]["kw_rank"] = rank + 1

        merged = []
        for key, entry in score_map.items():
            rrf_score = 0.0
            if entry["sem_rank"]:
                rrf_score += 1.0 / (k + entry["sem_rank"])
            if entry["kw_rank"]:
                rrf_score += 1.0 / (k + entry["kw_rank"])

            entry["data"]["hybrid_score"] = round(rrf_score, 4)
            entry["data"]["rrf_sem_rank"] = entry["sem_rank"]
            entry["data"]["rrf_kw_rank"] = entry["kw_rank"]
            merged.append(entry["data"])

        merged.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return merged[:limit]

    def stats(self) -> dict:
        """Get RAG engine stats including both semantic and keyword indexes."""
        try:
            count = self.collection.count()
            all_meta = self.collection.get(include=["metadatas"])
            sources = set()
            if all_meta and all_meta.get("metadatas"):
                for m in all_meta["metadatas"]:
                    if "source" in m:
                        sources.add(m["source"])

            kw_stats = self.keyword.stats()

            return {
                "total_chunks": count,
                "unique_sources": len(sources),
                "embedding_model": config.embedding_model,
                "keyword_index": kw_stats,
            }
        except Exception as e:
            return {"error": str(e)}

    def delete_source(self, source_path: str) -> int:
        """Delete all chunks for a source from both indexes."""
        count = 0
        try:
            existing = self.collection.get(
                where={"source": source_path},
                include=[],
            )
            if existing and existing["ids"]:
                self.collection.delete(ids=existing["ids"])
                count = len(existing["ids"])
        except Exception:
            pass

        try:
            self.keyword.remove_source(source_path)
        except Exception:
            pass

        return count


# Singleton
engine = RagEngine()
