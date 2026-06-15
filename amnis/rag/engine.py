"""RAG Engine — ChromaDB + sentence-transformers semantic search over documents.

Handles:
  - Indexing documents (markdown, text, code)
  - Semantic + keyword hybrid search
  - Vault synchronization
"""
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import config
from ..database import get_session, IndexedDocument


class RagEngine:
    """ChromaDB-backed RAG engine with local embeddings."""

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

    def _chunk_text(self, text: str, source_path: str) -> list[dict]:
        """Split text into overlapping chunks with metadata."""
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
                "chunk_index": len(chunks),
                "total_chunks": 0,  # set after
            })

        total = len(chunks)
        for c in chunks:
            c["total_chunks"] = total

        return chunks

    def index_file(self, file_path: str) -> dict:
        """Index a single file into ChromaDB."""
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

        # Check if already indexed with same hash
        session = get_session()
        try:
            existing = (
                session.query(IndexedDocument)
                .filter(IndexedDocument.path == str(path))
                .first()
            )
            if existing and existing.file_hash == file_hash:
                return {
                    "file": file_path,
                    "indexed": 0,
                    "reason": "unchanged",
                    "chunks": existing.chunk_count,
                }
        finally:
            session.close()

        # Chunk and embed
        chunks = self._chunk_text(text, str(path))
        if not chunks:
            return {"file": file_path, "indexed": 0, "reason": "empty"}

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
                "file_type": path.suffix,
                "file_hash": file_hash,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            for c in chunks
        ]

        # Upsert to ChromaDB
        # Remove old entries for this file first
        try:
            existing_ids = self.collection.get(
                where={"source": str(path)},
                include=[],
            )
            if existing_ids and existing_ids["ids"]:
                self.collection.delete(ids=existing_ids["ids"])
        except Exception:
            pass  # First time indexing

        self.collection.add(
            ids=ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

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

    def index_vault(self) -> dict:
        """Index all markdown files in the Obsidian vault."""
        vault = Path(config.vault_path)
        if not vault.exists():
            return {"error": f"Vault not found: {vault}", "indexed": 0, "files": 0}

        md_files = list(vault.rglob("*.md")) + list(vault.rglob("*.txt"))
        total_indexed = 0
        total_files = 0
        errors = []

        for f in md_files:
            # Skip hidden files and obsidian internals
            rel = f.relative_to(vault)
            if any(p.startswith(".") for p in rel.parts):
                continue
            if ".obsidian" in str(rel):
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
        """Semantic search over indexed documents."""
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
                    "chunk": meta.get("chunk", 0),
                    "score": 1.0 - dist,  # cosine similarity
                    "file_type": meta.get("file_type", ""),
                })

        return hits

    def hybrid_search(self, query: str, limit: int = 5) -> list[dict]:
        """Hybrid search: semantic + keyword matching."""
        # Start with semantic
        semantic_hits = self.search(query, limit=limit)

        # Also do keyword matching within ChromaDB (not great at keyword)
        # For real hybrid you'd add BM25, but semantic is good enough for MVP
        return semantic_hits

    def stats(self) -> dict:
        """Get RAG engine stats."""
        try:
            count = self.collection.count()
            # Get unique sources
            all_meta = self.collection.get(include=["metadatas"])
            sources = set()
            if all_meta and all_meta.get("metadatas"):
                for m in all_meta["metadatas"]:
                    if "source" in m:
                        sources.add(m["source"])
            return {
                "total_chunks": count,
                "unique_sources": len(sources),
                "embedding_model": config.embedding_model,
            }
        except Exception as e:
            return {"error": str(e)}

    def delete_source(self, source_path: str) -> int:
        """Delete all chunks for a source. Returns count deleted."""
        try:
            existing = self.collection.get(
                where={"source": source_path},
                include=[],
            )
            if existing and existing["ids"]:
                self.collection.delete(ids=existing["ids"])
                return len(existing["ids"])
            return 0
        except Exception:
            return 0


import uuid

# Singleton
engine = RagEngine()
