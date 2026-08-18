"""FTS5 keyword index + Reciprocal Rank Fusion.

Extracted from the RAG engine so the same machinery can back both document
chunk search and memory recall. Two behaviours differ from the 0.1 version:

* Queries are tokenised and each token is quoted before being handed to FTS5.
  The old code passed a lightly-scrubbed string straight into MATCH, so any
  query containing a bare ``AND``/``OR``/``NOT``/``*``/``:`` either changed
  meaning or raised, and the bare ``except`` hid it as "no results".
* ``search()`` returns the raw bm25 ``rank`` instead of inventing a 0-1 score
  from ``1.0 + rank/10``. bm25 is unbounded and corpus-dependent, so that
  formula clamped almost every real result to 0.0 — which then contributed
  nothing to fusion. Rank ordering is what fusion actually needs.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

from ..config import config

# Split on anything that is not a word character; FTS5 has its own tokenizer,
# we only need to stop operator characters from reaching the parser.
_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)


def build_match_query(query: str, operator: str = "OR") -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Every token is wrapped in double quotes (with internal quotes doubled), so
    reserved words and punctuation are treated as literals rather than syntax.
    """
    tokens = [t for t in _TOKEN_RE.split(query or "") if t]
    if not tokens:
        return ""
    quoted = ['"' + t.replace('"', '""') + '"' for t in tokens]
    return f" {operator} ".join(quoted)


class KeywordIndex:
    """A single FTS5 virtual table with a generic column set."""

    def __init__(
        self,
        table: str = "keyword_index",
        columns: Sequence[str] = ("content", "source", "heading", "chunk_idx", "file_hash"),
        db_path: Path | None = None,
    ):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError(f"Unsafe FTS table name: {table!r}")
        for col in columns:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", col):
                raise ValueError(f"Unsafe FTS column name: {col!r}")
        self.table = table
        self.columns = list(columns)
        self._db_path = db_path
        self._initialised = False

    # ─── plumbing ──────────────────────────────────────────────────────

    @property
    def db_path(self) -> Path:
        return self._db_path or config.keyword_db

    def _connect(self) -> sqlite3.Connection:
        path = self.db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        if not self._initialised:
            cols = ", ".join(self.columns)
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table} USING fts5("
                f"  {cols}, tokenize='porter unicode61')"
            )
            conn.commit()
            self._initialised = True
        return conn

    # ─── writes ────────────────────────────────────────────────────────

    def add(self, rows: Iterable[dict]) -> int:
        """Insert rows. Each dict is read using this index's column names."""
        payload = [tuple(str(r.get(c, "")) for c in self.columns) for r in rows]
        if not payload:
            return 0
        placeholders = ", ".join("?" for _ in self.columns)
        cols = ", ".join(self.columns)
        conn = self._connect()
        try:
            conn.executemany(f"INSERT INTO {self.table} ({cols}) VALUES ({placeholders})", payload)
            conn.commit()
        finally:
            conn.close()
        return len(payload)

    def remove_where(self, column: str, value: str) -> int:
        if column not in self.columns:
            raise ValueError(f"{column!r} is not a column of {self.table}")
        conn = self._connect()
        try:
            cur = conn.execute(f"DELETE FROM {self.table} WHERE {column} = ?", (value,))
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def clear(self) -> None:
        conn = self._connect()
        try:
            conn.execute(f"DELETE FROM {self.table}")
            conn.commit()
        finally:
            conn.close()

    # ─── reads ─────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 5, operator: str = "OR") -> list[dict]:
        """Return rows ordered by bm25 rank (most relevant first).

        Each result carries the raw ``rank`` and its 1-based ``rank_position``.
        No normalised score is fabricated — see the module docstring.
        """
        match = build_match_query(query, operator=operator)
        if not match:
            return []

        cols = ", ".join(self.columns)
        conn = self._connect()
        try:
            cursor = conn.execute(
                f"SELECT {cols}, rank FROM {self.table} WHERE {self.table} MATCH ? ORDER BY rank LIMIT ?",
                (match, limit),
            )
            results = []
            for position, row in enumerate(cursor, start=1):
                item = dict(zip(self.columns, row[:-1], strict=True))
                item["rank"] = row[-1]
                item["rank_position"] = position
                item["search_type"] = "keyword"
                results.append(item)
            return results
        except sqlite3.OperationalError:
            # A malformed MATCH is a programming error, not a user error;
            # build_match_query should have made this impossible.
            return []
        finally:
            conn.close()

    def count(self, column: str | None = None, value: str | None = None) -> int:
        conn = self._connect()
        try:
            if column is None:
                return conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
            if column not in self.columns:
                raise ValueError(f"{column!r} is not a column of {self.table}")
            return conn.execute(f"SELECT COUNT(*) FROM {self.table} WHERE {column} = ?", (value,)).fetchone()[
                0
            ]
        finally:
            conn.close()

    def stats(self, distinct_column: str = "source") -> dict:
        conn = self._connect()
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
            unique = 0
            if distinct_column in self.columns:
                unique = conn.execute(
                    f"SELECT COUNT(DISTINCT {distinct_column}) FROM {self.table}"
                ).fetchone()[0]
            return {"total_entries": total, "unique_sources": unique}
        finally:
            conn.close()


def rrf_fuse(rankings: list[list[str]], k: int | None = None) -> dict[str, float]:
    """Reciprocal Rank Fusion over several ranked ID lists.

    ``score(d) = sum over rankings of 1 / (k + rank(d))``

    RRF consumes *ranks*, never scores — that is the whole point of it. It lets
    a bm25 rank and a cosine distance be combined without needing them to share
    a scale, which the old ``semantic_weight`` blend silently assumed.
    """
    kk = config.rrf_k if k is None else k
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (kk + position)
    return scores
