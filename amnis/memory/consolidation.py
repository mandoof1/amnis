"""Consolidation — promote episodic logs into durable semantic facts.

Three substantive changes from 0.1:

* **Extraction is no longer "any line containing 'you'".** That heuristic
  swallowed instructions ("You should restart the daemon"), questions, list
  items and code, then stored them as permanent facts about the user. Lines
  now have to look like a declarative statement *about* the user, and an
  explicit reject list drops advice, conditionals and markup.
* **Similarity is vectorised.** The old ``_cosine_sim`` was a Python loop over
  384 floats, called O(n²) times, and re-embedded every stored fact for every
  candidate line. Facts are embedded once per run into a normalised matrix, so
  a comparison is one dot product.
* **``reflect()`` compares themes to themes.** It used to check a candidate
  theme against the observation pool, which by construction never contains
  themes — so the "already exists" branch was unreachable and every run added
  another near-identical theme.

``set_extractor()`` / ``set_synthesiser()` let a local LLM replace the
heuristics without touching this module.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import desc

from ..config import config
from ..database import ConversationLog, MemoryFact, session_scope

logger = logging.getLogger(__name__)

# ─── Embedding ─────────────────────────────────────────────────────────

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(config.embedding_model)
    return _embedder


def _embed(texts: list[str]) -> np.ndarray:
    """Embed texts as an L2-normalised float32 matrix.

    Normalising once means cosine similarity is a plain dot product, so the
    whole comparison collapses to a single matrix multiply.
    """
    if not texts:
        return np.zeros((0, config.embedding_dimension), dtype=np.float32)
    vectors = np.asarray(_get_embedder().encode(texts, show_progress_bar=False), dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _similarities(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one normalised vector against a normalised matrix."""
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    return matrix @ vector


# ─── Fact-shape heuristics ─────────────────────────────────────────────

# A line has to look like a statement *about the user* to be a candidate.
_FACT_PATTERNS = [
    re.compile(
        r"\byou (?:prefer|use|run|like|love|hate|work|own|have|need|want|always|never|usually)\b", re.I
    ),
    re.compile(r"\byou'?re\s+(?:a|an|the|using|running|working)\b", re.I),
    re.compile(r"\byou'?ve\s+(?:got|been|set|chosen)\b", re.I),
    re.compile(r"\byour\s+(?:\w+\s+){1,3}(?:is|are|was|were|uses|runs|lives|sits)\b", re.I),
]

# ...and must not look like advice, a question, a step, or markup.
_REJECT_PATTERNS = [
    re.compile(r"\byou (?:should|could|can|might|may|must|will|would|shall)\b", re.I),
    re.compile(r"^\s*(?:if|when|unless|whenever|once|after|before)\b", re.I),
    re.compile(r"\b(?:do|did|are|is|can|will|would|should|have|has) you\b", re.I),
    re.compile(r"^\s*[-*+>]\s"),  # markdown list item / quote
    re.compile(r"^\s*\d+[.)]\s"),  # numbered step
    re.compile(r"^\s*(?:```|~~~|\||#{1,6}\s)"),  # code fence, table, heading
    re.compile(r"^\s*(?:let me|i'?ll|i will|i can|here'?s|note that|for example)\b", re.I),
    re.compile(r"\b(?:TODO|FIXME|NOTE):", re.I),
]


def looks_like_fact(line: str) -> bool:
    """True if the line reads as a durable statement about the user."""
    line = line.strip()
    if len(line) < config.consolidation_min_line_length:
        return False
    if line.endswith("?") or line.endswith(":"):
        return False
    if any(p.search(line) for p in _REJECT_PATTERNS):
        return False
    return any(p.search(line) for p in _FACT_PATTERNS)


# Pluggable hooks — swap in an LLM without editing this module.
_extractor: Callable[[str], list[str]] | None = None
_synthesiser: Callable[[list[str]], str] | None = None


def set_extractor(fn: Callable[[str], list[str]] | None) -> None:
    """Replace fact extraction. ``fn(content) -> list[fact strings]``."""
    global _extractor
    _extractor = fn


def set_synthesiser(fn: Callable[[list[str]], str] | None) -> None:
    """Replace theme synthesis. ``fn(list[fact strings]) -> theme string``."""
    global _synthesiser
    _synthesiser = fn


def extract_candidates(content: str) -> list[str]:
    if _extractor is not None:
        return [c.strip() for c in _extractor(content) if c and c.strip()]
    return [line.strip() for line in (content or "").split("\n") if looks_like_fact(line)]


# ─── Polarity / contradiction ──────────────────────────────────────────

_NEGATION_WORDS = {
    "not",
    "never",
    "no",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "couldn't",
    "shouldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "haven't",
    "hasn't",
    "hadn't",
    "can't",
    "cannot",
    "without",
}
# Both base and third-person forms: consolidation sees "you love X" as often
# as "he loves X", and matching only one form halved the signal.
_POSITIVE_WORDS = {
    "like",
    "likes",
    "love",
    "loves",
    "enjoy",
    "enjoys",
    "prefer",
    "prefers",
    "want",
    "wants",
    "use",
    "uses",
    "has",
    "have",
    "work",
    "works",
    "good",
    "great",
    "excellent",
    "amazing",
    "wonderful",
}
_NEGATIVE_WORDS = {
    "dislike",
    "dislikes",
    "hate",
    "hates",
    "refuse",
    "refuses",
    "avoid",
    "avoids",
    "bad",
    "terrible",
    "awful",
    "horrible",
    "broken",
    "break",
    "fail",
    "fails",
}

_WORD_RE = re.compile(r"[a-z']+")


def _word_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _polarity(text: str) -> float:
    """-1.0 negative … 0.0 neutral … 1.0 positive.

    Uses whole-word membership. The old substring test made "has" match
    "hash", "no" match "notes", and "does" match "doesn't" — so polarity was
    close to noise.
    """
    words = _word_set(text)
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS) + 0.5 * len(words & _NEGATION_WORDS)
    total = pos + neg
    return 0.0 if total == 0 else (pos - neg) / total


def _has_negation(text: str) -> bool:
    return bool(_word_set(text) & _NEGATION_WORDS)


def _extract_topic(text: str) -> str:
    for prefix in ("you ", "your ", "you're ", "you've "):
        if text.lower().startswith(prefix):
            return " ".join(text[len(prefix) :].split()[:4]).rstrip(".,!?;:")
    return " ".join(text.split()[:5]).rstrip(".,!?;:")


def compute_importance(
    fact_text: str,
    access_count: int = 0,
    is_manual: bool = False,
    source: str = "auto-extracted",
) -> int:
    """Score 1-10 from explicitness, content signals, and access frequency."""
    importance = 5
    if is_manual or source == "manual":
        importance += 2

    words = _word_set(fact_text)
    if words & {
        "prefer",
        "prefers",
        "use",
        "uses",
        "work",
        "works",
        "like",
        "likes",
        "have",
        "has",
        "run",
        "runs",
        "setup",
    }:
        importance += 1
    # Domain keywords are configuration, not code. 0.1 hardcoded one user's
    # personal stack here, which scored every other user's memories wrongly.
    if config.importance_keywords and words & {k.lower() for k in config.importance_keywords}:
        importance += 1

    importance += min(access_count // 3, 2)
    return max(1, min(10, importance))


# ─── Main passes ───────────────────────────────────────────────────────


def _load_existing(session, limit: int = 300) -> tuple[list[MemoryFact], np.ndarray]:
    facts = (
        session.query(MemoryFact)
        .order_by(desc(MemoryFact.importance), desc(MemoryFact.timestamp))
        .limit(limit)
        .all()
    )
    matrix = _embed([f.fact[:400] for f in facts]) if facts else _embed([])
    return facts, matrix


def consolidate_recent(
    limit: int = 50,
    run_dedup: bool = True,
    run_contradiction_check: bool = True,
) -> dict:
    """Scan recent assistant turns and promote durable facts into memory."""
    extracted = 0
    merged = 0
    skipped = 0
    contradictions: list[dict] = []
    new_ids: list[tuple[str, str, str, list]] = []

    with session_scope() as session:
        logs = (
            session.query(ConversationLog)
            .filter(ConversationLog.role == "assistant")
            .order_by(desc(ConversationLog.timestamp))
            .limit(max(1, limit))
            .all()
        )

        existing: list[MemoryFact] = []
        matrix = _embed([])
        if run_dedup or run_contradiction_check:
            existing, matrix = _load_existing(session)

        seen_lines: set[str] = set()

        for log in logs:
            for line in extract_candidates(log.content or ""):
                normalised = line.lower()
                if normalised in seen_lines:
                    skipped += 1
                    continue
                seen_lines.add(normalised)

                if not (run_dedup or run_contradiction_check) or not existing:
                    fact = _new_fact(line, log.id)
                    session.add(fact)
                    new_ids.append((fact.id, fact.fact, fact.category, fact.tags))
                    extracted += 1
                    continue

                try:
                    vector = _embed([line])[0]
                except Exception as exc:  # noqa: BLE001 - fall back to plain store
                    logger.warning("Embedding failed during consolidation: %s", exc)
                    fact = _new_fact(line, log.id)
                    session.add(fact)
                    new_ids.append((fact.id, fact.fact, fact.category, fact.tags))
                    extracted += 1
                    continue

                sims = _similarities(vector, matrix)
                best = int(np.argmax(sims)) if sims.size else -1
                best_sim = float(sims[best]) if best >= 0 else 0.0

                if run_contradiction_check and best >= 0:
                    contra_idx = _find_contradiction(line, sims, existing)
                    if contra_idx is not None:
                        contra = existing[contra_idx]
                        contradictions.append(
                            {
                                "new_fact": line[:100],
                                "existing_fact": contra.fact[:100],
                                "existing_id": contra.id,
                            }
                        )
                        contra.confidence = max(0.1, (contra.confidence or 1.0) - 0.3)

                if run_dedup and best_sim >= config.dedup_similarity_threshold:
                    ef = existing[best]
                    if len(line) > len(ef.fact):
                        ef.fact = line[:500]
                    ef.importance = min(
                        10,
                        max(
                            ef.importance or 0,
                            compute_importance(line, ef.access_count or 0),
                        ),
                    )
                    ef.access_count = (ef.access_count or 0) + 1
                    ef.confidence = min(1.0, (ef.confidence or 0.6) + 0.1)
                    ef.last_accessed = datetime.now(UTC)
                    merged += 1
                    continue

                fact = _new_fact(line, log.id)
                session.add(fact)
                new_ids.append((fact.id, fact.fact, fact.category, fact.tags))
                extracted += 1

                # Append to the comparison pool so two near-identical lines in
                # the *same* run dedupe against each other. Previously the pool
                # was a snapshot, so one run could insert 40 copies of a line.
                existing.append(fact)
                matrix = np.vstack([matrix, vector.reshape(1, -1)]) if matrix.size else vector.reshape(1, -1)

    from .store import _index_memory_keywords

    for memory_id, text, category, tags in new_ids:
        _index_memory_keywords(memory_id, text, category, tags or [])

    result = {
        "logs_scanned": len(logs),
        "memories_extracted": extracted,
        "memories_merged": merged,
        "candidates_skipped": skipped,
        "contradictions_found": len(contradictions),
    }
    if contradictions:
        result["contradictions"] = contradictions[:5]
    return result


def _find_contradiction(line: str, sims: np.ndarray, existing: list[MemoryFact]) -> int | None:
    """Index of a near-identical fact with opposite polarity, if any."""
    new_pol = _polarity(line)
    if abs(new_pol) < 0.3:
        return None
    threshold = config.contradiction_distance_threshold
    for idx in np.argsort(-sims)[:10]:
        if float(sims[idx]) < threshold:
            break
        other_pol = _polarity(existing[int(idx)].fact)
        if (new_pol > 0.3 and other_pol < -0.3) or (new_pol < -0.3 and other_pol > 0.3):
            return int(idx)
    return None


def _new_fact(line: str, log_id: str) -> MemoryFact:
    return MemoryFact(
        id=str(uuid.uuid4()),
        fact=line[:500],
        category="event",
        importance=compute_importance(line, source="auto-extracted"),
        source="consolidation",
        confidence=0.6,
        tags=["auto-extracted", "conversation"],
        context=f"Extracted from conversation log {log_id}",
        timestamp=datetime.now(UTC),
        last_accessed=datetime.now(UTC),
        access_count=0,
    )


def reflect(max_clusters: int = 5, min_facts_per_cluster: int = 2) -> list[dict]:
    """Cluster observations into mid-level themes (Generative Agents, 2023)."""
    reflections: list[dict] = []
    new_themes: list[tuple[str, str, str, list]] = []

    with session_scope() as session:
        facts = (
            session.query(MemoryFact)
            .filter(MemoryFact.category.in_(["event", "general"]))
            .order_by(desc(MemoryFact.importance))
            .limit(100)
            .all()
        )
        if len(facts) < min_facts_per_cluster:
            return []

        matrix = _embed([f.fact[:200] for f in facts])

        clusters: list[list[int]] = []
        used: set[int] = set()
        for idx in range(len(facts)):
            if idx in used:
                continue
            sims = _similarities(matrix[idx], matrix)
            cluster = [idx] + [
                j for j in range(idx + 1, len(facts)) if j not in used and float(sims[j]) > 0.75
            ]
            used.update(cluster)
            if len(cluster) >= min_facts_per_cluster:
                clusters.append(cluster)
            if len(clusters) >= max_clusters:
                break

        if not clusters:
            return []

        # Compare candidate themes against *themes*, not against the
        # observation pool they were derived from.
        themes = (
            session.query(MemoryFact)
            .filter(MemoryFact.category == "theme")
            .order_by(desc(MemoryFact.timestamp))
            .limit(200)
            .all()
        )
        theme_matrix = _embed([t.fact[:200] for t in themes]) if themes else _embed([])

        for cluster_indices in clusters:
            cluster_facts = sorted(
                (facts[i] for i in cluster_indices),
                key=lambda f: (f.importance or 0) * (f.confidence or 0.0),
                reverse=True,
            )
            representative = cluster_facts[0]

            if _synthesiser is not None:
                theme_text = _synthesiser([f.fact for f in cluster_facts])
            else:
                theme_text = (
                    "[Theme] Repeated pattern: connected observations about "
                    f"{_extract_topic(representative.fact)[:60]}"
                )

            theme_vector = _embed([theme_text])[0]
            sims = _similarities(theme_vector, theme_matrix)
            if sims.size and float(sims.max()) >= 0.85:
                existing_theme = themes[int(np.argmax(sims))]
                existing_theme.importance = min(10, (existing_theme.importance or 5) + 1)
                existing_theme.confidence = min(1.0, (existing_theme.confidence or 0.5) + 0.05)
                existing_theme.last_accessed = datetime.now(UTC)
                reflections.append(
                    {
                        "type": "theme_updated",
                        "theme": existing_theme.fact[:150],
                        "supporting_facts": len(cluster_facts),
                        "confidence": existing_theme.confidence,
                    }
                )
                continue

            cluster_tags = sorted({t for f in cluster_facts for t in (f.tags or [])})[:3]
            theme = MemoryFact(
                id=str(uuid.uuid4()),
                fact=theme_text[:500],
                category="theme",
                importance=min(
                    10,
                    sum(f.importance or 0 for f in cluster_facts) // len(cluster_facts) + 1,
                ),
                source="reflection",
                confidence=0.5,
                tags=["reflection", "theme", *cluster_tags],
                context=f"Synthesized from {len(cluster_facts)} related observations",
                timestamp=datetime.now(UTC),
                last_accessed=datetime.now(UTC),
                access_count=0,
            )
            session.add(theme)
            themes.append(theme)
            theme_matrix = (
                np.vstack([theme_matrix, theme_vector.reshape(1, -1)])
                if theme_matrix.size
                else theme_vector.reshape(1, -1)
            )
            new_themes.append((theme.id, theme.fact, theme.category, theme.tags))

            reflections.append(
                {
                    "type": "theme_created",
                    "theme": theme_text[:150],
                    "supporting_facts": len(cluster_facts),
                    "confidence": theme.confidence,
                    "top_fact": representative.fact[:100],
                }
            )

    from .store import _index_memory_keywords

    for memory_id, text, category, tags in new_themes:
        _index_memory_keywords(memory_id, text, category, tags or [])

    return reflections


def run_pipeline() -> dict:
    """Extract facts from recent logs, then reflect over the result."""
    result = consolidate_recent(
        limit=config.consolidation_batch_size,
        run_dedup=True,
        run_contradiction_check=True,
    )
    result["reflections"] = reflect()
    return result
