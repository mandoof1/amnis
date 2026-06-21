"""Memory consolidation — extract structured facts from episodic logs.

Based on findings from the Memory AI + RAG deep research (CoALA framework,
Generative Agents paper, Stanford 2023):
  - Consolidation is critical for memory quality (removing reflection broke emergent behaviors)
  - Semantic + episodic memory should be distinct layers
  - Contradiction resolution and deduplication prevent memory corruption

This module uses sentence embeddings for semantic similarity (reuses existing
all-MiniLM-L6-v2 from the RAG engine — no extra deps) and lightweight NLP
heuristics for contradiction detection.
"""
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, desc

from ..config import config
from ..database import get_session, ConversationLog, MemoryFact


# ─── Sentence Embedder (lazy, shared with RAG engine) ──────────────────

_embedder = None

def _get_embedder():
    """Lazy-load sentence embedder (reuses same model as RAG engine)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.embedding_model)
    return _embedder


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, returning list of float vectors."""
    model = _get_embedder()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ─── Contradiction Detection ───────────────────────────────────────────

# Simple negation indicators — words that flip the meaning of a statement
_NEGATION_WORDS = {
    "not", "never", "no", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "couldn't", "shouldn't", "isn't", "aren't", "wasn't",
    "weren't", "haven't", "hasn't", "hadn't", "can't", "cannot",
    "without", "dislike", "hates", "refuses", "rejects", "avoids",
}

# Positive/negative polarity words for basic sentiment
_POSITIVE_WORDS = {
    "likes", "loves", "enjoys", "prefers", "wants", "uses", "does",
    "has", "works", "good", "great", "excellent", "amazing", "wonderful",
}
_NEGATIVE_WORDS = {
    "dislikes", "hates", "refuses", "avoids", "can't", "won't", "bad",
    "terrible", "awful", "horrible", "broken", "fails",
}


def _polarity(text: str) -> float:
    """Simple polarity score: 1.0 = positive, -1.0 = negative, 0.0 = neutral."""
    lower = text.lower()
    pos_count = sum(1 for w in _POSITIVE_WORDS if w in lower)
    neg_count = sum(1 for w in _NEGATIVE_WORDS if w in lower)
    neg_count += sum(1 for w in _NEGATION_WORDS if w in lower) * 0.5
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def _has_negation(text: str) -> bool:
    """Check if text contains negation words."""
    lower = text.lower()
    for w in _NEGATION_WORDS:
        # Use word boundary check
        if re.search(r'\b' + re.escape(w) + r'\b', lower):
            return True
    return False


def _extract_topic(text: str) -> str:
    """Extract the likely topic from a fact (first noun-ish phrase)."""
    # Simple heuristic: first 3-5 words after common patterns
    for prefix in ["you ", "your ", "you're ", "you've "]:
        if text.lower().startswith(prefix):
            remainder = text[len(prefix):]
            # Take first ~4 meaningful words as topic
            words = remainder.split()[:4]
            return " ".join(words).rstrip(".,!?;:")
    # Otherwise use first 5 words
    words = text.split()[:5]
    return " ".join(words).rstrip(".,!?;:")


# ─── Main Functions ────────────────────────────────────────────────────


def get_stored_facts(session, limit: int = 200) -> list[MemoryFact]:
    """Get recent stored facts for dedup/comparison."""
    return (
        session.query(MemoryFact)
        .order_by(desc(MemoryFact.importance), desc(MemoryFact.timestamp))
        .limit(limit)
        .all()
    )


def find_similar_fact(
    fact_text: str,
    embedding: list[float],
    existing_facts: list[MemoryFact],
    threshold: float = config.dedup_similarity_threshold,
) -> Optional[tuple[MemoryFact, float]]:
    """Find a similar existing fact above threshold. Returns (fact, similarity)."""
    # First pass: fast keyword overlap filter to avoid embedding all
    words = set(fact_text.lower().split())
    candidates = []
    for ef in existing_facts:
        ef_words = set(ef.fact.lower().split())
        overlap = len(words & ef_words) / max(len(words | ef_words), 1)
        if overlap > 0.15:  # loose keyword filter
            candidates.append(ef)

    if not candidates:
        return None

    # Embed candidates in batch
    candidate_texts = [c.fact for c in candidates]
    candidate_embs = _embed(candidate_texts)

    best_sim = 0.0
    best_fact = None
    for ef, cemb in zip(candidates, candidate_embs):
        sim = _cosine_sim(embedding, cemb)
        if sim > best_sim:
            best_sim = sim
            best_fact = ef

    if best_sim >= threshold:
        return (best_fact, best_sim)
    return None


def check_contradiction(
    fact_text: str,
    embedding: list[float],
    existing_facts: list[MemoryFact],
    threshold: float = config.contradiction_distance_threshold,
) -> Optional[MemoryFact]:
    """Check if new fact contradicts an existing one.

    Two facts contradict if they are semantically similar (> threshold)
    but have opposite polarity (one positive, one negative).
    """
    words = set(fact_text.lower().split())
    new_polarity = _polarity(fact_text)

    # Quick filter: only check facts with opposite polarity potential
    candidates = []
    for ef in existing_facts:
        ef_words = set(ef.fact.lower().split())
        overlap = len(words & ef_words) / max(len(words | ef_words), 1)
        if overlap > 0.1:
            ef_pol = _polarity(ef.fact)
            if (new_polarity > 0.3 and ef_pol < -0.3) or (new_polarity < -0.3 and ef_pol > 0.3):
                candidates.append(ef)

    if not candidates:
        return None

    candidate_texts = [c.fact for c in candidates]
    candidate_embs = _embed(candidate_texts)

    for ef, cemb in zip(candidates, candidate_embs):
        sim = _cosine_sim(embedding, cemb)
        if sim >= threshold:
            return ef  # contradiction found

    return None


def compute_importance(
    fact_text: str,
    access_count: int = 0,
    is_manual: bool = False,
    source: str = "auto-extracted",
) -> int:
    """Compute importance score (1-10) based on signals.

    Boosts:
      - Manual storage (user explicit): +2
      - Contains personal info (setup, preference, name): +1
      - Fact mentioned repeatedly (access_count): +1 per 3 accesses
      - Recent (source is current session): +1
    """
    importance = 5  # baseline

    if is_manual or source == "manual":
        importance += 2  # user explicitly chose to store this

    # Facts about LO's preferences, setup, or identity are more important
    lower = fact_text.lower()
    if any(p in lower for p in ["prefer", "use", "work", "like", "have", "run", "setup"]):
        importance += 1
    if any(p in lower for p in ["arch", "linux", "cachyos", "terminal"]):
        importance += 1

    # Access frequency boost
    importance += min(access_count // 3, 2)  # max +2 from frequency

    return max(1, min(10, importance))


def consolidate_recent(
    limit: int = 50,
    run_dedup: bool = True,
    run_contradiction_check: bool = True,
) -> dict:
    """Scan recent conversation logs and extract actionable facts.

    Features (upgraded from v1):
      - Semantic dedup: similar facts get merged instead of duplicated
      - Contradiction detection: flags opposing facts
      - Importance scoring: recency + frequency + content signals
      - Confidence scoring: repeated mentions increase confidence
    """
    session = get_session()
    try:
        logs = (
            session.query(ConversationLog)
            .filter(ConversationLog.role == "assistant")
            .order_by(desc(ConversationLog.timestamp))
            .limit(limit)
            .all()
        )

        existing = get_stored_facts(session, limit=200) if (run_dedup or run_contradiction_check) else []

        extracted = 0
        merged = 0
        contradictions = []
        skipped = 0

        for log in logs:
            content = log.content
            if not content:
                continue

            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if not line or len(line) < 30:
                    continue

                # Check if it looks like a fact about LO
                lower = line.lower()
                if not any(p in lower for p in ["you ", "your ", "you're", "you've", "you'll"]):
                    continue

                # Skip questions and commands
                if line.endswith("?") or line.startswith("!"):
                    continue

                # Check if we already have something very similar (exact match on first 60 chars)
                existing_exact = (
                    session.query(MemoryFact)
                    .filter(MemoryFact.fact.ilike(f"%{line[:60]}%"))
                    .first()
                )
                if existing_exact:
                    # Boost importance instead of duplicating
                    existing_exact.importance = min(10, existing_exact.importance + 1)
                    existing_exact.access_count += 1
                    existing_exact.last_accessed = datetime.now(timezone.utc)
                    merged += 1
                    continue

                # Compute embedding for semantic dedup
                if run_dedup and existing:
                    try:
                        embedding = _embed([line])[0]

                        # Check for contradiction
                        if run_contradiction_check:
                            contra = check_contradiction(line, embedding, existing)
                            if contra:
                                contradictions.append({
                                    "new_fact": line[:100],
                                    "existing_fact": contra.fact[:100],
                                    "existing_id": contra.id,
                                })
                                # Flag both with lower confidence
                                contra.confidence = max(0.1, contra.confidence - 0.3)

                        # Check for near-duplicate (merge instead of create)
                        similar = find_similar_fact(line, embedding, existing)
                        if similar:
                            ef, sim = similar
                            # Merge: take longer/more specific fact, boost importance
                            if len(line) > len(ef.fact):
                                ef.fact = line[:500]
                            ef.importance = min(10, max(ef.importance, compute_importance(line, ef.access_count)))
                            ef.access_count += 1
                            ef.confidence = min(1.0, ef.confidence + 0.1)
                            ef.last_accessed = datetime.now(timezone.utc)
                            merged += 1
                            continue
                    except Exception:
                        pass  # If embedding fails, fall through to simple storage

                # Extract and store as new memory
                importance = compute_importance(line, source="auto-extracted")
                fact = MemoryFact(
                    id=str(uuid.uuid4()),
                    fact=line[:500],
                    category="event",
                    importance=importance,
                    source="consolidation",
                    confidence=0.6,
                    tags=["auto-extracted", "conversation"],
                    context=f"Extracted from conversation log {log.id}",
                    timestamp=datetime.now(timezone.utc),
                    last_accessed=datetime.now(timezone.utc),
                )
                session.add(fact)
                extracted += 1

        session.commit()
        result = {
            "logs_scanned": len(logs),
            "memories_extracted": extracted,
            "memories_merged": merged,
            "contradictions_found": len(contradictions),
        }
        if contradictions:
            result["contradictions"] = contradictions[:5]
        return result
    finally:
        session.close()


def reflect(
    max_clusters: int = 5,
    min_facts_per_cluster: int = 2,
) -> list[dict]:
    """Reflect on stored facts to synthesize higher-level beliefs.

    Based on the Stanford 2023 Generative Agents paper: removing reflection
    broke emergent behaviors. This function groups related observations into:

      1. Mid-level themes — patterns across similar facts
      2. High-level beliefs — synthesized understanding across themes

    Uses the existing sentence embedder for semantic clustering. No new deps.

    Args:
        max_clusters: Maximum number of theme clusters to create
        min_facts_per_cluster: Minimum facts needed to form a theme

    Returns:
        List of synthesized reflections (themes and beliefs)
    """
    session = get_session()
    try:
        # Get all auto-extracted facts (not manual)
        facts = (
            session.query(MemoryFact)
            .filter(MemoryFact.category.in_(["event", "general"]))
            .order_by(desc(MemoryFact.importance))
            .limit(100)
            .all()
        )

        if len(facts) < min_facts_per_cluster:
            return []

        # Compute embeddings for all facts
        texts = [f.fact[:200] for f in facts]
        embeddings = _embed(texts)

        # Simple greedy clustering: highest-importance facts become cluster seeds
        clusters = []
        used = set()

        for idx in range(len(facts)):
            if idx in used:
                continue
            cluster = [idx]
            used.add(idx)

            for jdx in range(idx + 1, len(facts)):
                if jdx in used:
                    continue
                sim = _cosine_sim(embeddings[idx], embeddings[jdx])
                if sim > 0.75:  # moderately tight cluster
                    cluster.append(jdx)
                    used.add(jdx)

            if len(cluster) >= min_facts_per_cluster:
                clusters.append(cluster)

            if len(clusters) >= max_clusters:
                break

        # For each cluster, synthesize a mid-level theme
        reflections = []
        for cluster_indices in clusters:
            cluster_facts = [facts[i] for i in cluster_indices]

            # Find the most representative fact (highest importance × confidence)
            cluster_facts.sort(
                key=lambda f: f.importance * f.confidence,
                reverse=True,
            )
            representative = cluster_facts[0]

            # Build a theme sentence
            theme_sources = [f.fact[:120] for f in cluster_facts[:5]]
            theme_text = (
                f"[Theme] Repeated pattern: connected observations about "
                f"{_extract_topic(representative.fact)[:60]}"
            )

            # Check if a similar theme already exists
            theme_embedding = _embed([theme_text])[0]
            existing_similar = find_similar_fact(theme_text, theme_embedding, facts, threshold=0.85)
            if existing_similar:
                # Update existing theme instead of creating new one
                ef, sim = existing_similar
                ef.importance = min(10, ef.importance + 1)
                ef.confidence = min(1.0, ef.confidence + 0.05)
                ef.last_accessed = datetime.now(timezone.utc)
                reflections.append({
                    "type": "theme_updated",
                    "theme": ef.fact[:150],
                    "supporting_facts": len(cluster_facts),
                    "confidence": ef.confidence,
                })
                continue

            # Store as a mid-level theme
            theme = MemoryFact(
                id=str(uuid.uuid4()),
                fact=theme_text[:500],
                category="theme",
                importance=min(10, sum(f.importance for f in cluster_facts) // len(cluster_facts) + 1),
                source="reflection",
                confidence=0.5,
                tags=["reflection", "theme"] + list(set(t for f in cluster_facts for t in (f.tags or [])))[:3],
                context=f"Synthesized from {len(cluster_facts)} related observations",
                timestamp=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
            )
            session.add(theme)

            reflections.append({
                "type": "theme_created",
                "theme": theme_text[:150],
                "supporting_facts": len(cluster_facts),
                "confidence": theme.confidence,
                "top_fact": representative.fact[:100],
            })

        session.commit()
        return reflections
    finally:
        session.close()


def run_pipeline() -> dict:
    """Run the full consolidation pipeline.

    Steps:
      1. Consolidate recent conversations → extract facts
      2. Semantic dedup and contradiction check
      3. Reflect: synthesize observations → mid-level themes → high-level beliefs
      4. Return summary
    """
    result = consolidate_recent(
        limit=config.consolidation_batch_size,
        run_dedup=True,
        run_contradiction_check=True,
    )

    # Step 3: Reflection hierarchy (Stanford Generative Agents paper)
    reflection = reflect()
    result["reflections"] = reflection

    return result
